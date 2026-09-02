"""
Pipeline:
    labeled data
        -> normal YOLO11 supervised loss
    unlabeled data
        -> EMA teacher on weak view
        -> confidence-filtered pseudo labels
        -> student on strong view
        -> boundary-wise Listen2Student uncertainty gating
        -> unsupervised classification + localization loss
    student -> EMA teacher
"""

from __future__ import annotations

import copy
import random
import json
import yaml
import io
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional
from types import SimpleNamespace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2 as T
from torchvision import tv_tensors
from torchvision.ops import batched_nms
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


import cv2
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class UTv2Config:
    epochs: int = 50
    batch_size: int = 8
    val_batch_size: int = 32
    img_size: int = 640
    lr: float = 1e-3
    weight_decay: float = 1e-2

    ema_momentum: float = 0.999

    pseudo_conf: float = 0.4
    pseudo_nms_iou: float = 0.5
    uncertainty_margin: float = 0.10
    student_uncertainty_min: float = 0.0

    supervised_weight: float = 0.5
    unsupervised_cls_weight: float = 1.0
    unsupervised_box_weight: float = 1.0
    uncertainty_weight: float = 0.1

    unsupervised_weight: float = 3.0
    warmup_epochs: int = 3
    unsup_ramp_epochs: int = 5
    early_stopping_min_delta: float = 1e-3

    max_det: int = 25


# ============================================================================
# Model
# ============================================================================

def convert_to_standard_yolo(
    model: DetectionModel,
) -> DetectionModel:

    model = copy.deepcopy(model)

    old_head = model.model[-1]

    if not isinstance(old_head, UTv2Detect):
        return model

    new_head = Detect(
        nc=old_head.nc,
        reg_max=old_head.reg_max,
        end2end=getattr(old_head, "end2end", False),
        ch=infer_detect_channels(old_head),
    )

    old_state = old_head.state_dict()
    new_state = new_head.state_dict()

    for key, value in old_state.items():
        if key.startswith("cv_uncertainty."):
            continue

        if key in new_state and new_state[key].shape == value.shape:
            new_state[key].copy_(value)

    new_head.load_state_dict(
        new_state,
        strict=False,
    )

    for attr in (
        "f",
        "i",
        "type",
        "np",
        "stride",
        "legacy",
        "inplace",
    ):
        if hasattr(old_head, attr):
            setattr(new_head, attr, getattr(old_head, attr))

    model.model[-1] = new_head

    return model

class UTv2Detect(Detect):
    """YOLO Detect head extended with a 4-channel localization uncertainty head."""

    def __init__(self, nc=80, reg_max=16, end2end=False, ch=()):
        super().__init__(
            nc=nc,
            reg_max=reg_max,
            end2end=end2end,
            ch=ch,
        )

        # Defensive: compute intermediate channel size from first head
        # channel if available; otherwise fall back to a small constant.
        if ch and len(ch) > 0:
            c = max(int(ch[0]) // 4, 4)
        else:
            c = 4

        self.cv_uncertainty = nn.ModuleList(
            nn.Sequential(
                Conv(int(ci), c, 3),
                Conv(c, c, 3),
                nn.Conv2d(c, 4, 1),
            )
            for ci in ch
        )

    def forward_head(self, x, **kwargs):
        preds = super().forward_head(x, **kwargs)

        if not isinstance(preds, dict):
            return preds

        uncertainty = torch.cat(
            [
                self.cv_uncertainty[i](x[i]).flatten(2)
                for i in range(self.nl)
            ],
            dim=2,
        )

        # Positive sigma for L/T/R/B.
        preds["uncertainty"] = F.softplus(uncertainty) + 1e-6

        return preds


def infer_detect_channels(head: Detect) -> tuple[int, ...]:
    """Recover the input channels used by an existing Detect head."""

    ch = getattr(head, "ch", None)
    if ch:
        ch_tuple = tuple(int(c) for c in ch)
        if ch_tuple:
            return ch_tuple

    layers = getattr(head, "cv2", None)
    if layers is not None:
        inferred = []
        for layer in layers:
            first = layer[0] if isinstance(layer, nn.Sequential) else layer
            conv = getattr(first, "conv", None)
            if conv is not None and hasattr(conv, "in_channels"):
                inferred.append(int(conv.in_channels))
        if inferred:
            return tuple(inferred)

    raise ValueError("Unable to infer Detect input channels for UTv2 replacement.")

def replace_with_utv2(model: DetectionModel) -> DetectionModel:
    """Replace YOLO11 Detect or Segment while retaining all original YOLO weights."""

    old = model.model[-1]

    if not isinstance(old, Detect):
        raise TypeError(f"Expected Detect as final module, got {type(old)}")

    new = UTv2Detect(
        nc=old.nc,
        reg_max=old.reg_max,
        end2end=getattr(old, "end2end", False),
        ch=infer_detect_channels(old),
    )

    old_state = old.state_dict()
    new_state = new.state_dict()

    for key, value in old_state.items():
        if key in new_state and new_state[key].shape == value.shape:
            new_state[key].copy_(value)

    new.load_state_dict(new_state, strict=False)

    # Preserve the module metadata expected by Ultralytics' model forward path.
    for attr in ("f", "i", "type", "np", "stride", "legacy", "inplace"):
        if hasattr(old, attr):
            setattr(new, attr, getattr(old, attr))

    model.model[-1] = new

    return model


def build_model(weights: str, device: torch.device):
    yolo = YOLO(weights)
    cfg = yolo.model.args
    model = replace_with_utv2(yolo.model)
    return model.to(device), cfg


def find_detect(model: nn.Module) -> Detect:
    for module in reversed(list(model.modules())):
        if isinstance(module, Detect):
            return module
    raise RuntimeError("Detect module not found.")


# ============================================================================
# EMA
# ============================================================================

@torch.no_grad()
def update_ema(
    teacher: nn.Module,
    student: nn.Module,
    momentum: float,
):
    for teacher_p, student_p in zip(
        teacher.parameters(),
        student.parameters(),
    ):
        teacher_p.data.mul_(momentum).add_(
            student_p.data,
            alpha=1.0 - momentum,
        )

    # Keep BatchNorm / other buffers synchronized.
    for teacher_b, student_b in zip(
        teacher.buffers(),
        student.buffers(),
    ):
        teacher_b.copy_(student_b)


def freeze(model: nn.Module):
    for p in model.parameters():
        p.requires_grad_(False)


# ============================================================================
# Geometry / decoding
# ============================================================================

def xyxy_to_ltrb(
    boxes: torch.Tensor,
    anchors: torch.Tensor,
):
    """
    boxes:   [B,N,4] or [N,4], xyxy
    anchors: [N,2]
    """
    x = anchors[:, 0]
    y = anchors[:, 1]

    if boxes.ndim == 3:
        return torch.stack(
            (
                x[None] - boxes[..., 0],
                y[None] - boxes[..., 1],
                boxes[..., 2] - x[None],
                boxes[..., 3] - y[None],
            ),
            dim=-1,
        )

    return torch.stack(
        (
            x - boxes[..., 0],
            y - boxes[..., 1],
            boxes[..., 2] - x,
            boxes[..., 3] - y,
        ),
        dim=-1,
    )


@torch.no_grad()
def decode_predictions(
    model: DetectionModel,
    preds: Dict[str, torch.Tensor] | tuple[torch.Tensor, Dict[str, torch.Tensor]],
):
    """
    Decode raw training predictions.

    Returns:
        boxes       [B,N,4]   stride-scaled xyxy
        scores      [B,N,nc]  probabilities
        uncertainty [B,N,4]
        anchors     [N,2]
        stride      [N,1]
    """

    head = find_detect(model)

    if isinstance(preds, tuple):
        preds = preds[1]

    pred_distri = preds["boxes"].contiguous()

    scores = (
        preds["scores"]
        .contiguous()
        .sigmoid()
    )

    uncertainty = (
        preds["uncertainty"]
        .contiguous()
    )

    anchors, stride = make_anchors(
        preds["feats"],
        head.stride,
        0.5,
    )

    decode_anchors = anchors.transpose(0, 1)
    decode_stride = stride.transpose(0, 1)

    boxes = head.decode_bboxes(
        head.dfl(pred_distri),
        decode_anchors.unsqueeze(0),
        xywh=False,
    )

    boxes = (boxes * decode_stride).permute(0, 2, 1).contiguous()
    scores = scores.permute(0, 2, 1).contiguous()
    uncertainty = uncertainty.permute(0, 2, 1).contiguous()

    return {
        "boxes": boxes,
        "scores": scores,
        "uncertainty": uncertainty,
        "anchors": anchors,
        "stride": stride,
    }


# ============================================================================
# UTv2 detection loss
# ============================================================================

def focal_loss(pred_scores, target_scores, gamma=2.0, alpha=0.25):
    bce = F.binary_cross_entropy_with_logits(pred_scores, target_scores, reduction="none")
    p = pred_scores.sigmoid()
    p_t = p * target_scores + (1 - p) * (1 - target_scores)
    loss = bce * ((1 - p_t) ** gamma)
    alpha_t = alpha * target_scores + (1 - alpha) * (1 - target_scores)
    return loss * alpha_t

class UTv2DetectionLoss(v8DetectionLoss):
    """
    Existing Ultralytics YOLO detection loss + supervised uncertainty NLL.
    """

    def __init__(
        self,
        model,
        uncertainty_weight=1.0,
    ):
        super().__init__(model)
        self.uncertainty_weight = uncertainty_weight

        if isinstance(self.hyp, dict):
            self.hyp = SimpleNamespace(**self.hyp)
        self.hyp.box = 7.5
        self.hyp.cls = 0.5
        self.hyp.dfl = 1.5

    def __call__(
        self,
        preds,
        batch,
    ):
        return self.loss(preds, batch)

    def loss(
        self,
        preds,
        batch,
    ):
        """
        Native Ultralytics detection loss + UTv2 uncertainty loss.

        The standard YOLO box/classification/DFL losses are delegated directly
        to v8DetectionLoss to preserve pretrained-model behaviour.
        """

        # --------------------------------------------------------------
        # 1. Native YOLO loss — DO NOT reimplement this part
        # --------------------------------------------------------------

        total_loss, loss_items = super().loss(preds, batch)
        if total_loss.ndim != 0:
            total_loss = total_loss.sum()

        # No uncertainty prediction -> behave exactly like native YOLO.
        if not isinstance(preds, dict) or "uncertainty" not in preds:
            return total_loss, loss_items

        # --------------------------------------------------------------
        # 2. Build assignments only for the uncertainty branch
        # --------------------------------------------------------------

        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous()
        pred_sigma = preds["uncertainty"].permute(0, 2, 1).contiguous()

        anchor_points, stride_tensor = make_anchors(
            preds["feats"],
            self.stride,
            0.5,
        )

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]

        imgsz = (
            torch.tensor(
                preds["feats"][0].shape[2:],
                device=self.device,
                dtype=dtype,
            )
            * self.stride[0]
        )

        targets = torch.cat(
            (
                batch["batch_idx"].view(-1, 1),
                batch["cls"].view(-1, 1),
                batch["bboxes"],
            ),
            dim=1,
        )

        targets = self.preprocess(
            targets.to(self.device),
            batch_size,
            scale_tensor=imgsz[[1, 0, 1, 0]],
        )

        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(
            anchor_points,
            pred_distri,
        )

        (
            _,
            target_bboxes,
            target_scores,
            fg_mask,
            _,
        ) = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # --------------------------------------------------------------
        # 3. UTv2 uncertainty loss
        # --------------------------------------------------------------
        uncertainty_loss = total_loss.new_zeros(())

        if fg_mask.any():

            uncertainty_loss = self.uncertainty_nll(
                pred_bboxes,
                target_bboxes / stride_tensor,
                pred_sigma,
                anchor_points,
                fg_mask,
            )

            uncertainty_loss *= self.uncertainty_weight

        # --------------------------------------------------------------
        # 4. Native YOLO loss + training-only uncertainty term
        # --------------------------------------------------------------
        #
        # v8DetectionLoss returns total_loss already scaled by batch size.
        # Scale our per-batch uncertainty loss consistently.
        total_loss = total_loss + uncertainty_loss * batch_size
        loss_items["uncertainty_loss"] = uncertainty_loss.detach()

        return total_loss, loss_items

    @staticmethod
    def uncertainty_nll(
        pred_boxes,
        target_boxes,
        sigma,
        anchors,
        fg_mask,
    ):
        pred_ltrb = xyxy_to_ltrb(
            pred_boxes,
            anchors,
        )

        target_ltrb = xyxy_to_ltrb(
            target_boxes,
            anchors,
        )

        sigma = sigma.clamp(1e-2, 5.0)

        nll = (
            (pred_ltrb - target_ltrb.detach()).square()
            / (2.0 * sigma.square())
            + torch.log(sigma)
        ).sum(dim=-1)

        """iou = bbox_iou(
            pred_boxes,
            target_boxes,
            xywh=False,
            CIoU=False,
        ).clamp_min(0.0).squeeze(-1)
        weight = iou * fg_mask.float()"""

        weight = fg_mask.float()

        return (
            nll * weight
        ).sum() / weight.sum().clamp_min(1.0)

    def assign_targets(
        self,
        preds,
        batch,
    ):
        pred_distri = preds["boxes"].permute(
            0, 2, 1
        ).contiguous()

        pred_scores = preds["scores"].permute(
            0, 2, 1
        ).contiguous()

        anchor_points, stride_tensor = make_anchors(
            preds["feats"],
            self.stride,
            0.5,
        )

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]

        imgsz = (
            torch.tensor(
                preds["feats"][0].shape[2:],
                device=self.device,
                dtype=dtype,
            )
            * self.stride[0]
        )

        # ----------------------------------------------------------
        # Pad teacher uncertainty exactly like YOLO preprocess()
        # pads GT/pseudo targets.
        #
        # batch["uncertainty"]:
        #     [total_pseudo, 4]
        #
        # padded_teacher_sigma:
        #     [B, max_pseudo_per_image, 4]
        # ----------------------------------------------------------

        teacher_sigma_flat = batch["uncertainty"].to(
            self.device
        )

        if batch["batch_idx"].numel() > 0:
            counts = torch.bincount(
                batch["batch_idx"],
                minlength=batch_size,
            )
            max_gt = int(counts.max().item())
        else:
            max_gt = 0

        padded_teacher_sigma = torch.full(
            (batch_size, max_gt, 4),
            float("inf"),
            device=self.device,
            dtype=teacher_sigma_flat.dtype,
        )

        if max_gt > 0:
            gt_positions = torch.zeros(
                batch_size,
                dtype=torch.long,
                device=self.device,
            )

            for i in range(batch["batch_idx"].numel()):
                b = int(batch["batch_idx"][i].item())
                pos = gt_positions[b]

                padded_teacher_sigma[
                    b,
                    pos,
                ] = teacher_sigma_flat[i]

                gt_positions[b] += 1

        # ----------------------------------------------------------
        # Normal YOLO target preprocessing
        # ----------------------------------------------------------

        targets = torch.cat(
            (
                batch["batch_idx"].view(-1, 1),
                batch["cls"].view(-1, 1),
                batch["bboxes"],
            ),
            dim=1,
        )

        targets = self.preprocess(
            targets.to(self.device),
            batch_size,
            scale_tensor=imgsz[[1, 0, 1, 0]],
        )

        gt_labels, gt_bboxes = targets.split(
            (1, 4),
            2,
        )

        mask_gt = gt_bboxes.sum(
            2,
            keepdim=True,
        ).gt_(0.0)

        pred_bboxes = self.bbox_decode(
            anchor_points,
            pred_distri,
        )

        (
            target_labels,
            target_bboxes,
            target_scores,
            fg_mask,
            target_gt_idx,
        ) = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(
                gt_bboxes.dtype
            ),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        return {
            "pred_distri": pred_distri,
            "pred_scores": pred_scores,
            "pred_bboxes": pred_bboxes,
            "anchor_points": anchor_points,
            "stride_tensor": stride_tensor,
            "target_bboxes": target_bboxes,
            "target_scores": target_scores,
            "fg_mask": fg_mask,
            "imgsz": imgsz,
            "target_gt_idx": target_gt_idx,
            "padded_teacher_sigma": padded_teacher_sigma,
        }

# ============================================================================
# Listen2Student
# ============================================================================

def listen2student_mask(
    teacher_sigma: torch.Tensor,
    student_sigma: torch.Tensor,
    fg_mask: torch.Tensor,
    margin: float = 0.0,
    student_min: float = 0.0,
):
    """Select assigned boundaries where the teacher is sufficiently more certain."""
    finite = (
        torch.isfinite(teacher_sigma)
        & torch.isfinite(student_sigma)
    )
    teacher_more_certain = teacher_sigma * (1.0 + margin) < student_sigma
    student_is_valid = student_sigma > student_min

    return (
        teacher_more_certain
        & student_is_valid
        & finite
        & fg_mask.unsqueeze(-1)
    )

# ============================================================================
# Pseudo labels
# ============================================================================

@torch.no_grad()
def pseudo_labels(
    decoded: Dict[str, torch.Tensor],
    confidence: float,
    iou_threshold: float = 0.5,
    min_keep: int = 1,
    max_det: int = 300
):
    """
    Convert dense teacher predictions into sparse pseudo detections.

    Returns a list with one dictionary per batch image:
        boxes       [K, 4]
        classes     [K]
        confidence  [K]
        uncertainty [K, 4]
        indices     [K]  original dense prediction indices
    """
    scores = decoded["scores"]
    boxes = decoded["boxes"]
    uncertainty = decoded["uncertainty"]

    batch_pseudo = []

    for b in range(boxes.shape[0]):
        conf, cls = scores[b].max(dim=-1)

        keep = conf >= confidence

        if not keep.any() and min_keep > 0:
            # fallback: keep top-min_keep highest-confidence boxes even if below threshold
            topk = min(min_keep, conf.numel())
            keep = torch.zeros_like(conf, dtype=torch.bool)
            keep[conf.topk(topk).indices] = True
        elif not keep.any():
            batch_pseudo.append({
                "boxes": boxes.new_zeros((0, 4)),
                "classes": cls.new_zeros((0,), dtype=torch.long),
                "confidence": conf.new_zeros((0,)),
                "uncertainty": uncertainty.new_zeros((0, 4)),
                "indices": cls.new_zeros((0,), dtype=torch.long),
            })
            continue

        candidate_indices = torch.nonzero(
            keep,
            as_tuple=False,
        ).squeeze(1)

        candidate_boxes = boxes[b, candidate_indices]
        candidate_conf = conf[candidate_indices]
        candidate_cls = cls[candidate_indices]

        keep_nms = batched_nms(
            candidate_boxes,
            candidate_conf,
            candidate_cls,
            iou_threshold,
        )

        final_indices = candidate_indices[keep_nms]
        if final_indices.numel() > max_det:
            top_conf_idx = conf[final_indices].topk(max_det).indices
            final_indices = final_indices[top_conf_idx]

        batch_pseudo.append({
            "boxes": boxes[b, final_indices],
            "classes": cls[final_indices],
            "confidence": conf[final_indices],
            "uncertainty": uncertainty[b, final_indices],
            "indices": final_indices,
        })

    return batch_pseudo


# ============================================================================
# Trainer
# ============================================================================

def save_batch_visualization(
    images: torch.Tensor,
    batch: dict,
    out_path: str | Path,
    max_images: int = 8,
    class_names: Optional[list[str]] = None,
):
    """
    Save a grid visualization of a training batch with drawn bounding boxes.

    Args:
        images: [B, C, H, W] float tensor, values in [0, 1]
        batch: dict with 'batch_idx', 'cls', 'bboxes' (normalized cxcywh),
               same format used by the YOLO loss / your collate_labeled output
        out_path: where to save the resulting image (e.g. "debug/batch_0.jpg")
        max_images: cap how many images from the batch to draw (keeps grid readable)
        class_names: optional list to label boxes with class names instead of indices
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    imgs = images.detach().cpu()
    batch_idx = batch["batch_idx"].detach().cpu()
    cls = batch["cls"].detach().cpu()
    bboxes = batch["bboxes"].detach().cpu()  # normalized cx, cy, w, h

    n = min(imgs.shape[0], max_images)
    tiles = []

    for i in range(n):
        img = imgs[i].permute(1, 2, 0).numpy()  # C,H,W -> H,W,C
        img = (img * 255).clip(0, 255).astype(np.uint8).copy()
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]

        mask = batch_idx == i
        img_boxes = bboxes[mask]
        img_cls = cls[mask]

        for box, c in zip(img_boxes, img_cls):
            cx, cy, bw, bh = box.tolist()
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = str(int(c.item()))
            if class_names is not None and int(c.item()) < len(class_names):
                label = class_names[int(c.item())]

            cv2.putText(
                img, label, (x1, max(y1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
            )

        tiles.append(img)

    # simple horizontal-strip grid (wrap into rows of up to 4)
    cols = min(4, len(tiles))
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    row_imgs = [np.hstack(r) if len(r) == cols else np.hstack(
        r + [np.zeros_like(r[0])] * (cols - len(r))
    ) for r in rows]
    grid = np.vstack(row_imgs)

    cv2.imwrite(str(out_path), grid)

class UnbiasedTeacherV2Trainer:
    """
    YOLO11n Unbiased Teacher v2 trainer.

    Required loader interfaces:

        labeled_loader:
            yields (images, batch_dict)

        unlabeled_loader:
            yields (weak_images, strong_images)

    `batch_dict` must contain the same tensors expected by Ultralytics'
    DetectionLoss, notably:
        batch_idx
        cls
        bboxes
    """

    def __init__(
        self,
        model_weights: str,
        output_dir: Path,
        config: Optional[UTv2Config] = None,
        device: str = "cuda",
        seed: int = 0,
    ):
        """
        Initialize the UTv2 trainer.

        Args:
            model_weights: Path or name of pretrained YOLO weights to load.
            output_dir: Directory where training artifacts (history, metrics)
                will be saved.
            config: Optional UTv2Config overriding defaults.
            device: Torch device string (e.g. 'cuda' or 'cpu').
            seed: Random seed for reproducibility.
        """
        self.cfg = config or UTv2Config()
        self.device = torch.device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)  
        self.init_model_weights = model_weights      
        self.final_weights_path = self.output_dir / "weights" / "best.pt"

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Student starts from pretrained YOLO11n.
        self.student, self.yolo_cfg = build_model(
            model_weights,
            self.device,
        )

        # Teacher is EMA copy of student.
        self.teacher = copy.deepcopy(self.student).to(self.device)
        self.teacher.eval()
        freeze(self.teacher)

        self.criterion = UTv2DetectionLoss(
            self.student,
            uncertainty_weight=self.cfg.uncertainty_weight,
        )

        self.criterion_val = UTv2DetectionLoss(
            self.student,
            uncertainty_weight=0,
        )

        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.epochs
        )

        self.global_step = 0

    # ------------------------------------------------------------------
    # Supervised
    # ------------------------------------------------------------------

    def supervised_loss(
        self,
        images,
        batch,
    ):
        preds = self.student(images)
        return self.criterion(preds, batch)

    # ------------------------------------------------------------------
    # Unsupervised
    # ------------------------------------------------------------------

    @staticmethod
    def xyxy_to_xywh(boxes):
        x1, y1, x2, y2 = boxes.unbind(dim=-1)

        return torch.stack(
            (
                (x1 + x2) / 2,
                (y1 + y2) / 2,
                x2 - x1,
                y2 - y1,
            ),
            dim=-1,
        )

    def build_pseudo_batch(
        self,
        pseudo,
        image_size,
    ):
        """
        Convert sparse pseudo detections into the batch format expected
        by the YOLO target assignment/loss path.
        """
        batch_idx = []
        classes = []
        boxes = []
        uncertainty = []

        for i, detections in enumerate(pseudo):
            h, w = image_size
            pseudo_boxes = detections["boxes"].clone()

            if pseudo_boxes.numel() > 0:
                pseudo_boxes[:, [0, 2]] = pseudo_boxes[:, [0, 2]].clamp(0, w)
                pseudo_boxes[:, [1, 3]] = pseudo_boxes[:, [1, 3]].clamp(0, h)
                valid = (
                    torch.isfinite(pseudo_boxes).all(dim=1)
                    & (pseudo_boxes[:, 2] > pseudo_boxes[:, 0])
                    & (pseudo_boxes[:, 3] > pseudo_boxes[:, 1])
                )
                pseudo_boxes = pseudo_boxes[valid]
                pseudo_classes = detections["classes"][valid]
                pseudo_uncertainty = detections["uncertainty"][valid]
            else:
                pseudo_classes = detections["classes"]
                pseudo_uncertainty = detections["uncertainty"]

            n = pseudo_boxes.shape[0]

            if n == 0:
                continue

            batch_idx.append(
                torch.full(
                    (n,),
                    i,
                    device=self.device,
                    dtype=torch.long,
                )
            )

            classes.append(pseudo_classes)

            pseudo_xywh = self.xyxy_to_xywh(pseudo_boxes)
            pseudo_xywh[:, [0, 2]] /= w
            pseudo_xywh[:, [1, 3]] /= h
            boxes.append(pseudo_xywh)

            uncertainty.append(pseudo_uncertainty)

        if not batch_idx:
            return {
                "batch_idx": torch.zeros(
                    (0,),
                    device=self.device,
                    dtype=torch.long,
                ),
                "cls": torch.zeros(
                    (0,),
                    device=self.device,
                    dtype=torch.long,
                ),
                "bboxes": torch.zeros(
                    (0, 4),
                    device=self.device,
                ),
                "uncertainty": torch.zeros(
                    (0, 4),
                    device=self.device,
                ),
            }

        return {
            "batch_idx": torch.cat(batch_idx),
            "cls": torch.cat(classes),
            "bboxes": torch.cat(boxes),
            "uncertainty": torch.cat(uncertainty),
        }

    def unsupervised_loss(
        self,
        weak_images,
        strong_images,
        pseudo_conf,
    ):
        # Teacher produces pseudo labels from weak images.
        with torch.no_grad():
            self.teacher.eval()

            teacher_preds = self.teacher(weak_images)

            teacher_decoded = decode_predictions(
                self.teacher,
                teacher_preds,
            )

            pseudo = pseudo_labels(
                teacher_decoded,
                pseudo_conf,
                self.cfg.pseudo_nms_iou,
                self.cfg.max_det
            )

        # Student receives strong view.
        student_preds = self.student(strong_images)

        pseudo_batch = self.build_pseudo_batch(
            pseudo,
            image_size=strong_images.shape[-2:],
        )

        # No confident pseudo detections in the whole batch.
        if pseudo_batch["batch_idx"].numel() == 0:
            zero = student_preds["scores"].sum() * 0.0

            return zero, {
                "loss_cls": zero.detach(),
                "loss_box": zero.detach(),
                "num_pseudo": torch.tensor(
                    0,
                    device=self.device,
                ),
                "num_reg_boundaries": torch.tensor(
                    0,
                    device=self.device,
                ),
                "boundary_acceptance": torch.tensor(
                    0.0,
                    device=self.device,
                ),
            }

        assigned = self.criterion.assign_targets(
            student_preds,
            pseudo_batch,
        )

        pred_scores = assigned["pred_scores"]
        target_scores = assigned["target_scores"]
        fg_mask = assigned["fg_mask"]
        target_gt_idx = assigned["target_gt_idx"]
        padded_teacher_sigma = assigned["padded_teacher_sigma"]

        target_scores_sum = target_scores.sum().clamp_min(1.0)
 

        # --------------------------------------------------------------
        # Classification: normal YOLO-style classification against
        # assigned pseudo targets.
        # --------------------------------------------------------------

        loss_cls = (
            focal_loss(pred_scores, target_scores.to(pred_scores.dtype))
            .sum() / target_scores_sum
        )

        # --------------------------------------------------------------
        # Localization
        # --------------------------------------------------------------

        loss_box = pred_scores.sum() * 0.0

        accepted_boundaries = torch.tensor(
            0.0,
            device=self.device,
        )

        acceptance_ratio = torch.tensor(
            0.0,
            device=self.device,
        )

        if fg_mask.any():

            pred_bboxes = assigned["pred_bboxes"]
            target_bboxes = assigned["target_bboxes"]
            anchor_points = assigned["anchor_points"]
            stride_tensor = assigned["stride_tensor"]

            # Standard assigned student/pseudo-target locations.
            student_ltrb = xyxy_to_ltrb(
                pred_bboxes,
                anchor_points,
            )

            target_ltrb = xyxy_to_ltrb(
                target_bboxes / stride_tensor,
                anchor_points,
            )

            # ----------------------------------------------------------
            # Listen2Student uncertainty gate
            # ----------------------------------------------------------

            safe_gt_idx = target_gt_idx.clamp_min(0)

            teacher_sigma = torch.gather(
                padded_teacher_sigma,
                dim=1,
                index=safe_gt_idx.unsqueeze(-1).expand(-1, -1, 4),
            )

            student_sigma = (
                student_preds["uncertainty"]
                .permute(0, 2, 1)
                .contiguous()
            )

            boundary_mask = listen2student_mask(
                teacher_sigma,
                student_sigma,
                fg_mask,
                margin=self.cfg.uncertainty_margin,
                student_min=self.cfg.student_uncertainty_min,
            ).float()

            boundary_loss = F.smooth_l1_loss(
                student_ltrb, target_ltrb.detach(), beta=1.0, reduction="none"
            )

            loss_box = (
                boundary_loss * boundary_mask
            ).sum() / boundary_mask.sum().clamp_min(1.0)

            accepted_boundaries = boundary_mask.sum()

            total_possible_boundaries = fg_mask.sum() * 4

            acceptance_ratio = (
                accepted_boundaries
                / total_possible_boundaries.clamp_min(1.0)
            )

        batch_size = strong_images.shape[0]

        total = (
            self.cfg.unsupervised_cls_weight * loss_cls
            + self.cfg.unsupervised_box_weight * loss_box
        ) * batch_size

        return total, {
            "loss_cls": loss_cls.detach(),
            "loss_box": loss_box.detach(),
            "num_pseudo": torch.tensor(
                sum(x["boxes"].shape[0] for x in pseudo),
                device=self.device,
            ),
            "num_reg_boundaries": accepted_boundaries.detach(),
            "boundary_acceptance": acceptance_ratio.detach(),
        }

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def train_step(
        self,
        labeled_images,
        labeled_batch,
        weak_images,
        strong_images,
        unsupervised_weight=1.0,
        pseudo_conf: Optional[float] = None,
    ):
        self.student.train()
        self.optimizer.zero_grad(set_to_none=True)

        sup_loss, sup_parts = self.supervised_loss(
            labeled_images,
            labeled_batch,
        )

        unsup_loss, unsup_stats = self.unsupervised_loss(
            weak_images,
            strong_images,
            pseudo_conf=pseudo_conf,
        )

        total = (
            self.cfg.supervised_weight * sup_loss
            + unsupervised_weight * unsup_loss
        )

        total.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=10.0)
        self.optimizer.step()

        ema_momentum = self.get_ema_momentum()

        update_ema(
            self.teacher,
            self.student,
            ema_momentum,
        )

        self.global_step += 1

        return {
            "loss": total.detach(),
            "sup_loss": sup_loss.detach(),
            "unsup_loss": unsup_loss.detach(),
            "sup_parts": sup_parts,
            **unsup_stats,
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def get_ema_momentum(self):
        """
        Start with a faster teacher update and gradually approach
        the configured EMA momentum.
        """
        return min(
            1.0 - 1.0 / (self.global_step + 1),
            self.cfg.ema_momentum,
        )
    
    def train(
        self,
        labeled_loader: Iterable,
        unlabeled_loader: Iterable,
        val_loader: Iterable,
        unsup_val_loader: Iterable,
        epochs: Optional[int] = None,
        patience: Optional[int] = 10,
    ):
        epochs = epochs or self.cfg.epochs

        history = []
        unlabeled_iter = iter(unlabeled_loader)
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        # Persist final student and teacher weights in a YOLO-like layout
        weights_dir = self.final_weights_path.parent
        weights_dir.mkdir(parents=True, exist_ok=True)

        # Save training history and best metrics similar to YOLOTrainer.
        history_path = self.output_dir / "train_history.json"
        best_val_path = self.output_dir / "best_val_metrics.json"

        for epoch in range(epochs):
            self.student.train()
            self.teacher.eval()

            if epoch < self.cfg.warmup_epochs:
                unsup_weight = 0.0
            else:
                ramp_progress = (
                    epoch - self.cfg.warmup_epochs + 1
                ) / max(
                    self.cfg.unsup_ramp_epochs,
                    1,
                )
                unsup_weight = max(
                    min(
                        ramp_progress,
                        1.0,
                    ),
                    0.0,
                ) * self.cfg.unsupervised_weight

            epoch_stats = []

            # Train
            for j,labeled in enumerate(labeled_loader):
                try:
                    unlabeled = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    unlabeled = next(unlabeled_iter)

                labeled_images, labeled_batch = labeled
                weak_images, strong_images = unlabeled

                labeled_images = labeled_images.to(
                    self.device,
                    non_blocking=True,
                )

                weak_images = weak_images.to(
                    self.device,
                    non_blocking=True,
                )

                strong_images = strong_images.to(
                    self.device,
                    non_blocking=True,
                )

                labeled_batch = {
                    k: v.to(self.device)
                    if torch.is_tensor(v)
                    else v
                    for k, v in labeled_batch.items()
                }

                if (epoch == 0 or epoch == epochs - 1) and j == 0:
                    save_batch_visualization(
                        labeled_images, labeled_batch,
                        out_path=(self.output_dir/f"labeled_train_batch{epoch}.jpg").as_posix(),
                    )

                stats = self.train_step(
                    labeled_images,
                    labeled_batch,
                    weak_images,
                    strong_images,
                    unsupervised_weight=unsup_weight,
                    pseudo_conf=self.cfg.pseudo_conf,
                )

                epoch_stats.append(stats)

            self.scheduler.step() 

            # Validate student on labeled validation set using standard YOLO loss.
            val_results = self.validate_student_loss(
                val_loader,
            )
            val_unsup_results = self.validate_student_unsup_loss(
                unsup_val_loader
            )
            val_loss = self.cfg.supervised_weight * val_results["loss"] + self.cfg.unsupervised_weight * val_unsup_results["loss"]

            mean_loss = torch.stack(
                [
                    x["loss"].float().cpu()
                    for x in epoch_stats
                ]
            ).mean().item()

            mean_sup = torch.stack(
                [
                    x["sup_loss"].float().cpu()
                    for x in epoch_stats
                ]
            ).mean().item()

            mean_unsup = torch.stack(
                [
                    x["unsup_loss"].float().cpu()
                    for x in epoch_stats
                ]
            ).mean().item()

            mean_pseudo = sum(
                x["num_pseudo"].float().cpu().item()
                for x in epoch_stats
            ) / max(len(epoch_stats), 1)

            mean_boundary_acceptance = sum(
                x["boundary_acceptance"].float().cpu().item()
                for x in epoch_stats
            ) / max(len(epoch_stats), 1)    

            row = {
                "epoch": epoch + 1,
                "loss": mean_loss,
                "sup_loss": mean_sup,
                "unsup_loss": mean_unsup,
                "num_pseudo": mean_pseudo,
                "boundary_acceptance": mean_boundary_acceptance,
                "val_loss": val_loss,
                "val_sup_loss": val_results["loss"],
                "val_unsup_loss": val_unsup_results["loss"],
                "val_unsup_num_pseudo": val_unsup_results["num_pseudo"],
                "val_unsup_acceptance": val_unsup_results["val_acceptance"],
            }

            history.append(row)

            print(
                f"[UTv2] epoch {epoch + 1}/{epochs} | "
                f"loss={mean_loss:.5f} | "
                f"sup={mean_sup:.5f} | "
                f"unsup={mean_unsup:.5f} | "
                f"pseudo={mean_pseudo:.1f} | "
                f"accept={mean_boundary_acceptance:.3f} | "
                f"val={val_loss:.5f}"
            )

            if val_loss < best_val_loss:
                # Save best weights
                best_val_loss = val_loss
                epochs_without_improvement = 0
                #self.save_checkpoint(self.final_weights_path)
                self.export_student(self.final_weights_path)
            else:
                epochs_without_improvement += 1

            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

            #self.save_checkpoint(self.final_weights_path.parent / "last.pt")
            self.export_student(self.final_weights_path.parent / "last.pt")

            # Patience-based early stopping
            if patience is not None:
                if epochs_without_improvement >= patience:
                    print(
                        f"[UTv2] Early stopping after {epoch + 1} epochs; "
                        f"no Val Loss improvement for {patience} epochs."
                    )
                    break

        best_metrics = {}
        if len(history) > 0:
            best_metrics = min(history, key=lambda r: r["val_loss"])

        with open(best_val_path, "w", encoding="utf-8") as f:
            json.dump(best_metrics, f, indent=2)

        return history

    @torch.no_grad()
    def validate_student_loss(self, val_loader):
        """
        Compute mean supervised loss on a held-out labeled validation set.
        Uses the same UTv2DetectionLoss as training (no augmentation, no grad).
        Returns a dict of per-component averages plus the total.
        """
        self.student.train()  # keep dict-format forward output; no grad due to decorator

        totals = {"loss": 0.0, "box": 0.0, "cls": 0.0, "dfl": 0.0, "uncertainty": 0.0}
        n_batches = 0

        for images, batch in val_loader:
            images = images.to(self.device, non_blocking=True)
            batch = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            preds = self.student(images)
            total_loss, parts = self.criterion_val(preds, batch)

            # parts is loss.detach() -> [box, cls, dfl, uncertainty] (pre-sum, per-component)
            totals["loss"] += total_loss.item()
            totals["box"] += parts["box_loss"].item()
            totals["cls"] += parts["cls_loss"].item()
            totals["dfl"] += parts["dfl_loss"].item()
            totals["uncertainty"] += parts["uncertainty_loss"].item() if "uncertainty_loss" in parts else 0.0
            n_batches += 1

        self.student.train()

        n_batches = max(n_batches, 1)
        return {k: v / n_batches for k, v in totals.items()}

    @torch.no_grad()
    def validate_student_unsup_loss(self, unsup_val_loader):
        """
        Measure unsupervised (pseudo-label) loss on a held-out slice of the
        unlabeled pool — images never used for training. This is independent
        of the labeled val_loss and directly measures whether the SSL branch
        is extracting useful signal, not just fitting the small labeled set.
        """
        was_training = self.student.training
        self.student.train()  # keep dict-format forward output; no grad due to decorator
        self.teacher.eval()

        pseudo_conf = self.cfg.pseudo_conf

        totals = {"unsup_loss": 0.0, "loss_cls": 0.0, "loss_box": 0.0}
        total_pseudo = 0.0
        total_acceptance = 0.0
        n_batches = 0

        for weak_images, strong_images in unsup_val_loader:
            weak_images = weak_images.to(self.device, non_blocking=True)
            strong_images = strong_images.to(self.device, non_blocking=True)

            total, stats = self.unsupervised_loss(
                weak_images, strong_images, pseudo_conf=pseudo_conf,
            )

            totals["unsup_loss"] += total.item()
            totals["loss_cls"] += stats["loss_cls"].item()
            totals["loss_box"] += stats["loss_box"].item()
            total_pseudo += stats["num_pseudo"].item()
            total_acceptance += stats["boundary_acceptance"].item()
            n_batches += 1

        self.student.train(was_training)

        n_batches = max(n_batches, 1)
        return {
            "loss": totals["unsup_loss"] / n_batches,
            "cls": totals["loss_cls"] / n_batches,
            "box": totals["loss_box"] / n_batches,
            "num_pseudo": total_pseudo / n_batches,
            "val_acceptance": total_acceptance / n_batches,
        }

    def save_checkpoint(self, path: str | Path):
        """Save the student to `path` using a standard Ultralytics checkpoint format.

        The `model` payload must be a plain state_dict so the resulting file can be
        loaded by `YOLO(path)`/`model.load_state_dict()`-style workflows.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        ckpt = {
            "student": self.student.state_dict(),
            "teacher": self.teacher.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": vars(self.cfg)
        }

        torch.save(ckpt, path)

    def export_student(self, path: str | Path):
        """
        Export the trained student as a standard Ultralytics YOLO checkpoint.

        The training-only UTv2 uncertainty head is removed and replaced
        with the original standard Detect head.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Replace UTv2Detect with normal Detect.
        export_model = convert_to_standard_yolo(self.student)

        # Put model on CPU for checkpoint serialization.
        export_model = export_model.cpu()

        # Build a native Ultralytics YOLO wrapper.
        yolo = YOLO(self.init_model_weights)
        yolo.model = export_model
        yolo.task = "detect"

        # Save native Ultralytics checkpoint.
        yolo.save(path)

    # ------------------------------------------------------------------
    # Convenience adapter: build simple datasets from folders and train
    # ------------------------------------------------------------------

    def train_from_dirs(
        self,
        labeled_yaml: str,
        unlabeled_images_dir: str | Path,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        val_batch_size: Optional[int] = None,
        img_size: Optional[int] = None,
        patience: Optional[int] = 50
    ):
        """
        Build simple dataloaders from image/label folders and run UTv2 training.

        Expects YOLO-style label .txt files (class cx cy w h, normalized).
        """
        with open(labeled_yaml, "r", encoding="utf-8") as f:
            labeled_cfg_data = yaml.safe_load(f)
            main_path = Path(labeled_cfg_data["path"])
            labeled_images_dir = main_path / Path(labeled_cfg_data["train"])
            labeled_labels_dir = main_path / Path(labeled_cfg_data["train"].replace("images", "labels"))
            val_images_dir = main_path / Path(labeled_cfg_data["val"])
            val_labels_dir = main_path / Path(labeled_cfg_data["val"].replace("images", "labels"))


        unlabeled_images_dir = Path(unlabeled_images_dir)

        batch_size = int(batch_size or self.cfg.batch_size)
        val_batch_size = int(val_batch_size or self.cfg.val_batch_size)
        img_size = int(img_size or self.cfg.img_size)

        # Define transforms for labeled images (single view) and unlabeled images (weak + strong views).
        # labeled
        transform = T.Compose([
            T.ToImage(),
            T.Resize((img_size, img_size)),
            T.ColorJitter(0.4, 0., 0.7, 0.015),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(translate=(0.1, 0.1), degrees=5, scale=(0.5,1.5)),
            T.ClampBoundingBoxes(),
            T.ToDtype(torch.float32, scale=True),
        ])
        eval_transform = T.Compose([
            T.ToImage(),
            T.Resize((img_size, img_size)),
            T.ToDtype(torch.float32, scale=True),
        ])
        # weak and strong transforms for unlabeled views
        shared_t = T.Compose([
            T.ToImage(),
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(translate=(0.1, 0.1), degrees=0),
        ])
        weak_t = T.Compose([
            T.ToDtype(torch.float32, scale=True),
        ])
        strong_t = T.Compose([
            T.ColorJitter(0.4, 0., 0.7, 0.015),
            T.ToDtype(torch.float32, scale=True),
        ])
        

        def parse_label_file(lbl_path: Path, img_w: int, img_h: int):
            boxes = []
            classes = []
            if not lbl_path.exists():
                return classes, boxes

            for line in lbl_path.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])
                classes.append(cls)
                boxes.append([cx, cy, w, h])

            if len(boxes) == 0:
                return [], []

            return torch.tensor(classes, dtype=torch.long), torch.tensor(boxes, dtype=torch.float)

        class LabeledDataset(torch.utils.data.Dataset):
            def __init__(self, images_dir: Path, labels_dir: Path, transform, img_size):
                self.images = sorted([p for p in images_dir.glob("*.npy")])
                self.labels_dir = labels_dir
                self.transform = transform
                self.img_size = img_size

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                img_path = self.images[idx]
                img = np.load(img_path)
                h, w = img.shape[:2]

                lbl_path = self.labels_dir / f"{img_path.stem}.txt"
                classes, boxes = parse_label_file(lbl_path, w, h)

                if not torch.is_tensor(boxes) or boxes.numel() == 0:
                    classes = torch.zeros((0,), dtype=torch.long)
                    boxes = torch.zeros((0, 4), dtype=torch.float32)
                else:
                    boxes = boxes * torch.tensor([w, h, w, h], dtype=torch.float32)

                boxes = tv_tensors.BoundingBoxes(
                    boxes,
                    format=tv_tensors.BoundingBoxFormat.CXCYWH,
                    canvas_size=(h, w),
                )

                img_t, boxes_t = self.transform(img, boxes)  # only image + boxes, no labels involved

                new_h, new_w = boxes_t.canvas_size
                boxes_out = boxes_t.data.clone()
                classes_out = classes.clone()

                if boxes_out.numel() > 0:
                    # manual sanitize: drop degenerate boxes (clamp already handled out-of-canvas)
                    cx, cy, bw, bh = boxes_out.unbind(dim=-1)
                    valid = (
                        torch.isfinite(boxes_out).all(dim=-1)
                        & (bw > 1.0)   # at least 1px wide, post-clamp
                        & (bh > 1.0)   # at least 1px tall
                    )
                    boxes_out = boxes_out[valid]
                    classes_out = classes_out[valid]

                    boxes_out = boxes_out / torch.tensor([new_w, new_h, new_w, new_h], dtype=torch.float32)
                    boxes_out = boxes_out.clamp(0.0, 1.0)

                return img_t, classes_out, boxes_out

        class UnlabeledTwoViewDataset(torch.utils.data.Dataset):
            def __init__(self, images: Iterable, shared_transform, weak_transform, strong_transform):
                self.images = images
                self.transform = shared_transform
                self.weak = weak_transform
                self.strong = strong_transform

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                img_path = self.images[idx]
                img = np.load(img_path)
                img = self.transform(img)
                weak = self.weak(img)
                strong = self.strong(img)
                return weak, strong

        def collate_labeled(batch):
            imgs = [b[0] for b in batch]
            classes = [b[1] for b in batch]
            boxes = [b[2] for b in batch]

            imgs = torch.stack(imgs, dim=0)

            # flatten labels across batch
            batch_idx_list = []
            cls_list = []
            box_list = []
            for i, (c, bxs) in enumerate(zip(classes, boxes)):
                if isinstance(c, torch.Tensor) and c.numel() > 0:
                    n = c.shape[0]
                    batch_idx_list.append(torch.full((n,), i, dtype=torch.long))
                    cls_list.append(c)
                    box_list.append(bxs)

            if len(batch_idx_list) > 0:
                batch_idx = torch.cat(batch_idx_list, dim=0)
                cls = torch.cat(cls_list, dim=0)
                bboxes = torch.cat(box_list, dim=0)
            else:
                batch_idx = torch.zeros((0,), dtype=torch.long)
                cls = torch.zeros((0,), dtype=torch.long)
                bboxes = torch.zeros((0, 4), dtype=torch.float)

            return imgs, {"batch_idx": batch_idx, "cls": cls, "bboxes": bboxes}

        all_unlabeled_images = sorted(Path(unlabeled_images_dir).glob("*.npy"))
        random.shuffle(all_unlabeled_images)

        n_unsup_val = max(1, int(0.1 * len(all_unlabeled_images)))  # 10% held out, tune as needed
        unsup_val_images = all_unlabeled_images[:n_unsup_val]
        unsup_train_images = all_unlabeled_images[n_unsup_val:]

        labeled_ds = LabeledDataset(labeled_images_dir, labeled_labels_dir, transform, img_size)
        unlabeled_ds = UnlabeledTwoViewDataset(unsup_train_images, shared_t, weak_t, strong_t)
        unsup_val_ds = UnlabeledTwoViewDataset(unsup_val_images, eval_transform, weak_t, strong_t)
        val_ds = LabeledDataset(val_images_dir, val_labels_dir, eval_transform, img_size)

        num_workers = 0 # No workers with custom dataset

        labeled_loader = torch.utils.data.DataLoader(
            labeled_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_labeled,
        )

        unlabeled_loader = torch.utils.data.DataLoader(
            unlabeled_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_labeled,
        )
        unsup_val_loader = torch.utils.data.DataLoader(
            unsup_val_ds,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=0,
        )

        return self.train(labeled_loader, unlabeled_loader, val_loader, unsup_val_loader, epochs=epochs, patience=patience)
        