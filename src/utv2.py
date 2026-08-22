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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors

import numpy as np

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class UTv2Config:
    epochs: int = 50
    batch_size: int = 8
    img_size: int = 640
    lr: float = 1e-3
    weight_decay: float = 5e-4

    ema_momentum: float = 0.9996

    pseudo_conf: float = 0.7
    pseudo_nms_iou: float = 0.5
    uncertainty_margin: float = 0.0

    supervised_weight: float = 1.0
    unsupervised_cls_weight: float = 1.0
    unsupervised_box_weight: float = 1.0
    uncertainty_weight: float = 1.0

    warmup_epochs: int = 1
    unsup_ramp_epochs: int = 5


# ============================================================================
# Model
# ============================================================================

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


def replace_detect_with_utv2(model: DetectionModel):
    """Replace YOLO11 Detect while retaining all original YOLO weights."""

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
    model = replace_detect_with_utv2(yolo.model)
    return model.to(device)


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

class UTv2DetectionLoss(v8DetectionLoss):
    """
    Existing Ultralytics YOLO detection loss + supervised uncertainty NLL.

    The normal:
        assignment -> BCE -> IoU/Bbox -> DFL

    path is retained.
    """

    def __init__(
        self,
        model,
        uncertainty_weight=1.0,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.uncertainty_weight = uncertainty_weight

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
        loss = torch.zeros(4, device=self.device)

        pred_distri = (
            preds["boxes"]
            .permute(0, 2, 1)
            .contiguous()
        )

        pred_scores = (
            preds["scores"]
            .permute(0, 2, 1)
            .contiguous()
        )

        pred_sigma = (
            preds["uncertainty"]
            .permute(0, 2, 1)
            .contiguous()
        )

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

        # Ensure scalar denominator (Python int) to avoid Tensor-vs-int
        # comparison errors and downstream type issues.
        target_scores_sum = int(target_scores.sum().clamp_min(1.0).item())

        loss[1] = (
            self.bce(
                pred_scores,
                target_scores.to(dtype),
            ).sum()
            / target_scores_sum
        )

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

            loss[3] = self.uncertainty_nll(
                pred_bboxes,
                target_bboxes / stride_tensor,
                pred_sigma,
                anchor_points,
                target_scores,
                fg_mask,
            )

        hyp = self.hyp
        loss[0] *= hyp.get("box", 1.0) if isinstance(hyp, dict) else hyp.box
        loss[1] *= hyp.get("cls", 1.0) if isinstance(hyp, dict) else hyp.cls
        loss[2] *= hyp.get("dfl", 1.0) if isinstance(hyp, dict) else hyp.dfl
        loss[3] *= self.uncertainty_weight

        return loss.sum() * batch_size, loss.detach()

    @staticmethod
    def uncertainty_nll(
        pred_boxes,
        target_boxes,
        sigma,
        anchors,
        target_scores,
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

        sigma = sigma.clamp_min(1e-4)

        nll = (
            (pred_ltrb - target_ltrb.detach()).square()
            / (2.0 * sigma.square())
            + torch.log(sigma)
        ).sum(dim=-1)

        weight = (
            target_scores.max(dim=-1).values
            * fg_mask.float()
        )

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
    teacher_confident: torch.Tensor,
    margin: float = 0.0,
):
    """
    Trust a teacher boundary when the teacher is more certain than
    the student for that boundary.
    """
    mask = (
        teacher_sigma + margin
        < student_sigma
    )

    mask &= teacher_confident.unsqueeze(-1)

    return mask


# ============================================================================
# Pseudo labels
# ============================================================================

@torch.no_grad()
def pseudo_labels(
    decoded: Dict[str, torch.Tensor],
    confidence: float,
    iou_threshold: float = 0.5,
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

        if not keep.any():
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
        self.final_weights_path = self.output_dir / "weights" / "best.pt"

        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Student starts from pretrained YOLO11n.
        self.student = build_model(
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

        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
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
            n = detections["boxes"].shape[0]

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

            classes.append(detections["classes"])

            h, w = image_size
            pseudo_xywh = self.xyxy_to_xywh(detections["boxes"])
            pseudo_xywh[:, [0, 2]] /= w
            pseudo_xywh[:, [1, 3]] /= h
            boxes.append(pseudo_xywh)

            uncertainty.append(detections["uncertainty"])

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
                self.cfg.pseudo_conf,
                self.cfg.pseudo_nms_iou,
            )

        # Student receives strong view.
        student_preds = self.student(strong_images)

        student_decoded = decode_predictions(
            self.student,
            student_preds,
        )

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
            self.criterion.bce(
                pred_scores,
                target_scores.to(pred_scores.dtype),
            ).sum()
            / target_scores_sum
        )

        # --------------------------------------------------------------
        # Localization
        # --------------------------------------------------------------

        loss_box = pred_scores.sum() * 0.0
        num_reg_boundaries = 0

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

            # Student uncertainty at all assigned locations.
            student_sigma = student_decoded["uncertainty"]

            teacher_sigma = torch.gather(
                padded_teacher_sigma,
                dim=1,
                index=target_gt_idx.unsqueeze(-1).expand(-1, -1, 4),
            )

            boundary_mask = listen2student_mask(
                teacher_sigma,
                student_sigma,
                fg_mask,
                margin=self.cfg.uncertainty_margin,
            ).float()

            loss_box = (
                (student_ltrb - target_ltrb.detach()).square()
                * boundary_mask
            ).sum() / boundary_mask.sum().clamp_min(1.0)

            num_reg_boundaries = boundary_mask.sum()

        total = (
            self.cfg.unsupervised_cls_weight * loss_cls
            + self.cfg.unsupervised_box_weight * loss_box
        )

        return total, {
            "loss_cls": loss_cls.detach(),
            "loss_box": loss_box.detach(),
            "num_pseudo": torch.tensor(
                sum(x["boxes"].shape[0] for x in pseudo),
                device=self.device,
            ),
            "num_reg_boundaries": num_reg_boundaries.detach()
            if torch.is_tensor(num_reg_boundaries)
            else torch.tensor(
                num_reg_boundaries,
                device=self.device,
            ),
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
        )

        total = (
            self.cfg.supervised_weight * sup_loss
            + unsupervised_weight * unsup_loss
        )

        total.backward()
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
        epochs: Optional[int] = None,
        patience: Optional[int] = 10,
    ):
        epochs = epochs or self.cfg.epochs

        history = []
        unlabeled_iter = iter(unlabeled_loader)
        best_loss = float("inf")

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
                unsup_weight = min(
                    ramp_progress,
                    1.0,
                )

            epoch_stats = []

            for labeled in labeled_loader:
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

                stats = self.train_step(
                    labeled_images,
                    labeled_batch,
                    weak_images,
                    strong_images,
                    unsupervised_weight=unsup_weight,
                )

                epoch_stats.append(stats)

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

            row = {
                "epoch": epoch + 1,
                "loss": mean_loss,
                "sup_loss": mean_sup,
                "unsup_loss": mean_unsup,
                "num_pseudo": mean_pseudo,
            }

            history.append(row)

            print(
                f"[UTv2] epoch {epoch + 1}/{epochs} | "
                f"loss={mean_loss:.5f} | "
                f"sup={mean_sup:.5f} | "
                f"unsup={mean_unsup:.5f} | "
                f"pseudo={mean_pseudo:.1f}"
            )

            if mean_loss < best_loss:
                best_loss = mean_loss
                self.save_weights(self.final_weights_path)

            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

            self.save_weights(self.final_weights_path.parent / "last.pt")

            # Patience-based early stopping
            if patience is not None:
                if len(history) > patience:
                    recent_losses = [h["loss"] for h in history[-patience:]]
                    # No change in metric
                    EPS = 1e-4
                    mean = np.mean(recent_losses)
                    std = np.std(recent_losses)
                    is_stagnant = std < EPS and abs(mean - best_loss) < EPS
                    # Loss near zero
                    is_near_zero = mean < EPS
                    if is_stagnant or is_near_zero:
                        print(f"[UTv2] Early stopping after {epoch + 1} epochs.")
                        break

        best_metrics = {}
        if len(history) > 0:
            # Best epoch by minimal total loss.
            best_metrics = min(history, key=lambda r: r.get("loss", float("inf")))

        with open(best_val_path, "w", encoding="utf-8") as f:
            json.dump(best_metrics, f, indent=2)

        return history

    def save_weights(self, path: str | Path):
        """Save the student to `path` using a standard Ultralytics checkpoint format.

        The `model` payload must be a plain state_dict so the resulting file can be
        loaded by `YOLO(path)`/`model.load_state_dict()`-style workflows.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        ckpt = {
            "model": self.student.state_dict(),
            "teacher": self.teacher.state_dict(),
            "train_args": vars(self.args) if hasattr(self, "args") else {},
            "epoch": getattr(self, "global_step", 0),
        }

        torch.save(ckpt, path)

    # ------------------------------------------------------------------
    # Convenience adapter: build simple datasets from folders and train
    # ------------------------------------------------------------------

    def train_from_dirs(
        self,
        labeled_images_dir: str | Path,
        labeled_labels_dir: str | Path,
        unlabeled_images_dir: str | Path,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        img_size: Optional[int] = None,
        num_workers: int = 0,
        patience: Optional[int] = 50
    ):
        """
        Build simple dataloaders from image/label folders and run UTv2 training.

        Expects YOLO-style label .txt files (class cx cy w h, normalized).
        """
        labeled_images_dir = Path(labeled_images_dir)
        labeled_labels_dir = Path(labeled_labels_dir)
        unlabeled_images_dir = Path(unlabeled_images_dir)

        batch_size = int(batch_size or self.cfg.batch_size)
        img_size = int(img_size or self.cfg.img_size)

        # Define transforms for labeled images (single view) and unlabeled images (weak + strong views).
        # labeled
        transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])
        # weak and strong transforms for unlabeled views
        shared_t = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(translate=(0.1, 0.1)),
        ])
        weak_t = T.Compose([
            T.ToTensor(),
        ])
        strong_t = T.Compose([
            T.ColorJitter(0.4, 0., 0.7, 0.015),
            T.ToTensor(),
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
                self.images = sorted([p for p in images_dir.glob("*.*")])
                self.labels_dir = labels_dir
                self.transform = transform
                self.img_size = img_size

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                img_path = self.images[idx]
                img = Image.open(img_path).convert("RGB")
                w, h = img.size
                img_t = self.transform(img)

                lbl_path = self.labels_dir / f"{img_path.stem}.txt"
                classes, boxes = parse_label_file(lbl_path, w, h)

                return img_t, classes, boxes

        class UnlabeledTwoViewDataset(torch.utils.data.Dataset):
            def __init__(self, images_dir: Path, shared_transform, weak_transform, strong_transform):
                self.images = sorted([p for p in images_dir.glob("*.*")])
                self.transform = shared_transform
                self.weak = weak_transform
                self.strong = strong_transform

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                img_path = self.images[idx]
                img = Image.open(img_path).convert("RGB")
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

        labeled_ds = LabeledDataset(labeled_images_dir, labeled_labels_dir, transform, img_size)
        unlabeled_ds = UnlabeledTwoViewDataset(unlabeled_images_dir, shared_t, weak_t, strong_t)

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

        return self.train(labeled_loader, unlabeled_loader, epochs=epochs, patience=patience)
        


# ============================================================================
# Minimal utility for saving
# ============================================================================

def save_student(
    trainer: UnbiasedTeacherV2Trainer,
    path: str,
):
    trainer.save_weights(path)