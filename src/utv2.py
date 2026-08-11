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
from typing import Any, Dict, Iterable, Optional
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import make_anchors


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
    uncertainty_margin: float = 0.0
    student_uncertainty_threshold: Optional[float] = None

    supervised_weight: float = 1.0
    unsupervised_cls_weight: float = 1.0
    unsupervised_box_weight: float = 1.0
    uncertainty_weight: float = 1.0

    warmup_epochs: int = 1


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


def replace_detect_with_utv2(model: DetectionModel):
    """Replace YOLO11 Detect while retaining all original YOLO weights."""

    old = model.model[-1]

    if not isinstance(old, Detect):
        raise TypeError(f"Expected Detect as final module, got {type(old)}")

    new = UTv2Detect(
        nc=old.nc,
        reg_max=old.reg_max,
        end2end=getattr(old, "end2end", False),
        ch=getattr(old, "ch", ()),
    )

    old_state = old.state_dict()
    new_state = new.state_dict()

    for key, value in old_state.items():
        if key in new_state and new_state[key].shape == value.shape:
            new_state[key].copy_(value)

    new.load_state_dict(new_state, strict=False)
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
    preds: Dict[str, torch.Tensor],
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

    pred_distri = (
        preds["boxes"]
        .permute(0, 2, 1)
        .contiguous()
    )

    scores = (
        preds["scores"]
        .permute(0, 2, 1)
        .contiguous()
        .sigmoid()
    )

    uncertainty = (
        preds["uncertainty"]
        .permute(0, 2, 1)
        .contiguous()
    )

    anchors, stride = make_anchors(
        preds["feats"],
        head.stride,
        0.5,
    )

    boxes = head.bbox_decode(
        anchors,
        pred_distri,
    )

    boxes = boxes * stride

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
        target_scores_sum = max(int(target_scores.sum().item()), 1)

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

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
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


# ============================================================================
# Listen2Student
# ============================================================================

def listen2student_mask(
    teacher_sigma: torch.Tensor,
    student_sigma: torch.Tensor,
    teacher_confident: torch.Tensor,
    margin: float = 0.0,
    student_threshold: Optional[float] = None,
):
    """
    Boundary-wise Listen2Student gate.

    A teacher boundary is trusted only when the teacher is more certain than
    the student for that particular L/T/R/B boundary.
    """

    # Teacher is trusted when its boundary uncertainty is strictly
    # smaller than the student's uncertainty (optionally with a margin).
    mask = (teacher_sigma + margin) < student_sigma

    # Only consider locations where teacher produced a confident pseudo-label.
    mask &= teacher_confident.unsqueeze(-1)

    # If a student uncertainty threshold is provided, require the
    # student's uncertainty to be at most that threshold (i.e. less
    # uncertain than the given limit). This makes the threshold act as
    # a maximum-allowed student uncertainty for accepting teacher guidance.
    if student_threshold is not None:
        mask &= student_sigma <= student_threshold

    return mask


# ============================================================================
# Pseudo labels
# ============================================================================

@torch.no_grad()
def pseudo_labels(
    decoded: Dict[str, torch.Tensor],
    confidence: float,
):
    scores = decoded["scores"]

    conf, cls = scores.max(dim=-1)

    mask = conf >= confidence

    return {
        "boxes": decoded["boxes"],
        "classes": cls,
        "confidence": conf,
        "uncertainty": decoded["uncertainty"],
        "mask": mask,
    }


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
            )

        # Student receives strong view.
        student_preds = self.student(strong_images)

        student_decoded = decode_predictions(
            self.student,
            student_preds,
        )

        teacher_sigma = pseudo["uncertainty"]
        student_sigma = student_decoded["uncertainty"]

        confident = pseudo["mask"]

        # --------------------------------------------------------------
        # Listen2Student regression selection
        # --------------------------------------------------------------

        boundary_mask = listen2student_mask(
            teacher_sigma,
            student_sigma,
            confident,
            margin=self.cfg.uncertainty_margin,
            student_threshold=self.cfg.student_uncertainty_threshold,
        )

        # Use float mask for loss computations (weights / sums).
        boundary_mask = boundary_mask.float()

        # --------------------------------------------------------------
        # Classification pseudo-label loss
        # --------------------------------------------------------------

        student_logits = (
            student_preds["scores"]
            .permute(0, 2, 1)
            .contiguous()
        )

        cls = pseudo["classes"]

        target_cls = torch.zeros_like(student_logits)

        target_cls.scatter_(
            -1,
            cls.unsqueeze(-1),
            1.0,
        )

        cls_loss = F.binary_cross_entropy_with_logits(
            student_logits,
            target_cls,
            reduction="none",
        )

        # Weight classification targets by pseudo-label confidence and
        # only consider entries where the teacher was confident.
        pseudo_conf = pseudo["confidence"]
        cls_mask = confident.unsqueeze(-1).float()

        weight = cls_mask * pseudo_conf.unsqueeze(-1)

        loss_cls = (
            cls_loss * weight
        ).sum() / weight.sum().clamp_min(1.0)

        # --------------------------------------------------------------
        # Boundary-wise regression
        # --------------------------------------------------------------

        anchors = student_decoded["anchors"]
        stride = student_decoded["stride"]

        teacher_boxes = pseudo["boxes"] / stride
        student_boxes = student_decoded["boxes"] / stride

        teacher_ltrb = xyxy_to_ltrb(
            teacher_boxes,
            anchors,
        )

        student_ltrb = xyxy_to_ltrb(
            student_boxes,
            anchors,
        )

        loss_box = (
            (student_ltrb - teacher_ltrb.detach()).square()
            * boundary_mask
        ).sum() / boundary_mask.sum().clamp_min(1.0)

        total = (
            self.cfg.unsupervised_cls_weight * loss_cls
            + self.cfg.unsupervised_box_weight * loss_box
        )

        return total, {
            "loss_cls": loss_cls.detach(),
            "loss_box": loss_box.detach(),
            "num_pseudo": confident.sum().detach(),
            "num_reg_boundaries": boundary_mask.sum().detach(),
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

        update_ema(
            self.teacher,
            self.student,
            self.cfg.ema_momentum,
        )

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

    def train(
        self,
        labeled_loader: Iterable,
        unlabeled_loader: Iterable,
        epochs: Optional[int] = None,
    ):
        epochs = epochs or self.cfg.epochs

        history = []
        unlabeled_iter = iter(unlabeled_loader)

        for epoch in range(epochs):
            self.student.train()
            self.teacher.eval()

            if epoch < self.cfg.warmup_epochs:
                unsup_weight = 0.0
            else:
                unsup_weight = 1.0

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

        # Save training history and best metrics similar to YOLOTrainer.
        history_path = self.output_dir / "train_history.json"
        best_val_path = self.output_dir / "best_val_metrics.json"

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        best_metrics = {}
        if len(history) > 0:
            # Best epoch by minimal total loss.
            best_metrics = min(history, key=lambda r: r.get("loss", float("inf")))

        with open(best_val_path, "w", encoding="utf-8") as f:
            json.dump(best_metrics, f, indent=2)

        # Persist final student and teacher weights in a YOLO-like layout
        weights_dir = self.output_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        weights_path = weights_dir / "best.pt"

        try:
            self.save_weights(weights_path)
        except Exception:
            # Do not fail training if saving fails; log via print for now.
            print(f"Warning: failed to save weights to {weights_path}")

        return history

    def save_weights(self, path: str | Path):
        """Save the student and teacher state_dicts to `path`.

        Args:
            path: file path to write (e.g. '.../weights/best.pt').
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model": self.student.state_dict(),
                "teacher": self.teacher.state_dict(),
            },
            str(path),
        )

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

        transform = T.Compose([
            T.Resize((img_size, img_size)),
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
                # convert normalized cx,cy,w,h to absolute xyxy
                x1 = (cx - w / 2.0) * img_w
                y1 = (cy - h / 2.0) * img_h
                x2 = (cx + w / 2.0) * img_w
                y2 = (cy + h / 2.0) * img_h

                # Scale boxes to the resized image size used by transforms
                # (train_from_dirs resizes images to (img_size, img_size)).
                if img_w > 0 and img_h > 0:
                    scale_x = img_size / float(img_w)
                    scale_y = img_size / float(img_h)
                    x1 *= scale_x
                    x2 *= scale_x
                    y1 *= scale_y
                    y2 *= scale_y
                classes.append(cls)
                boxes.append([x1, y1, x2, y2])

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
            def __init__(self, images_dir: Path, weak_transform, strong_transform):
                self.images = sorted([p for p in images_dir.glob("*.*")])
                self.weak = weak_transform
                self.strong = strong_transform

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                img_path = self.images[idx]
                img = Image.open(img_path).convert("RGB")
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

        # weak and strong transforms for unlabeled views
        weak_t = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
        strong_t = T.Compose([
            T.Resize((img_size, img_size)),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.ToTensor(),
        ])

        labeled_ds = LabeledDataset(labeled_images_dir, labeled_labels_dir, transform, img_size)
        unlabeled_ds = UnlabeledTwoViewDataset(unlabeled_images_dir, weak_t, strong_t)

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

        return self.train(labeled_loader, unlabeled_loader, epochs=epochs)
        


# ============================================================================
# Minimal utility for saving
# ============================================================================

def save_student(
    trainer: UnbiasedTeacherV2Trainer,
    path: str,
):
    torch.save(
        {
            "model": trainer.student.state_dict(),
            "teacher": trainer.teacher.state_dict(),
        },
        path,
    )


if __name__ == "__main__":
    print("UnbiasedTeacherV2 YOLO11 module.")
    print("Use UnbiasedTeacherV2Trainer(...) from your experiment.")