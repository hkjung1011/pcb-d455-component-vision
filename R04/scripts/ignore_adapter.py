"""R04 training-only, spatial negative-classification ignore for YOLO11 detection.

This adapter changes no installed package and no on-disk YOLO label.  A -1 class
exists only in memory to carry ignore rectangles through the same Instances,
letterbox, flip, Format and collate operations as positive boxes.  The criterion
removes those pseudo-labels before assignment and suppresses only unassigned
negative classification loss in the ignore rectangles.  Positive GT regions and
all assigned positive anchors take precedence; box/DFL loss remains upstream.

Supported geometry is letterbox plus horizontal/vertical flips, deliberately
excluding RandomPerspective's tiny-box filtering even at identity settings.
No validation or test annotation receives ignore pseudo-labels.  Installed
Ultralytics 8.4.120 is the validated API; changed versions fail closed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import ultralytics
from ultralytics.data.augment import Compose, Format, LetterBox, RandomFlip, RandomHSV
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import colorstr
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.torch_utils import unwrap_model

VALIDATED_ULTRALYTICS_VERSION = "8.4.120"
SIDECAR_SCHEMA = "r04-training-ignore-v1"
IGNORE_REASONS = frozenset({"source_unknown", "target_fragment_lt50"})


def _check_version():
    if ultralytics.__version__ != VALIDATED_ULTRALYTICS_VERSION:
        raise RuntimeError("R04 ignore adapter requires Ultralytics " + VALIDATED_ULTRALYTICS_VERSION)


def _path_key(value):
    return str(Path(value).resolve())


class IgnoreYOLODataset(YOLODataset):
    """Train-only sidecar metadata with no extra image pixels or disk labels."""

    def __init__(self, *args, **kwargs):
        _check_version()
        if not kwargs.get("augment", False) or kwargs.get("task", "detect") != "detect":
            raise ValueError("IgnoreYOLODataset is restricted to detection training")
        if kwargs.get("single_cls", False) or kwargs.get("classes") is not None:
            raise ValueError("Class filtering/single_cls would corrupt ignore pseudo-labels")
        if kwargs.get("fraction", 1.0) != 1.0:
            raise ValueError("R04 training requires the complete frozen dataset")
        data = kwargs.get("data") or {}
        sidecar = Path(data.get("r04_ignore_sidecar", ""))
        if not sidecar.is_absolute() or not sidecar.is_file():
            raise ValueError("data.r04_ignore_sidecar must be an existing absolute JSON path")
        raw = sidecar.read_bytes()
        payload = json.loads(raw)
        if payload.get("schema") != SIDECAR_SCHEMA or not isinstance(payload.get("images"), dict):
            raise ValueError("Invalid R04 ignore sidecar schema")
        self.ignore_sidecar_sha256 = hashlib.sha256(raw).hexdigest()
        self.ignore_regions = {}
        self.ignore_reason_counts = {reason: 0 for reason in IGNORE_REASONS}
        for image, regions in payload["images"].items():
            if not Path(image).is_absolute() or not isinstance(regions, list):
                raise ValueError("Ignore image keys must be absolute paths with region lists")
            key = _path_key(image)
            if key in self.ignore_regions:
                raise ValueError("Duplicate normalized ignore image key")
            boxes = []
            for region in regions:
                box = np.asarray(region["bbox_xywhn"], dtype=np.float32)
                if box.shape != (4,) or not np.isfinite(box).all():
                    raise ValueError("Ignore bbox must have four finite normalized xywh coordinates")
                cx, cy, w, h = box
                if w <= 0 or h <= 0 or min(cx-w/2, cy-h/2) < -1e-6 or max(cx+w/2, cy+h/2) > 1+1e-6:
                    raise ValueError("Ignore bbox must be positive and lie inside the image")
                reason = region.get("reason")
                if reason not in IGNORE_REASONS or not region.get("source_instance_id"):
                    raise ValueError("Ignore reason and source instance must be explicit")
                boxes.append(box)
                self.ignore_reason_counts[reason] += 1
            self.ignore_regions[key] = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        super().__init__(*args, **kwargs)

    def get_labels(self):
        labels = super().get_labels()  # stock verified positive labels/cache only
        actual = {_path_key(label["im_file"]) for label in labels}
        if actual != set(self.ignore_regions):
            raise ValueError("Sidecar image set must exactly match frozen training images, including empty lists")
        total = 0
        for label in labels:
            if np.any(label["cls"] < 0):
                raise ValueError("Negative labels must never be present in on-disk YOLO labels/cache")
            boxes = self.ignore_regions[_path_key(label["im_file"])]
            if len(boxes):
                if label.get("segments") or label.get("keypoints") is not None:
                    raise ValueError("R04 ignore adapter supports bbox detection only")
                if label.get("bbox_format") != "xywh" or not label.get("normalized"):
                    raise ValueError("Unexpected upstream cached bbox representation")
                label["cls"] = np.concatenate((label["cls"], np.full((len(boxes), 1), -1, np.float32)))
                label["bboxes"] = np.concatenate((label["bboxes"], boxes.copy()))
                total += len(boxes)
        self.ignore_box_count = total
        return labels

    def build_transforms(self, hyp=None):
        # Identity RandomPerspective still discards 1px boxes: do not use it.
        for name in ("mosaic", "mixup", "cutmix", "copy_paste", "degrees", "translate", "scale", "shear", "perspective"):
            if float(getattr(hyp, name, 0)) != 0:
                raise ValueError(f"R04 ignore geometry requires {name}=0")
        if getattr(hyp, "multi_scale", 0):
            raise ValueError("R04 protocol requires multi_scale=0")
        return Compose([
            LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=True),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(p=hyp.flipud, direction="vertical"),
            RandomFlip(p=hyp.fliplr, direction="horizontal"),
            Format(bbox_format="xywh", normalize=True, batch_idx=True, bgr=hyp.bgr),
        ])


def _anchors_inside_boxes(anchor_xy, boxes_xyxy):
    """One boolean per anchor; inclusive boundaries consistently favour ignore."""
    if boxes_xyxy.numel() == 0:
        return torch.zeros(anchor_xy.shape[0], dtype=torch.bool, device=anchor_xy.device)
    inside = ((anchor_xy[:, None, :] >= boxes_xyxy[None, :, :2]) &
              (anchor_xy[:, None, :] <= boxes_xyxy[None, :, 2:])).all(dim=-1)
    return inside.any(dim=1)


class _IgnoreAwareBCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.BCEWithLogitsLoss(reduction="none")
        self.suppress_anchor = None
        self.assigned_anchor = None
        self.last_suppressed_anchor = None

    def forward(self, logits, target_scores):
        loss = self.base(logits, target_scores)
        if self.suppress_anchor is None:
            self.last_suppressed_anchor = None
            return loss
        if self.suppress_anchor.shape != logits.shape[:2]:
            raise ValueError("Ignore anchor mask shape mismatch")
        # Keep every class loss at an assigned foreground anchor, including its
        # other-class negatives. Positive region exclusion is already upstream.
        assigned = target_scores.gt(0).any(dim=-1)
        if self.assigned_anchor is not None:
            assigned = assigned | self.assigned_anchor.bool()
        suppressed = self.suppress_anchor & ~assigned
        self.last_suppressed_anchor = suppressed.detach()
        return loss.masked_fill(suppressed[..., None], 0)


class _CaptureForegroundAssigner(nn.Module):
    """Observe stock TAL's exact foreground mask without changing its outputs."""

    def __init__(self, assigner, bce):
        super().__init__()
        self.base = assigner
        self.bce = bce

    def forward(self, *args, **kwargs):
        output = self.base(*args, **kwargs)
        self.bce.assigned_anchor = output[3].detach()
        return output


class IgnoreDetectionLoss(v8DetectionLoss):
    """Stock TAL, bbox and DFL with explicit training-only negative BCE ignore."""

    def __init__(self, model):
        _check_version()
        if getattr(model, "end2end", False):
            raise ValueError("R04 adapter only supports YOLO11-style non-end-to-end detection")
        super().__init__(model)
        self.bce = _IgnoreAwareBCE()
        self.assigner = _CaptureForegroundAssigner(self.assigner, self.bce)

    def get_assigned_targets_and_loss(self, preds, batch):
        classes = batch["cls"].view(-1)
        if ((classes < 0) & (classes != -1)).any() or (classes >= self.nc).any():
            raise ValueError("Only class IDs 0..nc-1 and in-memory ignore ID -1 are valid")
        ignored = classes == -1
        self.bce.suppress_anchor = None
        self.bce.assigned_anchor = None
        if not ignored.any():
            return super().get_assigned_targets_and_loss(preds, batch)
        keep = ~ignored
        positive_batch = dict(batch)
        for key in ("cls", "bboxes", "batch_idx"):
            positive_batch[key] = batch[key][keep]
        anchor, strides = make_anchors(preds["feats"], self.stride, 0.5)
        anchor_xy = anchor * strides
        image_hw = torch.tensor(preds["feats"][0].shape[2:], device=anchor.device, dtype=anchor.dtype) * self.stride[0]
        all_boxes = xywh2xyxy(batch["bboxes"].to(anchor.device, anchor.dtype)) * image_hw[[1, 0, 1, 0]]
        image_indices = batch["batch_idx"].view(-1)
        masks = []
        for image_idx in range(preds["scores"].shape[0]):
            belongs = image_indices == image_idx
            in_ignore = _anchors_inside_boxes(anchor_xy, all_boxes[belongs & ignored])
            in_positive = _anchors_inside_boxes(anchor_xy, all_boxes[belongs & keep])
            masks.append(in_ignore & ~in_positive)
        self.bce.suppress_anchor = torch.stack(masks, dim=0)
        try:
            return super().get_assigned_targets_and_loss(preds, positive_batch)
        finally:
            # The graph retains its own mask; avoid stale masks across batches.
            self.bce.suppress_anchor = None


class IgnoreDetectionTrainer(DetectionTrainer):
    """Use via YOLO(...).train(trainer=IgnoreDetectionTrainer, data=...)."""

    def build_dataset(self, img_path, mode="train", batch=None):
        if mode != "train":
            return super().build_dataset(img_path, mode, batch)
        if self.args.task != "detect" or self.args.compile or self.args.distill_model is not None:
            raise ValueError("R04 requires ordinary eager detection training without distillation")
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return IgnoreYOLODataset(
            img_path=img_path, imgsz=self.args.imgsz, batch_size=batch,
            augment=True, hyp=self.args, rect=self.args.rect,
            cache=self.args.cache or None, single_cls=self.args.single_cls,
            stride=gs, pad=0.0, prefix=colorstr("train: "), task="detect",
            classes=self.args.classes, data=self.data, fraction=self.args.fraction,
        )

    def _setup_train(self):
        _check_version()
        super()._setup_train()  # EMA is created with a standard DetectionModel.
        model = unwrap_model(self.model)
        if isinstance(getattr(unwrap_model(self.ema.ema), "criterion", None), IgnoreDetectionLoss):
            raise RuntimeError("EMA must remain independent of the training-only criterion")
        model.criterion = IgnoreDetectionLoss(model)

    def get_class_counts(self):
        classes = np.concatenate([label["cls"].flatten() for label in self.train_loader.dataset.labels])
        return np.bincount(classes[classes >= 0].astype(int), minlength=self.data["nc"]).astype(np.float32)

    def plot_training_labels(self):
        # Never send pseudo class -1 to class-name lookup or class histograms.
        from ultralytics.utils.plotting import plot_labels
        labels = self.train_loader.dataset.labels
        boxes = np.concatenate([lb["bboxes"][lb["cls"].reshape(-1) >= 0] for lb in labels])
        classes = np.concatenate([lb["cls"][lb["cls"].reshape(-1) >= 0] for lb in labels])
        plot_labels(boxes, classes.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def plot_training_samples(self, batch, ni):
        visible = dict(batch)
        keep = batch["cls"].view(-1) >= 0
        for key in ("cls", "bboxes", "batch_idx"):
            visible[key] = batch[key][keep]
        return super().plot_training_samples(visible, ni)
