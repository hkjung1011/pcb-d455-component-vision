"""Native PCBVision semantic IoU evaluation, never converted-instance mAP.

Native labels: 0 other, 1 IC, 2 electrolytic capacitor, 3 connector region.
YOLO predictions: 0 IC, 1 electrolytic capacitor, 2 connector region.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image

from .geometry import generate_tiles

MODEL_NAMES = ("IC", "electrolytic_capacitor", "connector_region")
SEMANTIC_NAMES = ("other",) + MODEL_NAMES


def _ordered_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        try:
            parsed = {int(key): value for key, value in names.items()}
        except (ValueError, TypeError) as exc:
            raise ValueError("Class-name dictionary must use IDs 0, 1, 2") from exc
        if len(parsed) != len(names) or sorted(parsed) != list(range(len(parsed))):
            raise ValueError("Class-name dictionary must use consecutive class IDs")
        return [parsed[key] for key in sorted(parsed)]
    if isinstance(names, (list, tuple)):
        return list(names)
    raise ValueError("Model/manifest must expose an ordered three-class name mapping")


def _check_names(names: Any, origin: str) -> None:
    values = _ordered_names(names)
    if len(values) != 3 or any(not isinstance(value, str) for value in values):
        raise ValueError(f"{origin}: exactly three PCBVision foreground classes required")
    if tuple(value.lower() for value in values) != tuple(value.lower() for value in MODEL_NAMES):
        raise ValueError(f"{origin}: class IDs must map to {MODEL_NAMES}, got {values}")


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def confusion_counts(ground_truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Return 4x4 pixel confusion, rows=ground truth, columns=prediction."""
    truth = np.asarray(ground_truth)
    pred = np.asarray(prediction)
    if truth.ndim != 2 or truth.shape != pred.shape:
        raise ValueError("Semantic ground truth and prediction must be matching 2D arrays")
    for name, array in (("ground truth", truth), ("prediction", pred)):
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{name} must contain integer semantic IDs")
        if array.size == 0 or int(array.min()) < 0 or int(array.max()) > 3:
            raise ValueError(f"{name} has IDs outside native 0..3 or no pixels")
    pairs = truth.astype(np.int64).ravel() * 4 + pred.astype(np.int64).ravel()
    return np.bincount(pairs, minlength=16).reshape(4, 4)


def summarize_confusion(confusion: np.ndarray) -> dict[str, Any]:
    counts = np.asarray(confusion)
    if counts.shape != (4, 4) or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
        raise ValueError("Confusion must be a nonnegative integer 4x4 array")
    true_positive = np.diag(counts).astype(np.float64)
    union = counts.sum(axis=1) + counts.sum(axis=0) - np.diag(counts)
    ious = [float(true_positive[index] / union[index]) if union[index] else None for index in range(4)]
    foreground = [value for value in ious[1:] if value is not None]
    all_present = [value for value in ious if value is not None]
    total = int(counts.sum())
    return {
        "foreground_miou": float(np.mean(foreground)) if foreground else None,
        "miou_including_background": float(np.mean(all_present)) if all_present else None,
        "class_iou": dict(zip(SEMANTIC_NAMES, ious)),
        "class_confusion": counts.tolist(),
        "confusion_axes": {"rows": "ground_truth", "columns": "prediction", "classes": list(SEMANTIC_NAMES)},
        "class_ground_truth_pixels": dict(zip(SEMANTIC_NAMES, map(int, counts.sum(axis=1)))),
        "class_predicted_pixels": dict(zip(SEMANTIC_NAMES, map(int, counts.sum(axis=0)))),
        "foreground_classes_with_nonzero_union": len(foreground),
        "absent_class_policy": "IoU=null when union=0; exclude that class from aggregate mean",
        "pixel_accuracy": float(true_positive.sum() / total) if total else None,
        "evaluated_pixels": total,
    }


def rasterize_instances(
    polygons: Sequence[Any], class_ids: Any, confidences: Any,
    image_shape: Sequence[int], *, conf: float = 0.25,
    offset: tuple[int, int] = (0, 0), labels: np.ndarray | None = None,
    scores: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Paint original-coordinate polygons; highest confidence owns each pixel.

    The temporary binary mask is limited to each polygon's clipped bounding box.
    Passing labels/scores merges overlapping tiles without full-image instance masks.
    Exact confidence ties retain the earlier prediction in deterministic tile order.
    """
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0 or not math.isfinite(conf) or not 0 <= conf <= 1:
        raise ValueError("Positive image dimensions and confidence in [0,1] required")
    if labels is None:
        labels = np.zeros((height, width), dtype=np.uint8)
    if scores is None:
        scores = np.full((height, width), -1.0, dtype=np.float32)
    if labels.shape != (height, width) or scores.shape != (height, width):
        raise ValueError("Stitch arrays do not match image dimensions")
    classes = _numpy(class_ids).reshape(-1)
    confidence_values = _numpy(confidences).reshape(-1)
    if len(polygons) != len(classes) or len(polygons) != len(confidence_values):
        raise ValueError("Polygon, class and confidence counts differ")
    for polygon, class_value, confidence_value in zip(polygons, classes, confidence_values):
        confidence = float(confidence_value)
        class_number = float(class_value)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Prediction confidence must be finite in [0,1]")
        if not math.isfinite(class_number) or not class_number.is_integer() or not 0 <= class_number < 3:
            raise ValueError("Prediction class ID is incompatible with PCBVision three-class mapping")
        if confidence < conf:
            continue
        points = np.asarray(polygon, dtype=np.float64)
        if points.size == 0:
            continue
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.all(np.isfinite(points)):
            raise ValueError("Predicted polygon must have >=3 finite x/y points")
        points = points + np.asarray(offset, dtype=np.float64)
        x1 = max(0, int(np.floor(points[:, 0].min())))
        y1 = max(0, int(np.floor(points[:, 1].min())))
        x2 = min(width, int(np.ceil(points[:, 0].max())) + 1)
        y2 = min(height, int(np.ceil(points[:, 1].max())) + 1)
        if x2 <= x1 or y2 <= y1:
            continue
        local = np.rint(points - (x1, y1))
        if np.any(np.abs(local) > 1_000_000):
            raise ValueError("Prediction polygon coordinates are unreasonably far outside image")
        binary = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.fillPoly(binary, [local.astype(np.int32)], 1)
        destination_scores = scores[y1:y2, x1:x2]
        stored_confidence = np.float32(confidence)
        replace = (binary != 0) & (stored_confidence > destination_scores)
        labels[y1:y2, x1:x2][replace] = int(class_number) + 1
        destination_scores[replace] = stored_confidence
    return labels, scores


def _path(base: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Source image/native semantic mask must have a nonempty path")
    result = Path(value).expanduser()
    return (result if result.is_absolute() else base / result).resolve()


def _stress_image(image: np.ndarray, condition: str) -> np.ndarray:
    if condition == "native":
        return image
    if condition == "half_resolution":
        height, width = image.shape[:2]
        reduced = cv2.resize(image, (max(1, width // 2), max(1, height // 2)), interpolation=cv2.INTER_AREA)
        return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_LINEAR)
    if condition in {"blur", "moderate_blur"}:
        return cv2.GaussianBlur(image, (5, 5), sigmaX=1.2, sigmaY=1.2)
    raise ValueError("condition must be native, half_resolution, blur or moderate_blur")


def _save_overlay(image: np.ndarray, truth: np.ndarray, pred: np.ndarray, path: Path) -> None:
    # BGR colours. Panels: observed input | native GT | semantic prediction.
    colours = np.array([[0, 0, 0], [70, 80, 230], [30, 190, 240], [230, 200, 30]], dtype=np.uint8)
    panels = [image]
    for mask in (truth, pred):
        overlay = cv2.addWeighted(image, 0.6, colours[mask], 0.4, 0)
        overlay[mask == 0] = image[mask == 0]
        panels.append(overlay)
    scale = min(1.0, 520 / image.shape[1], 1000 / image.shape[0])
    target_size = (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale)))
    montage = np.concatenate([cv2.resize(panel, target_size, interpolation=cv2.INTER_AREA) for panel in panels], axis=1)
    ok, encoded = cv2.imencode(".png", montage)
    if not ok:
        raise OSError("Could not encode semantic evaluation preview")
    path.write_bytes(encoded.tobytes())


def _timings(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean_ms": float(np.mean(values)) if values else None,
        "p50_ms": float(np.percentile(values, 50)) if values else None,
        "p95_ms": float(np.percentile(values, 95)) if values else None,
        "total_ms": float(sum(values)),
    }


def evaluate_native_semantic(
    model: Any, manifest_path: str | Path, split: str = "val", imgsz: int = 1024,
    conf: float = 0.25, tile_size: int | tuple[int, float] | None = None,
    device: Any = 0, condition: str = "native", output_dir: str | Path | None = None,
    retina_masks: bool = False,
) -> dict[str, Any]:
    """Evaluate native semantic masks using class unions from model polygons.

    Threshold is explicit and must be selected on validation, never on test.
    Optional tile_size=int uses 20% overlap; (size, overlap_fraction) overrides it.
    retina_masks=False avoids native-resolution GPU masks by default; masks.xy
    still maps polygon coordinates back to each original input crop.
    Public data/stress simulations do not establish D455 deployment performance.
    """
    _check_names(getattr(model, "names", None), "model")
    if split not in {"val", "test"}:
        raise ValueError("Semantic evaluation requires the val or test split")
    if isinstance(imgsz, bool) or not isinstance(imgsz, int) or imgsz <= 0:
        raise ValueError("imgsz must be a positive integer")
    if not math.isfinite(conf) or not 0 <= conf <= 1:
        raise ValueError("confidence threshold must be in [0,1]")
    if not isinstance(retina_masks, bool):
        raise ValueError("retina_masks must be a boolean")
    if condition not in {"native", "half_resolution", "blur", "moderate_blur"}:
        raise ValueError("Unsupported image condition")
    path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("records"), list):
        raise ValueError("Manifest must contain a records list")
    if "names" in manifest:
        _check_names(manifest["names"], "manifest")
    group_split: dict[str, str] = {}
    source_split: dict[Path, str] = {}
    for record in manifest["records"]:
        if not isinstance(record, dict) or record.get("split") not in {"train", "val", "test"}:
            continue
        record_split = record["split"]
        group = record.get("group_id")
        if isinstance(group, str) and group.strip():
            if group in group_split and group_split[group] != record_split:
                raise ValueError(f"Physical board/group leakage across splits: {group}")
            group_split[group] = record_split
        if record.get("source_image_path"):
            source = _path(path.parent, record["source_image_path"])
            if source in source_split and source_split[source] != record_split:
                raise ValueError(f"Source image appears across splits: {source}")
            source_split[source] = record_split
    records = [record for record in manifest["records"] if isinstance(record, dict) and record.get("split") == split]
    if not records:
        raise ValueError(f"No records in {split}")
    output = Path(output_dir).expanduser().resolve() if output_dir is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    confusion = np.zeros((4, 4), dtype=np.int64)
    samples: list[dict[str, Any]] = []
    groups: set[str] = set()
    seen_paths: set[Path] = set()
    call_ms: list[float] = []
    image_ms: list[float] = []
    started = time.perf_counter()
    for record_index, record in enumerate(records):
        began_image = time.perf_counter()
        image_path = _path(path.parent, record.get("source_image_path"))
        mask_path = _path(path.parent, record.get("native_semantic_mask_path"))
        group = record.get("group_id")
        if not isinstance(group, str) or not group.strip():
            raise ValueError("Every evaluation record needs a physical board/group ID")
        if image_path in seen_paths:
            raise ValueError(f"Repeated source image in evaluation split: {image_path}")
        seen_paths.add(image_path)
        groups.add(group)
        with Image.open(image_path) as opened:
            image = cv2.cvtColor(np.asarray(opened.convert("RGB")), cv2.COLOR_RGB2BGR)
        with Image.open(mask_path) as opened:
            truth = np.asarray(opened).copy()
        if truth.ndim != 2 or truth.shape != image.shape[:2]:
            raise ValueError(f"Native semantic mask must match original image dimensions: {mask_path}")
        if not np.issubdtype(truth.dtype, np.integer) or truth.size == 0 or int(truth.min()) < 0 or int(truth.max()) > 3:
            raise ValueError(f"Native semantic mask contains values outside 0..3: {mask_path}")
        truth = truth.astype(np.uint8, copy=False)
        observed = _stress_image(image, condition)
        prediction = np.zeros(truth.shape, dtype=np.uint8)
        confidence_canvas = np.full(truth.shape, -1.0, dtype=np.float32)
        if tile_size is None:
            tiles = [(observed, (0, 0))]
        else:
            size, overlap = tile_size if isinstance(tile_size, tuple) else (tile_size, 0.2)
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ValueError("tile_size must be a positive integer")
            tiles = [(tile.extract(observed), (tile.bounds.x1, tile.bounds.y1)) for tile in generate_tiles(observed.shape, tile_size=size, overlap=overlap)]
        image_call_ms: list[float] = []
        detected_polygons = 0
        skipped_degenerate_masks = 0
        for crop, offset in tiles:
            began_prediction = time.perf_counter()
            results = model.predict(source=crop, imgsz=imgsz, conf=conf, device=device, verbose=False, retina_masks=retina_masks, max_det=1000)
            duration = (time.perf_counter() - began_prediction) * 1000
            call_ms.append(duration)
            image_call_ms.append(duration)
            if len(results) != 1:
                raise ValueError("Expected exactly one prediction result per image crop")
            result = results[0]
            masks = getattr(result, "masks", None)
            boxes = getattr(result, "boxes", None)
            if masks is None:
                if boxes is not None and len(boxes):
                    raise ValueError("Model returned boxes without masks; segmentation model required")
                continue
            if boxes is None:
                raise ValueError("Segmentation masks have no matching class/confidence boxes")
            polygons = masks.xy
            detected_polygons += len(polygons)
            classes = _numpy(boxes.cls).reshape(-1)
            confidences = _numpy(boxes.conf).reshape(-1)
            if len(polygons) != len(classes) or len(polygons) != len(confidences):
                raise ValueError("Polygon, class and confidence counts differ")
            if not np.all(np.isfinite(classes)) or np.any(classes != np.floor(classes)) or np.any((classes < 0) | (classes > 2)):
                raise ValueError("Prediction class ID is incompatible with PCBVision three-class mapping")
            if not np.all(np.isfinite(confidences)) or np.any((confidences < 0) | (confidences > 1)):
                raise ValueError("Prediction confidence must be finite in [0,1]")
            valid_indices = []
            valid_polygons = []
            for polygon_index, polygon in enumerate(polygons):
                points = np.asarray(polygon, dtype=np.float64)
                # masks.xy can expose an empty or line/point contour for a box.
                # Such a contour has no polygon area; preserve the indices of
                # the remaining masks instead of shifting their class/confidence.
                if points.shape == (0,):
                    skipped_degenerate_masks += 1
                    continue
                if points.ndim != 2 or points.shape[1] != 2:
                    raise ValueError("Predicted polygon must be an N x 2 array")
                if not np.all(np.isfinite(points)):
                    raise ValueError("Predicted polygon contains nonfinite coordinates")
                if len(points) < 3:
                    skipped_degenerate_masks += 1
                    continue
                valid_indices.append(polygon_index)
                valid_polygons.append(points)
            rasterize_instances(valid_polygons, classes[valid_indices], confidences[valid_indices], image.shape, conf=conf, offset=offset, labels=prediction, scores=confidence_canvas)
        sample_confusion = confusion_counts(truth, prediction)
        confusion += sample_confusion
        image_duration = (time.perf_counter() - began_image) * 1000
        image_ms.append(image_duration)
        sample = {
            "source_image_path": str(image_path), "native_semantic_mask_path": str(mask_path),
            "group_id": group, "height": int(image.shape[0]), "width": int(image.shape[1]),
            "tile_count": len(tiles), "predicted_polygon_count_before_union": detected_polygons,
            "skipped_degenerate_predicted_masks": skipped_degenerate_masks,
            "image_elapsed_ms": image_duration, "prediction_calls_ms": image_call_ms,
            **summarize_confusion(sample_confusion),
        }
        if output is not None and record_index < 3:
            overlay_path = output / f"preview_{record_index:03d}.png"
            _save_overlay(observed, truth, prediction, overlay_path)
            sample["preview_path"] = str(overlay_path)
        samples.append(sample)
    report = {
        "schema": "pcb-native-semantic-evaluation-v1",
        "metric_type": "native_semantic_IoU",
        "instance_map_eligible": False,
        "instance_metric_note": "Native source has semantic class IDs, not verified instance IDs; no box/mask mAP computed.",
        "manifest_path": str(path), "split": split,
        "domain": manifest.get("domain", "public_proxy"),
        "condition": condition, "condition_is_simulated": condition != "native",
        "condition_note": "Native source image" if condition == "native" else "Synthetic image degradation with unchanged native GT; not a real D455 measurement",
        "confidence_threshold": float(conf), "imgsz": imgsz,
        "retina_masks": retina_masks,
        "tile_size": tile_size, "tile_overlap": (tile_size[1] if isinstance(tile_size, tuple) else 0.2) if tile_size is not None else None,
        "overlap_policy": "Highest-confidence polygon owns pixel; equal confidence keeps earlier tile/prediction",
        "image_count": len(records), "board_count": len(groups),
        "skipped_degenerate_predicted_masks": sum(sample["skipped_degenerate_predicted_masks"] for sample in samples),
        "timing_stats": {"prediction_calls": _timings(call_ms), "images": _timings(image_ms), "total_seconds": time.perf_counter() - started, "warmup_excluded": False},
        "sample_details": samples,
        **summarize_confusion(confusion),
    }
    if output is not None:
        np.save(output / "confusion.npy", confusion, allow_pickle=False)
        (output / "confusion.json").write_text(json.dumps({"classes": list(SEMANTIC_NAMES), "rows": "ground_truth", "columns": "prediction", "counts": confusion.tolist()}, indent=2) + "\n", encoding="utf-8")
        (output / "semantic_evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return report
