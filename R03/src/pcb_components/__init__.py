"""Native-pixel PCB image quality and tiled-instance geometry utilities."""

from .geometry import (
    CroppedMask,
    Rect,
    Tile,
    deduplicate_instances,
    generate_tiles,
    mask_iou,
    restore_bbox,
    restore_mask,
    validate_roi,
)
from .quality import ImageQualityReport, analyze_image_quality

__all__ = [
    "CroppedMask", "Rect", "Tile", "deduplicate_instances", "generate_tiles",
    "mask_iou", "restore_bbox", "restore_mask", "validate_roi",
    "ImageQualityReport", "analyze_image_quality",
]
