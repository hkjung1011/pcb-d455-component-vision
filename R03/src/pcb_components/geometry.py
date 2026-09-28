"""Tile and mask coordinates without resampling or full-frame mask allocation.

All rectangles use pixel-edge xyxy coordinates: left/top inclusive, right/bottom
exclusive. Code is original; tiling and mask-IoU suppression are standard
techniques (concept references are listed in the accompanying README).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from typing import Iterable, Sequence

import numpy as np


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    return int(value)


def _image_hw(image_shape: Sequence[int]) -> tuple[int, int]:
    if len(image_shape) < 2:
        raise ValueError("image_shape must contain at least height and width")
    height = _integer(image_shape[0], "image height")
    width = _integer(image_shape[1], "image width")
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    return height, width


@dataclass(frozen=True)
class Rect:
    """Nonempty integer xyxy rectangle; negative local origins are permitted."""

    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        for name in ("x1", "y1", "x2", "y2"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("rectangle must have positive width and height")

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2

    def intersection(self, other: Rect) -> Rect | None:
        x1, y1 = max(self.x1, other.x1), max(self.y1, other.y1)
        x2, y2 = min(self.x2, other.x2), min(self.y2, other.y2)
        return None if x2 <= x1 or y2 <= y1 else Rect(x1, y1, x2, y2)


def _rect(value: Rect | Sequence[int]) -> Rect:
    if isinstance(value, Rect):
        return value
    if len(value) != 4:
        raise ValueError("rectangle must have exactly four xyxy values")
    return Rect(*value)


def validate_roi(
    image_shape: Sequence[int], roi: Rect | Sequence[int] | None = None
) -> Rect:
    """Return a valid fixed ROI; reject out-of-frame coordinates instead of clipping."""
    height, width = _image_hw(image_shape)
    result = Rect(0, 0, width, height) if roi is None else _rect(roi)
    if result.x1 < 0 or result.y1 < 0 or result.x2 > width or result.y2 > height:
        raise ValueError(f"ROI {result.as_xyxy()} exceeds image width={width}, height={height}")
    return result


@dataclass(frozen=True)
class Tile:
    index: int
    bounds: Rect
    image_shape: tuple[int, int]

    def __post_init__(self) -> None:
        index = _integer(self.index, "tile index")
        if index < 0:
            raise ValueError("tile index must be nonnegative")
        shape = _image_hw(self.image_shape)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "image_shape", shape)
        object.__setattr__(self, "bounds", validate_roi(shape, self.bounds))

    @property
    def width(self) -> int:
        return self.bounds.width

    @property
    def height(self) -> int:
        return self.bounds.height

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Return a native-pixel view. Copy explicitly if a writable copy is needed."""
        if not isinstance(image, np.ndarray) or image.ndim < 2:
            raise TypeError("image must be a numpy array with at least two dimensions")
        if tuple(image.shape[:2]) != self.image_shape:
            raise ValueError("image dimensions differ from the tile's source dimensions")
        b = self.bounds
        return image[b.y1:b.y2, b.x1:b.x2]


def _axis_starts(start: int, length: int, requested_size: int, overlap: float) -> list[int]:
    size = min(length, requested_size)
    last = start + length - size
    if last == start:
        return [start]
    # floor keeps the effective overlap at least the requested fraction, except
    # for the unavoidable one-pixel stride at extremely high overlap settings.
    stride = max(1, math.floor(size * (1.0 - overlap)))
    starts = list(range(start, last + 1, stride))
    if starts[-1] != last:
        starts.append(last)
    return starts


def generate_tiles(
    image_shape: Sequence[int],
    tile_size: int | Sequence[int] = 1024,
    overlap: float = 0.2,
    roi: Rect | Sequence[int] | None = None,
) -> tuple[Tile, ...]:
    """Cover an image/ROI at native pixels; pair tile_size is (height, width).

    Last tiles align to the right/bottom edge; no duplicate bounds, padding, or
    resizing is introduced. An image smaller than tile_size remains smaller.
    Final edge overlap can exceed the requested overlap.
    """
    shape = _image_hw(image_shape)
    region = validate_roi(shape, roi)
    if isinstance(tile_size, Integral) and not isinstance(tile_size, bool):
        tile_height = tile_width = int(tile_size)
    else:
        if not isinstance(tile_size, Sequence) or len(tile_size) != 2:
            raise ValueError("tile_size must be a positive integer or (height, width)")
        tile_height = _integer(tile_size[0], "tile height")
        tile_width = _integer(tile_size[1], "tile width")
    if tile_height <= 0 or tile_width <= 0:
        raise ValueError("tile dimensions must be positive")
    if isinstance(overlap, bool) or not isinstance(overlap, Real):
        raise TypeError("overlap must be a real number in [0, 1)")
    overlap = float(overlap)
    if not math.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError("overlap must be finite and in [0, 1)")
    xs = _axis_starts(region.x1, region.width, tile_width, overlap)
    ys = _axis_starts(region.y1, region.height, tile_height, overlap)
    width, height = min(tile_width, region.width), min(tile_height, region.height)
    bounds = (Rect(x, y, x + width, y + height) for y in ys for x in xs)
    return tuple(Tile(index, bound, shape) for index, bound in enumerate(bounds))


def restore_bbox(local_bbox: Sequence[float], tile: Tile) -> tuple[float, float, float, float] | None:
    """Clip a tile-relative xyxy bbox to the tile, then translate to the image.

    Return None for a valid box wholly outside the tile. Reversed/empty or
    nonfinite input boxes are errors. No inference letterbox inversion is done.
    """
    if len(local_bbox) != 4:
        raise ValueError("bbox must have four xyxy coordinates")
    box = np.asarray(local_bbox, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("bbox coordinates must be finite scalars")
    x1, y1, x2, y2 = map(float, box)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must have positive width and height")
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(tile.width), x2), min(float(tile.height), y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1 + tile.bounds.x1, y1 + tile.bounds.y1, x2 + tile.bounds.x1, y2 + tile.bounds.y1


@dataclass(frozen=True)
class CroppedMask:
    """One instance in original-image coordinates; stores only its local mask.

    bounds is the mask-array rectangle, not a separately regressed detector
    bbox. Instances passed to deduplicate_instances must belong to one image.
    """

    class_id: int
    score: float
    bounds: Rect
    mask: np.ndarray = field(compare=False, repr=False)
    source_tile_index: int | None = None
    _pixel_area: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        class_id = _integer(self.class_id, "class_id")
        if class_id < 0:
            raise ValueError("class_id must be nonnegative")
        score = float(self.score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("score must be finite and in [0, 1]")
        bounds = _rect(self.bounds)
        if bounds.x1 < 0 or bounds.y1 < 0:
            raise ValueError("restored mask bounds cannot have negative origins")
        mask = np.asarray(self.mask)
        if mask.dtype != np.bool_ or mask.ndim != 2:
            raise TypeError("CroppedMask requires a two-dimensional boolean mask")
        if mask.shape != (bounds.height, bounds.width):
            raise ValueError("mask dimensions must match bounds exactly")
        area = int(np.count_nonzero(mask))
        if area == 0:
            raise ValueError("CroppedMask must contain at least one foreground pixel")
        if self.source_tile_index is not None:
            index = _integer(self.source_tile_index, "source_tile_index")
            if index < 0:
                raise ValueError("source_tile_index must be nonnegative")
            object.__setattr__(self, "source_tile_index", index)
        # Own storage avoids a tiny mask keeping a full tile/model tensor alive.
        mask = np.array(mask, dtype=np.bool_, order="C", copy=True)
        mask.setflags(write=False)
        object.__setattr__(self, "class_id", class_id)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "_pixel_area", area)

    @property
    def pixel_area(self) -> int:
        return self._pixel_area


def restore_mask(
    local_mask: np.ndarray,
    tile: Tile,
    *,
    class_id: int,
    score: float,
    local_bounds: Rect | Sequence[int] | None = None,
    threshold: float = 0.5,
) -> CroppedMask | None:
    """Translate a native tile mask or explicitly located local crop to image xy.

    local_bounds=None requires shape=(tile.height, tile.width). Otherwise the
    mask shape must equal local_bounds dimensions, measured in tile pixels.
    Out-of-tile portions are clipped. Empty results return None. Input is never
    resized: undo model letterboxing/resizing before calling this function.
    Boolean, 0/1, 0/255 or probability masks are accepted with value > threshold.
    """
    mask = np.asarray(local_mask)
    if mask.ndim != 2 or mask.dtype.kind not in "bifu":
        raise TypeError("local_mask must be a two-dimensional numeric/bool array")
    if not np.isfinite(mask).all():
        raise ValueError("mask contains nonfinite values")
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    local = Rect(0, 0, tile.width, tile.height) if local_bounds is None else _rect(local_bounds)
    if mask.shape != (local.height, local.width):
        raise ValueError("mask shape does not match its native tile/local_bounds dimensions")
    valid = local.intersection(Rect(0, 0, tile.width, tile.height))
    if valid is None:
        return None
    binary = mask[valid.y1-local.y1:valid.y2-local.y1, valid.x1-local.x1:valid.x2-local.x1] > threshold
    rows = np.flatnonzero(binary.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(binary.any(axis=0))
    left, right = int(cols[0]), int(cols[-1]) + 1
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    x = tile.bounds.x1 + valid.x1
    y = tile.bounds.y1 + valid.y1
    return CroppedMask(
        class_id=class_id,
        score=score,
        bounds=Rect(x+left, y+top, x+right, y+bottom),
        mask=binary[top:bottom, left:right],
        source_tile_index=tile.index,
    )


def mask_iou(first: CroppedMask, second: CroppedMask) -> float:
    """Compute foreground IoU using only intersecting crop storage; class independent."""
    overlap = first.bounds.intersection(second.bounds)
    if overlap is None:
        return 0.0
    a = first.mask[
        overlap.y1-first.bounds.y1:overlap.y2-first.bounds.y1,
        overlap.x1-first.bounds.x1:overlap.x2-first.bounds.x1,
    ]
    b = second.mask[
        overlap.y1-second.bounds.y1:overlap.y2-second.bounds.y1,
        overlap.x1-second.bounds.x1:overlap.x2-second.bounds.x1,
    ]
    intersection = int(np.count_nonzero(np.logical_and(a, b)))
    return intersection / (first.pixel_area + second.pixel_area - intersection)


def deduplicate_instances(
    instances: Iterable[CroppedMask], iou_threshold: float = 0.6
) -> list[CroppedMask]:
    """Stable score-ordered, class-aware mask NMS on ONE original image.

    Suppress a lower-score same-class mask at IoU >= threshold. Never union or
    merge masks, use bbox IoU, or discard solely because boxes touch/overlap.
    Partial seam duplicates with low mask IoU may remain: tune on validation
    images and inspect seam cases rather than claiming exact instance identity.
    """
    iou_threshold = float(iou_threshold)
    if not math.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be finite and in (0, 1]")
    candidates = list(instances)
    if any(not isinstance(item, CroppedMask) for item in candidates):
        raise TypeError("all instances must be CroppedMask objects")
    # Python sorting is stable: equal scores retain their original input order.
    candidates.sort(key=lambda item: item.score, reverse=True)
    kept: list[CroppedMask] = []
    by_class: dict[int, list[CroppedMask]] = defaultdict(list)
    for candidate in candidates:
        if any(mask_iou(candidate, prior) >= iou_threshold for prior in by_class[candidate.class_id]):
            continue
        kept.append(candidate)
        by_class[candidate.class_id].append(candidate)
    return kept
