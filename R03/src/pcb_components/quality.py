"""Read-only image quality measurements; no enhancement or camera acceptance gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class ImageQualityReport:
    width: int
    height: int
    channels: int
    input_order: str
    laplacian_variance: float
    brightness_mean_0_255: float
    brightness_p05_0_255: float
    brightness_p95_0_255: float
    saturation_mean_0_255: float | None
    high_saturation_pixel_pct: float | None
    near_black_pixel_pct: float
    near_white_pixel_pct: float
    any_channel_low_clip_pct: float
    any_channel_high_clip_pct: float
    clip_low: int
    clip_high: int
    near_black_threshold: int
    near_white_threshold: int
    saturation_threshold: int

    def to_dict(self) -> dict:
        return asdict(self)


def _u8_threshold(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value <= 255:
        raise ValueError(f"{name} must be in [0, 255]")
    return int(value)


def analyze_image_quality(
    image: np.ndarray,
    *,
    input_order: str = "BGR",
    clip_low: int = 0,
    clip_high: int = 255,
    near_black_threshold: int = 5,
    near_white_threshold: int = 250,
    saturation_threshold: int = 250,
) -> ImageQualityReport:
    """Measure native uint8 grayscale or 3-channel BGR/RGB pixels without changes.

    Brightness uses OpenCV grayscale luminance; saturation uses OpenCV uint8
    HSV S. Clip percentages count pixels with ANY channel <= clip_low or
    >= clip_high (not necessarily sensor clipping). Near-black/white percentages
    use grayscale luminance. Grayscale saturation fields are None.

    Laplacian variance is a relative blur heuristic that also responds to PCB
    texture, resolution, illumination and noise. No metric is a focus/exposure
    PASS, nor proof that tiny components can be classified or segmented.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")
    if image.dtype != np.uint8:
        raise TypeError("image must be uint8; convert explicitly without silently scaling data")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError("image must be HxW grayscale or HxWx3 BGR/RGB")
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError("quality measurements require at least 2x2 pixels")
    if input_order not in ("BGR", "RGB"):
        raise ValueError("input_order must be BGR or RGB")
    clip_low = _u8_threshold(clip_low, "clip_low")
    clip_high = _u8_threshold(clip_high, "clip_high")
    near_black_threshold = _u8_threshold(near_black_threshold, "near_black_threshold")
    near_white_threshold = _u8_threshold(near_white_threshold, "near_white_threshold")
    saturation_threshold = _u8_threshold(saturation_threshold, "saturation_threshold")
    if clip_low >= clip_high or near_black_threshold >= near_white_threshold:
        raise ValueError("low thresholds must be strictly less than their high thresholds")
    try:
        import cv2
    except ImportError as error:
        raise ImportError("analyze_image_quality requires OpenCV (opencv-python)") from error

    if image.ndim == 2:
        gray = image
        channels = 1
        saturation_mean = high_saturation = None
        low_clipped = image <= clip_low
        high_clipped = image >= clip_high
    else:
        channels = 3
        gray_code = cv2.COLOR_BGR2GRAY if input_order == "BGR" else cv2.COLOR_RGB2GRAY
        hsv_code = cv2.COLOR_BGR2HSV if input_order == "BGR" else cv2.COLOR_RGB2HSV
        gray = cv2.cvtColor(image, gray_code)
        saturation = cv2.cvtColor(image, hsv_code)[:, :, 1]
        saturation_mean = float(saturation.mean())
        high_saturation = float(np.mean(saturation >= saturation_threshold) * 100.0)
        low_clipped = np.any(image <= clip_low, axis=2)
        high_clipped = np.any(image >= clip_high, axis=2)
    p05, p95 = np.percentile(gray, [5, 95])
    return ImageQualityReport(
        width=int(image.shape[1]), height=int(image.shape[0]), channels=channels,
        input_order="GRAY" if channels == 1 else input_order,
        laplacian_variance=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        brightness_mean_0_255=float(gray.mean()),
        brightness_p05_0_255=float(p05), brightness_p95_0_255=float(p95),
        saturation_mean_0_255=saturation_mean, high_saturation_pixel_pct=high_saturation,
        near_black_pixel_pct=float(np.mean(gray <= near_black_threshold) * 100.0),
        near_white_pixel_pct=float(np.mean(gray >= near_white_threshold) * 100.0),
        any_channel_low_clip_pct=float(low_clipped.mean() * 100.0),
        any_channel_high_clip_pct=float(high_clipped.mean() * 100.0),
        clip_low=clip_low, clip_high=clip_high,
        near_black_threshold=near_black_threshold, near_white_threshold=near_white_threshold,
        saturation_threshold=saturation_threshold,
    )
