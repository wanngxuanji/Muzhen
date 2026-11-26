"""White-text-with-dark-outline (白字黑边) subtitle mask and frame metrics."""

from __future__ import annotations

import cv2
import numpy as np

from transapp.core.models import ExtractConfig

_MIN_OUTLINE_RATIO = 0.12
_GIANT_BLOB_HEIGHT_RATIO = 0.85
_GIANT_BLOB_WIDTH_RATIO = 0.50
_GIANT_BLOB_OUTLINE_RATIO = 0.35
_GIANT_BLOB_FILL_RATIO = 0.75


def _outline_radius(roi_h: int) -> int:
    """Scale outline-ring dilation radius in [2, 4] px with ROI height."""
    return max(2, min(4, round(roi_h / 24)))


def build_stroke_mask(bgr_roi: np.ndarray, config: ExtractConfig) -> np.ndarray:
    """Extract a binary uint8 (0/255) mask of white strokes with dark outlines."""
    if bgr_roi.ndim < 2 or bgr_roi.shape[0] == 0 or bgr_roi.shape[1] == 0 or bgr_roi.size == 0:
        shape = bgr_roi.shape[:2] if bgr_roi.ndim >= 2 else (0, 0)
        return np.zeros(shape, dtype=np.uint8)

    roi_h, roi_w = bgr_roi.shape[:2]
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    white_bin = ((val >= config.white_v_min) & (sat <= config.white_s_max)).astype(np.uint8)
    if not np.any(white_bin):
        return np.zeros((roi_h, roi_w), dtype=np.uint8)

    dark_mask = val <= config.outline_v_max
    radius = _outline_radius(roi_h)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_bin, connectivity=8)
    out = np.zeros((roi_h, roi_w), dtype=np.uint8)

    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < config.min_blob_px:
            continue

        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])

        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(roi_w, x + w + radius)
        y2 = min(roi_h, y + h + radius)

        comp = (labels[y1:y2, x1:x2] == label_idx).astype(np.uint8)
        dilated = cv2.dilate(comp, kernel)
        # Exclude neighboring white strokes so closely packed CJK strokes do not dilute ring support.
        ring = (dilated > 0) & (white_bin[y1:y2, x1:x2] == 0)
        ring_px = int(np.count_nonzero(ring))
        if ring_px == 0:
            continue

        dark_px = int(np.count_nonzero(ring & dark_mask[y1:y2, x1:x2]))
        outline_ratio = dark_px / ring_px

        if h > _GIANT_BLOB_HEIGHT_RATIO * roi_h:
            fill_ratio = area / float(max(1, w * h))
            is_wide_solid = (
                w > _GIANT_BLOB_WIDTH_RATIO * roi_h and fill_ratio > _GIANT_BLOB_FILL_RATIO
            )
            if outline_ratio < _GIANT_BLOB_OUTLINE_RATIO or is_wide_solid:
                continue

        if outline_ratio >= _MIN_OUTLINE_RATIO:
            out[y1:y2, x1:x2][comp > 0] = 255

    return out


def mask_fg_ratio(mask: np.ndarray) -> float:
    """Return fraction of non-zero foreground pixels in a mask."""
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(mask.size)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray, dilate_px: int = 1) -> float:
    """Compute Intersection-over-Union between two masks, optionally tolerant by dilate_px."""
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"Mask shapes must match, got {mask_a.shape} and {mask_b.shape}")
    if mask_a.size == 0:
        return 0.0

    bin_a = (mask_a > 0).astype(np.uint8)
    bin_b = (mask_b > 0).astype(np.uint8)

    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bin_a = cv2.dilate(bin_a, kernel)
        bin_b = cv2.dilate(bin_b, kernel)

    fg_a = bin_a > 0
    fg_b = bin_b > 0
    union = int(np.count_nonzero(fg_a | fg_b))
    if union == 0:
        return 0.0
    inter = int(np.count_nonzero(fg_a & fg_b))
    return float(inter) / float(union)


def frame_sharpness(bgr_roi: np.ndarray) -> float:
    """Return Laplacian variance as a sharpness score for peak-frame selection."""
    if bgr_roi.ndim < 2 or bgr_roi.shape[0] == 0 or bgr_roi.shape[1] == 0 or bgr_roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY) if bgr_roi.ndim == 3 else bgr_roi
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
