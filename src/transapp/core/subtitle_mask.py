"""White-text-with-dark-outline (白字黑边) subtitle mask, temporal fusion, and frame scoring."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from transapp.core.models import ExtractConfig

_MIN_OUTLINE_RATIO = 0.12
_GIANT_BLOB_HEIGHT_RATIO = 0.85
_GIANT_BLOB_WIDTH_RATIO = 0.50
_GIANT_BLOB_OUTLINE_RATIO = 0.35
_GIANT_BLOB_FILL_RATIO = 0.75
_REF_FRAME_HEIGHT = 1080.0
_FG_PX_MIN_SCALE = 0.00040
_FG_PX_EDGE_SCALE = 0.00018


@dataclass(slots=True)
class _StrokeComponent:
    x: int
    y: int
    w: int
    h: int
    x1: int
    y1: int
    x2: int
    y2: int
    comp_mask: np.ndarray
    touches_top: bool
    touches_bottom: bool
    border_contact_px: int
    has_bilateral_side_outline: bool


def scale_factor_for_frame(frame_h: int | None) -> float:
    """Return resolution scale factor relative to 1080p (1.0 when frame_h is unknown)."""
    if frame_h is None or frame_h <= 0:
        return 1.0
    return max(0.25, float(frame_h) / _REF_FRAME_HEIGHT)


def effective_min_blob_px(config: ExtractConfig, frame_h: int | None = None) -> int:
    """Scale min_blob_px quadratically with frame height when frame_h is known."""
    if frame_h is None or frame_h <= 0:
        return max(1, int(config.min_blob_px))
    s = scale_factor_for_frame(frame_h)
    return max(6, round(float(config.min_blob_px) * s * s))


def effective_dilate_px(frame_h: int | None = None) -> int:
    """Scale morphological tolerance radius with frame height (1px @480p/1080p, 2px @4K)."""
    if frame_h is None or frame_h <= 0:
        return 1
    s = scale_factor_for_frame(frame_h)
    return max(1, min(3, round(1.0 * s)))


def _outline_radius(roi_h: int, frame_h: int | None = None) -> int:
    """Scale outline-ring dilation radius with frame height (or ROI height when unknown)."""
    if frame_h is not None and frame_h > 0:
        s = scale_factor_for_frame(frame_h)
        return max(2, min(8, round(3.0 * s)))
    return max(2, min(4, round(roi_h / 24)))


def fg_pixel_thresholds(
    roi_h: int,
    roi_w: int,
    config: ExtractConfig,
    frame_h: int | None = None,
) -> tuple[int, int]:
    """Compute (min_fg_px, edge_fg_px) invariant to ROI area when frame_h is known."""
    roi_area = max(1, roi_h * roi_w)
    ratio_min_px = max(1, int(round(config.fg_ratio_min * roi_area)))
    ratio_edge_px = max(1, int(round(config.fg_ratio_edge * roi_area)))

    if frame_h is None or frame_h <= 0:
        return (ratio_min_px, min(ratio_edge_px, ratio_min_px))

    scaled_min = max(16, int(round(_FG_PX_MIN_SCALE * frame_h * frame_h)))
    scaled_edge = max(8, int(round(_FG_PX_EDGE_SCALE * frame_h * frame_h)))
    # Use the smaller of ratio-based and frame-scaled pixel counts so wide ROIs never drop
    # short 1-2 character cues ("嗯。", "好！"), while keeping a safe absolute floor.
    fg_min_px = min(ratio_min_px, scaled_min)
    fg_edge_px = min(ratio_edge_px, scaled_edge, fg_min_px)
    return (max(1, fg_min_px), max(1, fg_edge_px))


def _has_directional_outline(
    ring: np.ndarray,
    dark_sub: np.ndarray,
    *,
    touches_top: bool,
    touches_bottom: bool,
    touches_left: bool,
    touches_right: bool,
) -> tuple[bool, bool]:
    """Check dark outline support across unclipped sides.

    Returns (passes_directional_check, has_bilateral_side_outline).
    """
    ring_dark = ring & dark_sub
    rh, rw = ring_dark.shape
    mid_y = max(1, rh // 2)
    mid_x = max(1, rw // 2)

    top_dark = int(np.count_nonzero(ring_dark[:mid_y, :]))
    bottom_dark = int(np.count_nonzero(ring_dark[mid_y:, :]))
    left_dark = int(np.count_nonzero(ring_dark[:, :mid_x]))
    right_dark = int(np.count_nonzero(ring_dark[:, mid_x:]))

    has_bilateral_side = (left_dark >= 2) and (right_dark >= 2)

    # Only evaluate opposite-side support on sides not clipped by the ROI border.
    if touches_top and not touches_bottom:
        # Clipped at top (e.g. claw hanging from y=0): real glyphs still have dark outline on
        # left & right flanks; an unbordered prop with dark tips only at bottom fails this.
        return (has_bilateral_side or (left_dark + right_dark >= 4 and bottom_dark >= 2)), has_bilateral_side
    if touches_bottom and not touches_top:
        return (has_bilateral_side or (left_dark + right_dark >= 4 and top_dark >= 2)), has_bilateral_side
    if touches_left and not touches_right:
        return ((top_dark >= 2 and bottom_dark >= 2) or right_dark >= 2), has_bilateral_side
    if touches_right and not touches_left:
        return ((top_dark >= 2 and bottom_dark >= 2) or left_dark >= 2), has_bilateral_side

    # Interior component: require dark outline on at least 2 quadrants/sides so one-sided
    # background highlights next to a single dark edge are suppressed.
    active_sides = int(top_dark > 0) + int(bottom_dark > 0) + int(left_dark > 0) + int(right_dark > 0)
    return (active_sides >= 2), has_bilateral_side


def build_stroke_mask(
    bgr_roi: np.ndarray,
    config: ExtractConfig,
    *,
    frame_h: int | None = None,
) -> np.ndarray:
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
    if not np.any(dark_mask):
        return np.zeros((roi_h, roi_w), dtype=np.uint8)

    radius = _outline_radius(roi_h, frame_h=frame_h)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    # Pre-filter: white blobs with zero dark pixels within outline radius cannot pass outline check.
    dark_near = cv2.dilate(dark_mask.astype(np.uint8), kernel) > 0
    min_blob_px = effective_min_blob_px(config, frame_h=frame_h)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_bin, connectivity=8)
    kept: list[_StrokeComponent] = []

    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_blob_px:
            continue

        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])

        # Quick skip if bounding box has no dark pixels anywhere nearby
        if not np.any(dark_near[y : y + h, x : x + w]):
            continue

        touches_top = y == 0
        touches_bottom = (y + h) >= roi_h
        touches_left = x == 0
        touches_right = (x + w) >= roi_w
        fill_ratio = area / float(max(1, w * h))

        # Reject wide solid bars/clouds spanning >2.5*roi_h horizontally
        if w > 2.5 * roi_h and fill_ratio > 0.55:
            continue

        # Reject solid background columns/shirts touching both top and bottom borders
        if touches_top and touches_bottom and w > 0.40 * roi_h and fill_ratio > 0.65:
            continue

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

        dark_sub = dark_mask[y1:y2, x1:x2]
        dark_px = int(np.count_nonzero(ring & dark_sub))
        outline_ratio = dark_px / ring_px

        if h > _GIANT_BLOB_HEIGHT_RATIO * roi_h:
            is_wide_solid = (
                w > _GIANT_BLOB_WIDTH_RATIO * roi_h and fill_ratio > _GIANT_BLOB_FILL_RATIO
            )
            if outline_ratio < _GIANT_BLOB_OUTLINE_RATIO or is_wide_solid:
                continue

        if outline_ratio < _MIN_OUTLINE_RATIO:
            continue

        border_contact_px = 0
        if touches_top:
            border_contact_px = max(border_contact_px, int(np.count_nonzero(comp[0, :])))
        if touches_bottom:
            border_contact_px = max(border_contact_px, int(np.count_nonzero(comp[-1, :])))

        passes_dir, has_bilateral_side = _has_directional_outline(
            ring,
            dark_sub,
            touches_top=touches_top,
            touches_bottom=touches_bottom,
            touches_left=touches_left,
            touches_right=touches_right,
        )
        if not passes_dir:
            continue

        # Suppress thick border-clipped props (e.g. claw hanging from y=0 with dark tips only)
        # while preserving real glyphs clipped slightly by a tight ROI.
        if (
            config.suppress_border_blobs
            and (touches_top or touches_bottom)
            and border_contact_px >= max(6, int(0.35 * w))
            and not has_bilateral_side
        ):
            continue

        kept.append(
            _StrokeComponent(
                x=x,
                y=y,
                w=w,
                h=h,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                comp_mask=comp,
                touches_top=touches_top,
                touches_bottom=touches_bottom,
                border_contact_px=border_contact_px,
                has_bilateral_side_outline=has_bilateral_side,
            )
        )

    if not kept:
        return np.zeros((roi_h, roi_w), dtype=np.uint8)

    if config.suppress_border_blobs:
        interior = [c for c in kept if not (c.touches_top or c.touches_bottom)]
        border_touching = [c for c in kept if (c.touches_top or c.touches_bottom)]

        if interior and border_touching:
            band_y1 = min(c.y for c in interior)
            band_y2 = max(c.y + c.h for c in interior)
            med_h = float(np.median([c.h for c in interior]))
            filtered_border: list[_StrokeComponent] = []
            for c in border_touching:
                overlap_y = max(0, min(band_y2, c.y + c.h) - max(band_y1, c.y))
                if overlap_y >= 0.40 * c.h:
                    filtered_border.append(c)
                elif c.has_bilateral_side_outline and c.border_contact_px < max(6, int(0.30 * c.w)):
                    # Allow minor tip touch if close horizontally to the main text line
                    min_dx = min(
                        max(0, max(c.x - (ic.x + ic.w), ic.x - (c.x + c.w))) for ic in interior
                    )
                    if min_dx <= max(2.0 * med_h, 0.08 * roi_w):
                        filtered_border.append(c)
            kept = interior + filtered_border
        elif not interior and border_touching:
            # All components touch top/bottom edge: reject solitary wide-contact non-glyph props
            # (like a hanging claw) while keeping real tight-ROI glyphs (even 1-3 chars).
            if len(border_touching) <= 2:
                kept = [
                    c
                    for c in border_touching
                    if c.has_bilateral_side_outline
                    and not (c.touches_top and c.touches_bottom)
                    and c.border_contact_px < max(12, int(0.50 * c.w))
                ]

    out = np.zeros((roi_h, roi_w), dtype=np.uint8)
    for c in kept:
        out[c.y1 : c.y2, c.x1 : c.x2][c.comp_mask > 0] = 255
    return out


def fast_luma_channel(bgr_roi: np.ndarray) -> np.ndarray:
    """Fast C++ SIMD HSV-V approximation: per-pixel max(B, G, R) as uint8."""
    if bgr_roi.ndim == 2:
        return bgr_roi
    return cv2.max(cv2.max(bgr_roi[:, :, 0], bgr_roi[:, :, 1]), bgr_roi[:, :, 2])


def build_temporal_stable_mask(
    frames_bgr: list[np.ndarray],
    masks: list[np.ndarray],
    config: ExtractConfig,
    *,
    frame_h: int | None = None,
) -> np.ndarray:
    """Compute a temporal-consensus stroke mask suppressing transient moving background props."""
    if not masks:
        return np.zeros((0, 0), dtype=np.uint8)
    if len(masks) == 1:
        return (masks[0] > 0).astype(np.uint8) * 255

    roi_h, roi_w = masks[0].shape[:2]
    k_count = len(masks)
    dil_px = effective_dilate_px(frame_h)
    k_size = 2 * dil_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    freq_cnt = np.zeros((roi_h, roi_w), dtype=np.uint8)
    for m in masks:
        dil = cv2.dilate((m > 0).astype(np.uint8), kernel)
        freq_cnt += (dil > 0).astype(np.uint8)
    freq = freq_cnt.astype(np.float32) * (1.0 / float(k_count))

    # Robust luminance stability (single uint8 sort gives median + IQR, ~5x faster than 3 percentiles).
    if k_count >= 4 and len(frames_bgr) == k_count:
        v_stack = np.stack([fast_luma_channel(bgr) for bgr in frames_bgr], axis=0)
        v_sorted = np.sort(v_stack, axis=0)
        med_v = v_sorted[k_count // 2]
        q25 = v_sorted[k_count // 4]
        q75 = v_sorted[(3 * k_count) // 4]
        iqr_v = q75.astype(np.int16) - q25.astype(np.int16)
        seed_core = (freq >= 0.55) & (med_v >= max(120, config.white_v_min - 25)) & (iqr_v <= 50)
    else:
        seed_core = freq >= 0.50

    if not np.any(seed_core):
        # Fallback: pick the cleanest non-empty mask with highest pixel count.
        best_mask = max(masks, key=lambda m: int(np.count_nonzero(m)))
        return (best_mask > 0).astype(np.uint8) * 255

    # Hysteresis reconstruction: recover partially glare-occluded glyphs aligned with seed_core.
    union_cand = (freq >= 0.35).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(union_cand, connectivity=8)

    seed_ys, seed_xs = np.where(seed_core)
    seed_y1, seed_y2 = int(seed_ys.min()), int(seed_ys.max())
    seed_x1, seed_x2 = int(seed_xs.min()), int(seed_xs.max())
    band_h = max(8, seed_y2 - seed_y1 + 1)

    stable = np.zeros((roi_h, roi_w), dtype=np.uint8)
    for label_idx in range(1, num_labels):
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        comp = labels[y : y + h, x : x + w] == label_idx
        seed_overlap = int(np.count_nonzero(comp & seed_core[y : y + h, x : x + w]))

        keep_comp = (seed_overlap / float(max(1, area))) >= 0.35
        if not keep_comp:
            # Recover narrow/glared neighbor glyph on the same horizontal text band
            overlap_y = max(0, min(seed_y2 + 1, y + h) - max(seed_y1, y))
            dx = max(0, max(x - (seed_x2 + 1), seed_x1 - (x + w)))
            if overlap_y >= 0.45 * h and dx <= max(1.8 * band_h, 0.08 * roi_w):
                keep_comp = True

        if keep_comp:
            stable[y : y + h, x : x + w][comp] = 255

    if not np.any(stable):
        stable[seed_core] = 255

    # Refine back to un-dilated union of original stroke pixels inside the stable support
    # so stroke widths stay crisp rather than bloated by dilation.
    orig_union = np.zeros((roi_h, roi_w), dtype=bool)
    for m in masks:
        orig_union |= m > 0
    refined = (stable > 0) & orig_union
    if np.any(refined):
        return refined.astype(np.uint8) * 255
    return stable


def score_candidate_frame(
    bgr_roi: np.ndarray,
    mask: np.ndarray,
    stable_mask: np.ndarray,
    *,
    frame_h: int | None = None,
    _precomputed: tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> float:
    """Score a candidate frame: reward complete crisp strokes & clean background, penalize clutter."""
    if bgr_roi.ndim < 2 or bgr_roi.size == 0 or stable_mask.size == 0:
        return -1e9

    if _precomputed is None:
        stable_bin = (stable_mask > 0).astype(np.uint8)
        stable_px = int(np.count_nonzero(stable_bin))
        if stable_px == 0:
            return frame_sharpness(bgr_roi)
        dil_px = effective_dilate_px(frame_h)
        k_tol = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil_px + 1, 2 * dil_px + 1))
        r_out = _outline_radius(bgr_roi.shape[0], frame_h=frame_h)
        k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_out + 1, 2 * r_out + 1))
        dil_out = cv2.dilate(stable_bin, k_out)
        ring_mask = (dil_out > 0) & (stable_bin == 0)
        edge_band = (cv2.dilate(stable_bin, k_tol) > 0) ^ (cv2.erode(stable_bin, k_tol) > 0)
        k_leak = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (dil_px + 1) + 1, 2 * (dil_px + 1) + 1))
        near_stable = cv2.dilate(stable_bin, k_leak) > 0
        r_bg = r_out + 3
        k_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_bg + 1, 2 * r_bg + 1))
        bg_region = cv2.dilate(stable_bin, k_bg) == 0
    else:
        stable_px, stable_bin, k_tol, ring_mask, edge_band, near_stable, bg_region = _precomputed
        if stable_px == 0:
            return frame_sharpness(bgr_roi)

    mask_bin = (mask > 0).astype(np.uint8)

    # 1. Stroke completeness: fraction of stable strokes visible in this frame
    covered = int(np.count_nonzero((stable_bin > 0) & (cv2.dilate(mask_bin, k_tol) > 0)))
    completeness = float(covered) / float(stable_px)

    # 2. Stroke-to-outline contrast on luminance channel
    v_chan = fast_luma_channel(bgr_roi)
    core_v = float(v_chan[stable_bin > 0].mean())
    ring_v = float(v_chan[ring_mask].mean()) if np.any(ring_mask) else 0.0
    contrast = max(0.0, min(220.0, core_v - ring_v)) / 220.0

    # 3. Foreground edge sharpness (measured ONLY on stroke boundary, not background!)
    gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY) if bgr_roi.ndim == 3 else bgr_roi
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    if np.any(edge_band):
        fg_sharp = float(np.log1p(max(0.0, float(lap[edge_band].var()))) / 10.0)
    else:
        fg_sharp = 0.0

    # 4. Background leak penalty: extra mask pixels outside dilated stable text support
    leak_px = int(np.count_nonzero((mask_bin > 0) & (~near_stable)))
    bg_leak = min(2.0, float(leak_px) / float(stable_px))

    # 5. Background clutter penalty: prefer flat/clean backgrounds outside the subtitle ring
    if np.any(bg_region):
        bg_clutter = float(np.log1p(max(0.0, float(lap[bg_region].var()))) / 10.0)
    else:
        bg_clutter = 0.0

    return (
        3.0 * completeness
        + 2.0 * contrast
        + 0.5 * fg_sharp
        - 1.5 * bg_leak
        - 0.8 * bg_clutter
    )


def precompute_score_regions(
    stable_mask: np.ndarray,
    roi_h: int,
    *,
    frame_h: int | None = None,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute static morphological regions once per segment for fast multi-candidate scoring."""
    stable_bin = (stable_mask > 0).astype(np.uint8)
    stable_px = int(np.count_nonzero(stable_bin))
    dil_px = effective_dilate_px(frame_h)
    k_tol = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil_px + 1, 2 * dil_px + 1))
    r_out = _outline_radius(roi_h, frame_h=frame_h)
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_out + 1, 2 * r_out + 1))
    dil_out = cv2.dilate(stable_bin, k_out)
    ring_mask = (dil_out > 0) & (stable_bin == 0)
    edge_band = (cv2.dilate(stable_bin, k_tol) > 0) ^ (cv2.erode(stable_bin, k_tol) > 0)
    k_leak = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (dil_px + 1) + 1, 2 * (dil_px + 1) + 1))
    near_stable = cv2.dilate(stable_bin, k_leak) > 0
    r_bg = r_out + 3
    k_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_bg + 1, 2 * r_bg + 1))
    bg_region = cv2.dilate(stable_bin, k_bg) == 0
    return (stable_px, stable_bin, k_tol, ring_mask, edge_band, near_stable, bg_region)


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
    """Return Laplacian variance as a sharpness score for a ROI image."""
    if bgr_roi.ndim < 2 or bgr_roi.shape[0] == 0 or bgr_roi.shape[1] == 0 or bgr_roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2GRAY) if bgr_roi.ndim == 3 else bgr_roi
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
