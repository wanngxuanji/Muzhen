from pathlib import Path

import cv2
import numpy as np
import pytest

from transapp.core.models import ExtractConfig
from transapp.core.subtitle_mask import (
    build_stroke_mask,
    frame_sharpness,
    mask_fg_ratio,
    mask_iou,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    assert img is not None, f"Failed to decode fixture: {path}"
    return img


def _draw_outlined_text(canvas: np.ndarray, text: str = "SUBTITLE 123") -> np.ndarray:
    out = canvas.copy()
    h, w = out.shape[:2]
    org = (max(16, w // 10), int(h * 0.68))
    glyph = np.zeros((h, w), dtype=np.uint8)
    cv2.putText(glyph, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 255, 2, cv2.LINE_AA)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    border = cv2.dilate(glyph, kernel)
    out[border > 0] = (10, 10, 10)
    out[glyph > 127] = (250, 250, 250)
    return out


@pytest.mark.parametrize("fixture_name", ["sample_dark_bg.png", "sample_bright_bg.png"])
def test_real_fixtures_produce_non_empty_binary_stroke_mask(fixture_name: str):
    roi = _load_bgr(FIXTURES_DIR / fixture_name)
    cfg = ExtractConfig()

    mask = build_stroke_mask(roi, cfg)

    assert mask.dtype == np.uint8
    assert mask.shape == roi.shape[:2]
    assert set(np.unique(mask).tolist()).issubset({0, 255})
    ratio = mask_fg_ratio(mask)
    assert 0.02 < ratio < 0.45


def test_unbordered_blank_bright_orange_and_white_sky_are_suppressed():
    cfg = ExtractConfig()
    h, w = 96, 640

    blank_dark = np.full((h, w, 3), 20, dtype=np.uint8)
    bright_orange = np.full((h, w, 3), (35, 145, 245), dtype=np.uint8)
    white_sky = np.full((h, w, 3), (242, 240, 238), dtype=np.uint8)

    # Large white shirt occupying full ROI height with dark background beside it, but unbordered
    white_shirt_on_dark = np.full((h, w, 3), 18, dtype=np.uint8)
    white_shirt_on_dark[:, 180:460] = (235, 235, 235)

    for patch in (blank_dark, bright_orange, white_sky, white_shirt_on_dark):
        mask = build_stroke_mask(patch, cfg)
        assert mask.dtype == np.uint8
        assert mask.shape == (h, w)
        assert mask_fg_ratio(mask) < 0.002


def test_shifting_bright_orange_circle_behind_outlined_text_preserves_mask_iou():
    cfg = ExtractConfig()
    h, w = 100, 640

    bg_before = np.full((h, w, 3), (30, 35, 40), dtype=np.uint8)
    cv2.circle(bg_before, (160, 50), 75, (25, 140, 250), -1)
    frame_before = _draw_outlined_text(bg_before, "HELLO TEST")

    bg_after = np.full((h, w, 3), (30, 35, 40), dtype=np.uint8)
    cv2.circle(bg_after, (460, 50), 75, (25, 140, 250), -1)
    frame_after = _draw_outlined_text(bg_after, "HELLO TEST")

    mask_before = build_stroke_mask(frame_before, cfg)
    mask_after = build_stroke_mask(frame_after, cfg)

    assert mask_fg_ratio(mask_before) > 0.01
    assert mask_iou(mask_before, mask_after) >= 0.85


def test_tiny_specks_below_min_blob_px_are_filtered():
    cfg = ExtractConfig(min_blob_px=12)
    roi = np.zeros((80, 200, 3), dtype=np.uint8)
    # 2x2 white speck on black background (area = 4 < 12)
    roi[30:32, 50:52] = (255, 255, 255)
    # 3x3 white speck (area = 9 < 12)
    roi[40:43, 100:103] = (255, 255, 255)

    mask = build_stroke_mask(roi, cfg)
    assert mask_fg_ratio(mask) == 0.0


def test_empty_and_zero_size_inputs_and_helper_functions():
    cfg = ExtractConfig()

    for shape in ((0, 0, 3), (0, 100, 3), (50, 0, 3)):
        empty_bgr = np.zeros(shape, dtype=np.uint8)
        mask = build_stroke_mask(empty_bgr, cfg)
        assert mask.dtype == np.uint8
        assert mask.shape == shape[:2]
        assert mask_fg_ratio(mask) == 0.0
        assert frame_sharpness(empty_bgr) == 0.0

    a = np.zeros((20, 40), dtype=np.uint8)
    b = np.zeros((20, 40), dtype=np.uint8)
    assert mask_iou(a, b) == 0.0

    a[4:10, 4:14] = 255
    b[4:10, 4:14] = 255
    assert mask_iou(a, b, dilate_px=0) == pytest.approx(1.0)

    c = np.zeros((20, 40), dtype=np.uint8)
    c[4:10, 20:30] = 255
    assert mask_iou(a, c, dilate_px=0) == 0.0

    # Shifted by 1 px: dilate_px=1 makes IoU higher than dilate_px=0
    d = np.zeros((20, 40), dtype=np.uint8)
    d[4:10, 5:15] = 255
    assert mask_iou(a, d, dilate_px=1) > mask_iou(a, d, dilate_px=0)

    sharp_img = _draw_outlined_text(np.zeros((100, 500, 3), dtype=np.uint8), "SHARP")
    blurred_img = cv2.GaussianBlur(sharp_img, (9, 9), 2.5)
    assert frame_sharpness(sharp_img) > frame_sharpness(blurred_img) > 0.0
