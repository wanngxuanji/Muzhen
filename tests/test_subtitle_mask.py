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


def test_unbordered_white_strokes_on_midtone_and_bright_bg_are_suppressed():
    cfg = ExtractConfig()
    h, w = 96, 480

    for bg_bgr in ((140, 140, 140), (35, 145, 245)):
        canvas = np.full((h, w, 3), bg_bgr, dtype=np.uint8)
        # Draw white strokes directly WITHOUT dark outline so white_bin > 0 and ring_px > 0,
        # forcing rejection via outline_ratio < _MIN_OUTLINE_RATIO.
        cv2.putText(canvas, "NO BORDER", (40, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (245, 245, 245), 2, cv2.LINE_AA)
        mask = build_stroke_mask(canvas, cfg)
        assert mask_fg_ratio(mask) < 0.002


def test_tightly_cropped_tall_outlined_glyphs_are_preserved():
    cfg = ExtractConfig()
    roi_h, roi_w = 56, 200
    canvas = np.full((roi_h, roi_w, 3), (35, 145, 245), dtype=np.uint8)

    # Tall narrow vertical stroke ('I'/'1') exceeding 0.85 * roi_h (50 > 47.6) with dark outline
    glyph = np.zeros((roi_h, roi_w), dtype=np.uint8)
    glyph[3:53, 30:38] = 255
    # Dense square CJK-like character ('田') exceeding 0.85 * roi_h (50 > 47.6, w=44 > 0.5*roi_h)
    glyph[3:53, 80:124] = 255
    glyph[10:25, 87:99] = 0
    glyph[10:25, 105:117] = 0
    glyph[31:46, 87:99] = 0
    glyph[31:46, 105:117] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    border = cv2.dilate(glyph, kernel)
    canvas[border > 0] = (10, 10, 10)
    canvas[glyph > 0] = (250, 250, 250)

    mask = build_stroke_mask(canvas, cfg)
    # Both tall narrow glyph and tall dense glyph must be preserved
    assert np.count_nonzero(mask[3:53, 30:38]) == 50 * 8
    assert np.count_nonzero(mask[3:53, 80:124]) == int(np.count_nonzero(glyph[3:53, 80:124]))


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


def test_scale_invariance_480p_1080p_4k_and_wide_roi_short_cue():
    from transapp.core.subtitle_mask import (
        effective_dilate_px,
        effective_min_blob_px,
        fg_pixel_thresholds,
    )

    cfg = ExtractConfig(min_blob_px=12)
    assert effective_min_blob_px(cfg, frame_h=480) == 6
    assert effective_min_blob_px(cfg, frame_h=1080) == 12
    assert effective_min_blob_px(cfg, frame_h=2160) == 48

    assert effective_dilate_px(frame_h=480) == 1
    assert effective_dilate_px(frame_h=1080) == 1
    assert effective_dilate_px(frame_h=2160) == 2

    # Wide 1080p ROI (1536x216): ratio-only would require ~1327px and drop 1-2 char cues;
    # frame-scaled threshold keeps it around ~466px regardless of ROI width.
    fg_min_wide, fg_edge_wide = fg_pixel_thresholds(216, 1536, cfg, frame_h=1080)
    assert 300 <= fg_min_wide <= 600
    assert fg_edge_wide < fg_min_wide


def test_top_border_claw_prop_suppressed_while_tight_roi_glyphs_preserved():
    cfg = ExtractConfig(suppress_border_blobs=True)
    h, w = 96, 640

    # Synthetic #285 claw: white paw hanging from y=0 on blue sky with 3 dark claw tips ONLY at bottom
    claw_roi = np.full((h, w, 3), (225, 175, 55), dtype=np.uint8)  # bright sky BGR
    claw_roi[0:44, 280:330] = (242, 242, 242)
    claw_roi[36:56, 284:296] = (18, 22, 28)
    claw_roi[38:58, 300:312] = (18, 22, 28)
    claw_roi[36:56, 316:328] = (18, 22, 28)
    claw_mask = build_stroke_mask(claw_roi, cfg)
    assert mask_fg_ratio(claw_mask) == 0.0

    # Tightly cropped real glyph touching y=0 with dark outline on BOTH left & right flanks must survive
    tight_roi = np.full((48, 240, 3), (40, 110, 190), dtype=np.uint8)
    glyph = np.zeros((48, 240), dtype=np.uint8)
    glyph[0:38, 60:68] = 255
    glyph[0:38, 100:108] = 255
    glyph[0:38, 140:148] = 255
    border = cv2.dilate(glyph, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    tight_roi[border > 0] = (10, 10, 10)
    tight_roi[glyph > 0] = (250, 250, 250)
    tight_mask = build_stroke_mask(tight_roi, cfg)
    assert np.count_nonzero(tight_mask) >= 3 * 30 * 8


def test_temporal_stable_mask_suppresses_moving_prop_and_scores_clean_bg_higher():
    from transapp.core.subtitle_mask import (
        build_temporal_stable_mask,
        score_candidate_frame,
    )

    cfg = ExtractConfig()
    h, w = 96, 640
    clean_bg = np.full((h, w, 3), (35, 40, 45), dtype=np.uint8)

    # Busy noisy checkerboard background (high full-ROI Laplacian variance!)
    busy_bg = clean_bg.copy()
    yy, xx = np.indices((h, w))
    busy_bg[((yy // 3 + xx // 3) % 2) == 0] = (180, 190, 200)

    clean_frame = _draw_outlined_text(clean_bg, "STABLE LINE")
    busy_frame = _draw_outlined_text(busy_bg, "STABLE LINE")

    # Add a transient outlined prop on frame 0 only (should be removed by temporal consensus)
    prop_frame = clean_frame.copy()
    prop_frame[20:54, 520:560] = (10, 10, 10)
    prop_frame[24:50, 524:556] = (248, 248, 248)

    frames = [prop_frame, clean_frame, clean_frame, clean_frame, busy_frame]
    masks = [build_stroke_mask(f, cfg) for f in frames]
    stable = build_temporal_stable_mask(frames, masks, cfg)

    # Prop on the right side (x >= 520) only appeared in 1/5 frames -> suppressed
    assert int(np.count_nonzero(stable[:, 515:])) == 0
    # Main subtitle text survived
    assert int(np.count_nonzero(stable[:, :460])) > 100

    # New scorer must prefer clean background over busy checkerboard (opposite of raw Laplacian!)
    assert frame_sharpness(busy_frame) > frame_sharpness(clean_frame)
    score_clean = score_candidate_frame(clean_frame, masks[1], stable)
    score_busy = score_candidate_frame(busy_frame, masks[4], stable)
    assert score_clean > score_busy
