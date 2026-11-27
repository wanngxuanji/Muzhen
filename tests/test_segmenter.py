from __future__ import annotations

import cv2
import numpy as np
import pytest

from transapp.core.models import ExtractConfig, ExtractionCancelled
from transapp.core.segmenter import segment_roi_frames
from transapp.core.subtitle_mask import build_stroke_mask, frame_sharpness, mask_fg_ratio
from transapp.core.video_reader import FrameSlice


def _make_blank_roi(h: int = 96, w: int = 640, bg_bgr: tuple[int, int, int] = (24, 28, 32)) -> np.ndarray:
    return np.full((h, w, 3), bg_bgr, dtype=np.uint8)


def _draw_outlined_cue(
    canvas: np.ndarray,
    text: str,
    *,
    org_x: int = 60,
    font_scale: float = 1.25,
    fg_value: int = 250,
    blur_ksize: int = 0,
) -> np.ndarray:
    out = canvas.copy()
    h, w = out.shape[:2]
    org = (org_x, int(h * 0.68))
    glyph = np.zeros((h, w), dtype=np.uint8)
    cv2.putText(glyph, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, 2, cv2.LINE_AA)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    border = cv2.dilate(glyph, kernel)

    overlay = out.copy()
    overlay[border > 0] = (10, 10, 10)
    overlay[glyph > 127] = (fg_value, fg_value, fg_value)
    if blur_ksize >= 3:
        overlay = cv2.GaussianBlur(overlay, (blur_ksize, blur_ksize), 0)
    return overlay


def _make_faded_cue(
    base_bg: np.ndarray,
    text: str,
    alpha: float,
    *,
    blur_ksize: int = 5,
) -> np.ndarray:
    solid = _draw_outlined_cue(base_bg, text, blur_ksize=blur_ksize)
    blended = cv2.addWeighted(solid, alpha, base_bg, 1.0 - alpha, 0.0)
    return blended


def test_segmenter_merges_moving_orange_bg_and_splits_back_to_back_cues():
    cfg = ExtractConfig()
    h, w = 96, 640
    frames: list[FrameSlice] = []
    progress_calls: list[tuple[int, int]] = []

    for idx in range(100):
        pts_ms = idx * 40
        bg = _make_blank_roi(h, w)
        if 0 <= pts_ms < 400:
            roi = bg
        elif 400 <= pts_ms < 1600:
            if pts_ms >= 920:
                rect_x = 40 + ((idx - 23) * 28) % 440
                cv2.rectangle(bg, (rect_x, 8), (rect_x + 140, h - 8), (25, 140, 250), -1)
            roi = _draw_outlined_cue(bg, "CUE A TEXT", org_x=80)
        elif 1600 <= pts_ms < 2000:
            roi = bg
        elif 2000 <= pts_ms < 3000:
            roi = _draw_outlined_cue(bg, "SECOND B", org_x=180)
        else:
            roi = _draw_outlined_cue(bg, "DIFFERENT THIRD", org_x=40)
        frames.append(FrameSlice(frame_index=idx, pts_ms=pts_ms, bgr_roi=roi))

    segments = segment_roi_frames(
        frames,
        cfg,
        on_frame_progress=lambda fi, pts: progress_calls.append((fi, pts)),
    )

    assert len(segments) == 3
    seg_a, seg_b, seg_c = segments

    assert seg_a.start_ms == 400
    assert seg_a.end_ms == 1600
    assert 10 <= seg_a.frame_index <= 39

    assert seg_b.start_ms == 2000
    assert seg_b.end_ms == 3000
    assert 50 <= seg_b.frame_index <= 74

    assert seg_c.start_ms == 3000
    assert seg_c.end_ms == 4000
    assert 75 <= seg_c.frame_index <= 99

    assert seg_a.end_ms <= seg_b.start_ms
    assert seg_b.end_ms <= seg_c.start_ms

    for seg in segments:
        assert seg.peak_bgr.shape == (h, w, 3)
        assert seg.peak_bgr.flags["C_CONTIGUOUS"]
        assert frame_sharpness(seg.peak_bgr) > 0.0

    assert progress_calls[0] == (0, 0)
    assert progress_calls[-1] == (99, 3960)


def test_segmenter_fade_in_hysteresis_and_sharp_peak_selection():
    cfg = ExtractConfig()
    h, w = 96, 640
    bg = _make_blank_roi(h, w)

    # Calibrate soft fade-in frames so fg_ratio lies inside [fg_ratio_edge, fg_ratio_min)
    soft_1 = _make_faded_cue(bg, "FADE IN CUE D", alpha=0.80, blur_ksize=5)
    soft_2 = _make_faded_cue(bg, "FADE IN CUE D", alpha=0.82, blur_ksize=5)
    # Force exact edge-band density if needed by checking mask ratios
    soft_1_ratio = mask_fg_ratio(build_stroke_mask(soft_1, cfg))
    soft_2_ratio = mask_fg_ratio(build_stroke_mask(soft_2, cfg))
    if not (cfg.fg_ratio_edge <= soft_1_ratio < cfg.fg_ratio_min):
        # Construct deterministic edge fade frames from solid mask strokes
        solid_ref = _draw_outlined_cue(bg, "FADE IN CUE D", org_x=90)
        solid_mask = build_stroke_mask(solid_ref, cfg)
        ys, xs = np.where(solid_mask > 0)
        keep_1 = len(ys) // 6
        keep_2 = len(ys) // 5
        soft_1 = bg.copy()
        soft_2 = bg.copy()
        # Keep dark border around a partial subset of white strokes to produce edge-level density
        for frame_out, count in ((soft_1, keep_1), (soft_2, keep_2)):
            sub_mask = np.zeros((h, w), dtype=np.uint8)
            sub_mask[ys[:count], xs[:count]] = 255
            border = cv2.dilate(sub_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
            frame_out[border > 0] = (12, 12, 12)
            frame_out[sub_mask > 0] = (205, 205, 205)
            frame_out[:] = cv2.GaussianBlur(frame_out, (3, 3), 0)

    solid_sharp = _draw_outlined_cue(bg, "FADE IN CUE D", org_x=90)
    solid_soft = cv2.GaussianBlur(solid_sharp, (5, 5), 1.5)

    r1 = mask_fg_ratio(build_stroke_mask(soft_1, cfg))
    r2 = mask_fg_ratio(build_stroke_mask(soft_2, cfg))
    assert cfg.fg_ratio_edge <= r1 < cfg.fg_ratio_min
    assert cfg.fg_ratio_edge <= r2 < cfg.fg_ratio_min

    frames: list[FrameSlice] = []
    for idx in range(20):
        pts_ms = idx * 40
        if idx < 4:
            roi = bg.copy()
        elif idx == 4:
            roi = soft_1.copy()
        elif idx == 5:
            roi = soft_2.copy()
        elif idx in (6, 7, 15, 16):
            roi = solid_soft.copy()
        elif 8 <= idx <= 14:
            roi = solid_sharp.copy()
        else:
            roi = bg.copy()
        frames.append(FrameSlice(frame_index=idx, pts_ms=pts_ms, bgr_roi=roi))

    segments = segment_roi_frames(frames, cfg)

    assert len(segments) == 1
    seg = segments[0]
    assert 160 <= seg.start_ms <= 200
    assert seg.start_ms == 160
    assert seg.end_ms == 680
    assert 8 <= seg.frame_index <= 14
    assert frame_sharpness(seg.peak_bgr) > frame_sharpness(soft_2)
    assert frame_sharpness(seg.peak_bgr) > frame_sharpness(solid_soft)


def test_segmenter_cancellation_stops_consuming_immediately():
    cfg = ExtractConfig()
    h, w = 64, 320
    consumed = 0

    def _frame_gen():
        nonlocal consumed
        for idx in range(50):
            consumed += 1
            yield FrameSlice(frame_index=idx, pts_ms=idx * 40, bgr_roi=_make_blank_roi(h, w))

    with pytest.raises(ExtractionCancelled):
        segment_roi_frames(_frame_gen(), cfg, cancel_check=lambda: consumed >= 5)

    assert consumed == 5


def test_segmenter_empty_blank_single_frame_and_coarse_step_3_boundaries():
    cfg = ExtractConfig()
    assert segment_roi_frames([], cfg) == []

    blank_only = [FrameSlice(i, i * 40, _make_blank_roi(64, 320)) for i in range(10)]
    assert segment_roi_frames(blank_only, cfg) == []

    cue_single = _draw_outlined_cue(_make_blank_roi(96, 480), "SINGLE")
    single_res = segment_roi_frames([FrameSlice(0, 250, cue_single)], cfg)
    assert len(single_res) == 1
    assert single_res[0].start_ms == 250
    assert single_res[0].end_ms == 290
    assert single_res[0].frame_index == 0

    # coarse_step=3: Cue 1 (0..3), 1-frame blank gap (4), Cue 2 starts on non-coarse tail (5..7) -> EOF
    cfg_step3 = ExtractConfig(coarse_step=3)
    bg = _make_blank_roi(96, 640)
    cue_1 = _draw_outlined_cue(bg, "ALPHA", org_x=40)
    cue_2 = _draw_outlined_cue(bg, "OMEGA", org_x=280)
    stream = [
        FrameSlice(0, 0, cue_1),
        FrameSlice(1, 33, cue_1),
        FrameSlice(2, 66, cue_1),
        FrameSlice(3, 100, cue_1),
        FrameSlice(4, 133, bg),
        FrameSlice(5, 166, cue_2),
        FrameSlice(6, 200, cue_2),
        FrameSlice(7, 233, cue_2),
    ]
    segs = segment_roi_frames(stream, cfg_step3)
    assert len(segs) == 2
    assert (segs[0].start_ms, segs[0].end_ms) == (0, 133)
    assert (segs[1].start_ms, segs[1].end_ms) == (166, 266)
    assert segs[0].end_ms <= segs[1].start_ms


def _make_partial_stroke_frame(
    bg: np.ndarray, solid_mask: np.ndarray, keep_fraction: float
) -> np.ndarray:
    h, w = bg.shape[:2]
    ys, xs = np.where(solid_mask > 0)
    count = max(1, int(round(len(ys) * keep_fraction)))
    sub_mask = np.zeros((h, w), dtype=np.uint8)
    sub_mask[ys[:count], xs[:count]] = 255
    border = cv2.dilate(sub_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    out = bg.copy()
    out[border > 0] = (12, 12, 12)
    out[sub_mask > 0] = (235, 235, 235)
    return out


def test_segmenter_gradual_fade_in_and_fade_out_do_not_split():
    cfg = ExtractConfig(coarse_step=2)
    h, w = 96, 640
    bg = _make_blank_roi(h, w)
    solid = _draw_outlined_cue(bg, "SMOOTH FADE CUE", org_x=60)
    solid_mask = build_stroke_mask(solid, cfg)

    # Create a mid-fade frame with fg_ratio >= fg_ratio_min (~35% of solid strokes, IoU < 0.55)
    mid_fade = _make_partial_stroke_frame(bg, solid_mask, 0.32)
    edge_fade = _make_partial_stroke_frame(bg, solid_mask, 0.05)
    mid_ratio = mask_fg_ratio(build_stroke_mask(mid_fade, cfg))
    edge_ratio = mask_fg_ratio(build_stroke_mask(edge_fade, cfg))
    assert mid_ratio >= cfg.fg_ratio_min
    assert cfg.fg_ratio_edge <= edge_ratio < cfg.fg_ratio_min

    stream = [
        FrameSlice(0, 0, bg),
        FrameSlice(1, 40, bg),
        FrameSlice(2, 80, edge_fade),   # coarse: edge-band fade-in
        FrameSlice(3, 120, mid_fade),   # mid fade-in
        FrameSlice(4, 160, mid_fade),   # coarse: >= fg_ratio_min, still ramping
        FrameSlice(5, 200, solid),
        FrameSlice(6, 240, solid),      # coarse: full solid
        FrameSlice(7, 280, solid),
        FrameSlice(8, 320, solid),      # coarse: full solid
        FrameSlice(9, 360, solid),
        FrameSlice(10, 400, edge_fade), # coarse: soft edge-band fade-out
        FrameSlice(11, 440, edge_fade), # edge-band fade-out
        FrameSlice(12, 480, bg),        # coarse: blank
        FrameSlice(13, 520, bg),
    ]
    segs = segment_roi_frames(stream, cfg)
    assert len(segs) == 1
    assert segs[0].start_ms == 80
    assert segs[0].end_ms == 480


def test_segmenter_cancel_before_first_frame_and_coarse_step_1_eof():
    cfg = ExtractConfig(coarse_step=1)
    with pytest.raises(ExtractionCancelled):
        segment_roi_frames([], cfg, cancel_check=lambda: True)

    bg = _make_blank_roi(96, 480)
    cue = _draw_outlined_cue(bg, "STEP ONE")
    stream = [
        FrameSlice(0, 0, bg),
        FrameSlice(1, 50, cue),
        FrameSlice(2, 100, cue),
        FrameSlice(3, 150, bg),
    ]
    segs = segment_roi_frames(stream, cfg)
    assert len(segs) == 1
    assert (segs[0].start_ms, segs[0].end_ms) == (50, 150)


def test_segmenter_long_cue_decimation_picks_interior_peak():
    cfg = ExtractConfig(coarse_step=2)
    bg = _make_blank_roi(96, 480)
    blurred = _draw_outlined_cue(bg, "LONG CUE", blur_ksize=5)
    sharp = _draw_outlined_cue(bg, "LONG CUE")

    # 150 active frames (> _MAX_CANDIDATE_FRAMES=64) with sharpest frames around the middle
    stream = [
        FrameSlice(i, i * 40, sharp if 60 <= i <= 90 else blurred)
        for i in range(150)
    ]
    segs = segment_roi_frames(stream, cfg)
    assert len(segs) == 1
    assert (segs[0].start_ms, segs[0].end_ms) == (0, 6000)
    assert 45 <= segs[0].frame_index <= 105
    assert frame_sharpness(segs[0].peak_bgr) > frame_sharpness(blurred)


def _draw_two_line_cue(
    canvas: np.ndarray, top_text: str, bottom_text: str
) -> np.ndarray:
    h, w = canvas.shape[:2]
    glyph = np.zeros((h, w), dtype=np.uint8)
    cv2.putText(glyph, top_text, (60, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2, cv2.LINE_AA)
    cv2.putText(glyph, bottom_text, (60, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2, cv2.LINE_AA)
    border = cv2.dilate(glyph, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    out = canvas.copy()
    out[border > 0] = (10, 10, 10)
    out[glyph > 127] = (250, 250, 250)
    return out


def _draw_bottom_line_cue(canvas: np.ndarray, bottom_text: str) -> np.ndarray:
    h, w = canvas.shape[:2]
    glyph = np.zeros((h, w), dtype=np.uint8)
    cv2.putText(glyph, bottom_text, (60, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 255, 2, cv2.LINE_AA)
    border = cv2.dilate(glyph, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    out = canvas.copy()
    out[border > 0] = (10, 10, 10)
    out[glyph > 127] = (250, 250, 250)
    return out


def test_segmenter_back_to_back_two_line_to_one_line_splits_without_delay():
    cfg = ExtractConfig(coarse_step=2)
    bg = _make_blank_roi(96, 640)
    two_line = _draw_two_line_cue(bg, "LONGER TOP LINE ONE", "BOTTOM")
    one_line = _draw_bottom_line_cue(bg, "BOTTOM")

    mask_2 = build_stroke_mask(two_line, cfg)
    mask_1 = build_stroke_mask(one_line, cfg)
    fg_2 = mask_fg_ratio(mask_2)
    fg_1 = mask_fg_ratio(mask_1)
    # Confirm this hits the exact hazardous regime: fg_1 >= fg_ratio_min, fg_1 < 0.6 * fg_2, IoU < split_iou_min
    assert fg_1 >= cfg.fg_ratio_min
    assert fg_1 < fg_2 * 0.6

    # Case A: even-frame transition (0..9 two-line, 10..12 short 3-frame one-line, 13..15 blank)
    stream_even = [
        FrameSlice(i, i * 40, two_line if i < 10 else (one_line if i < 13 else bg))
        for i in range(16)
    ]
    segs_even = segment_roi_frames(stream_even, cfg)
    assert len(segs_even) == 2
    assert (segs_even[0].start_ms, segs_even[0].end_ms) == (0, 400)
    assert (segs_even[1].start_ms, segs_even[1].end_ms) == (400, 520)

    # Case B: odd-frame transition (0..8 two-line, 9..16 one-line, 17..19 blank)
    stream_odd = [
        FrameSlice(i, i * 40, two_line if i < 9 else (one_line if i < 17 else bg))
        for i in range(20)
    ]
    segs_odd = segment_roi_frames(stream_odd, cfg)
    assert len(segs_odd) == 2
    assert (segs_odd[0].start_ms, segs_odd[0].end_ms) == (0, 360)
    assert (segs_odd[1].start_ms, segs_odd[1].end_ms) == (360, 680)


def test_segmenter_bridges_240ms_flash_gap_but_keeps_quiet_repeated_dialogue_separate():
    cfg = ExtractConfig(coarse_step=2, merge_gap_ms=160, flash_bridge_max_ms=320)
    h, w = 96, 640
    dark_bg = _make_blank_roi(h, w, (24, 28, 32))
    flash_bg = _make_blank_roi(h, w, (245, 245, 245))
    cue = _draw_outlined_cue(dark_bg, "FLASH BRIDGE", org_x=80)

    # Case 1: 240ms bright flash whiteout (frames 10..15) inside a single cue -> bridged into 1 segment
    flash_stream = [
        FrameSlice(i, i * 40, flash_bg if 10 <= i < 16 else cue)
        for i in range(26)
    ]
    flash_segs = segment_roi_frames(flash_stream, cfg)
    assert len(flash_segs) == 1
    assert flash_segs[0].start_ms == 0
    assert flash_segs[0].end_ms == 1040
    assert flash_segs[0].stable_mask is not None
    assert int(np.count_nonzero(flash_segs[0].stable_mask)) > 0

    # Case 2: 240ms clean dark quiet gap between genuine repeated dialogue -> stays 2 separate segments
    repeat_stream = [
        FrameSlice(i, i * 40, dark_bg if 10 <= i < 16 else cue)
        for i in range(26)
    ]
    repeat_segs = segment_roi_frames(repeat_stream, cfg)
    assert len(repeat_segs) == 2
    assert (repeat_segs[0].start_ms, repeat_segs[0].end_ms) == (0, 400)
    assert (repeat_segs[1].start_ms, repeat_segs[1].end_ms) == (640, 1040)


def test_segmenter_ema_anchor_resists_transient_bright_prop_poisoning():
    cfg = ExtractConfig(coarse_step=2)
    h, w = 96, 640
    dark_bg = _make_blank_roi(h, w, (24, 28, 32))
    clean_cue = _draw_outlined_cue(dark_bg, "STEADY CUE", org_x=60)

    # Transient huge white outlined prop passing on the right side for 2 frames (would poison anchor_fg!)
    poisoned_cue = clean_cue.copy()
    poisoned_cue[10:86, 400:610] = (8, 8, 8)
    poisoned_cue[16:80, 406:604] = (248, 248, 248)

    stream = [
        FrameSlice(i, i * 40, poisoned_cue if i in (8, 9) else clean_cue)
        for i in range(24)
    ]
    segs = segment_roi_frames(stream, cfg)
    assert len(segs) == 1
    assert segs[0].start_ms == 0
    assert segs[0].end_ms == 960
    # Peak selection must pick a clean frame without the right-side bright prop
    assert float(segs[0].peak_bgr[20:76, 410:600].mean()) < 60.0


def test_segmenter_partial_glare_continuation_preserves_single_cue_and_clean_peak():
    cfg = ExtractConfig(coarse_step=2)
    h, w = 96, 640
    dark_bg = _make_blank_roi(h, w, (24, 28, 32))
    full_cue = _draw_outlined_cue(dark_bg, "FULL LINE TEXT", org_x=40)

    # Localized bright glare washes out left half of the cue for several frames
    glared_cue = full_cue.copy()
    glared_cue[:, 0:260] = (248, 248, 248)

    stream = [
        FrameSlice(i, i * 40, glared_cue if 8 <= i < 14 else full_cue)
        for i in range(24)
    ]
    segs = segment_roi_frames(stream, cfg)
    assert len(segs) == 1
    assert segs[0].start_ms == 0
    assert segs[0].end_ms == 960
    # Chosen peak must be an un-glared full frame
    assert float(segs[0].peak_bgr[:, 0:60].mean()) < 80.0
