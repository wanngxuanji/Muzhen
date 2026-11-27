"""Unit tests for the 2-Stage Deep Learning extractor with Majority Voting."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from transapp.core.deep_extractor import (
    _FrameVote,
    _choose_majority_vote,
    detect_subtitle_spans,
    extract_subtitles_deep,
)
from transapp.core.models import ExtractConfig
from transapp.core.ocr_engine import OcrEngine
from transapp.core.video_reader import FrameSlice


def _solid_roi(tag: int, h: int = 76, w: int = 480) -> np.ndarray:
    return np.full((h, w, 3), (tag % 250, (tag * 3) % 250, 40), dtype=np.uint8)


def test_majority_voting_overrides_edge_latin_artifacts_and_recovers_fuller_text():
    cfg = ExtractConfig(conf_threshold=0.85)
    dummy_engine = OcrEngine(ocr_callable=lambda _img: (None, None))
    img = _solid_roi(10)

    # 1 noisy frame "好N" / "可是M" cannot beat 3 clean frames "太好了"
    votes_clean = [
        _FrameVote(100, "好N", 0.73, img),
        _FrameVote(260, "太好了", 0.99, img),
        _FrameVote(420, "太好了", 0.98, img),
        _FrameVote(580, "太好了", 0.99, img),
    ]
    txt, conf, _ = _choose_majority_vote(votes_clean, dummy_engine, cfg)
    assert txt == "太好了"
    assert conf == pytest.approx(0.99)

    # Clean fuller containment "一个蕴藏着神秘力量的" beats glare-clipped "个蕴藏着神秘力量的"
    votes_yi = [
        _FrameVote(100, "个蕴藏着神秘力量的", 0.98, img),
        _FrameVote(260, "个蕴藏着神秘力量的", 0.98, img),
        _FrameVote(420, "一个蕴藏着神秘力量的", 0.99, img),
        _FrameVote(580, "一个蕴藏着神秘力量的", 0.98, img),
    ]
    txt_yi, conf_yi, _ = _choose_majority_vote(votes_yi, dummy_engine, cfg)
    assert txt_yi == "一个蕴藏着神秘力量的"
    assert conf_yi == pytest.approx(0.99)


def test_majority_voting_brightness_retry_cleans_persistent_edge_latin_noise():
    cfg = ExtractConfig(conf_threshold=0.85)
    img = _solid_roi(20)

    class RetryEngine(OcrEngine):
        def __init__(self):
            super().__init__(ocr_callable=lambda _img: (None, None))
            self.retry_called = False

        def recognize_roi(self, bgr_roi: np.ndarray, *, brightness_filter: bool = False):
            if brightness_filter:
                self.retry_called = True
                return ("叶枫", 0.96)
            return ("Y叶枫", 0.78)

    eng = RetryEngine()
    votes = [_FrameVote(100, "Y叶枫", 0.78, img), _FrameVote(260, "Y叶枫", 0.77, img)]
    txt, conf, _ = _choose_majority_vote(votes, eng, cfg)
    assert eng.retry_called is True
    assert txt == "叶枫"
    assert conf == pytest.approx(0.96)


def test_extract_subtitles_deep_bridges_flash_and_splits_back_to_back_cues():
    cfg = ExtractConfig(coarse_step=1, min_duration_ms=120, flash_bridge_max_ms=320)

    # Frame timeline (25fps, 40ms/frame):
    # 0..9:   "那场战斗结束一年后" (with 2 noisy frames)
    # 10..14: 200ms blank flash gap (no boxes detected)
    # 15..22: "那场战斗结束一年后" (same cue resuming after flash -> must merge into 1 cue)
    # 23..32: "新战斗陀螺" (back-to-back cue with same box area -> split internally by Stage 2 text!)
    class ScriptedDet:
        def __call__(self, canvas: np.ndarray):
            # Each cell encodes its frame index in channel 0
            h, w = canvas.shape[:2]
            stride = 76 + 16
            boxes = []
            for idx in range(8):
                y0 = 16 + idx * stride
                if y0 + 76 > h:
                    break
                fidx = int(canvas[y0 + 38, w // 2, 0])
                if 10 <= fidx <= 14:
                    continue
                boxes.append([[40.0, y0 + 18.0], [360.0, y0 + 18.0], [360.0, y0 + 58.0], [40.0, y0 + 58.0]])

            class _DetOut:
                def __init__(self, b):
                    self.boxes = np.array(b, dtype=np.float32) if b else None

            return _DetOut(boxes)

    class ScriptedOcr(OcrEngine):
        def __init__(self):
            super().__init__(det_callable=ScriptedDet())

        def recognize_roi(self, bgr_roi: np.ndarray, *, brightness_filter: bool = False):
            fidx = int(bgr_roi[38, 240, 0])
            if fidx in (2, 16):
                return ("那场战斗结束一年后2", 0.76)
            if fidx < 23:
                return ("那场战斗结束一年后", 0.99)
            return ("新战斗陀螺", 0.99)

    frames = [FrameSlice(i, i * 40, _solid_roi(i, h=76, w=480)) for i in range(33)]
    segs = extract_subtitles_deep(frames, fps=25.0, start_pts_ms=0, ocr_engine=ScriptedOcr(), config=cfg)

    assert len(segs) == 2
    assert segs[0].text == "那场战斗结束一年后"
    assert segs[0].confidence == pytest.approx(0.99)
    assert segs[0].start_ms == 0
    assert segs[0].end_ms >= 880
    assert segs[1].text == "新战斗陀螺"
    assert segs[1].confidence == pytest.approx(0.99)


_FIXED_BOX = (0.20, 0.24, 0.80, 0.76)


def _make_patterned_roi(
    frame_idx: int,
    pattern_id: int | None,
    *,
    h: int = 76,
    w: int = 480,
    moving_bg: bool = True,
) -> np.ndarray:
    """Create a synthetic BGR ROI with moving background and distinct outlined glyph strokes."""
    if moving_bg:
        bg_b = 45 + (frame_idx * 11) % 130
        bg_g = 60 + (frame_idx * 7) % 120
        bg_r = 160 - (frame_idx * 9) % 110
        img = np.full((h, w, 3), (bg_b, bg_g, bg_r), dtype=np.uint8)
        bar_x = (frame_idx * 17) % (w - 80)
        cv2.rectangle(img, (bar_x, 0), (bar_x + 80, h), (35, 145, 225), thickness=-1)
    else:
        img = np.full((h, w, 3), (30, 30, 30), dtype=np.uint8)

    if pattern_id is not None:
        # Draw 6 distinct high-contrast outlined glyph patterns inside the exact same box
        x0 = int(round(_FIXED_BOX[0] * w)) + 8
        y0 = int(round(_FIXED_BOX[1] * h)) + 4
        cell_w = 44
        cell_h = 30
        fg = (250, 250, 250) if pattern_id != 2 else (60, 240, 250)
        outline = (10, 10, 10)
        for idx in range(6):
            cx = x0 + idx * cell_w
            cy = y0
            variant = (pattern_id * 5 + idx) % 6
            for color, thick in ((outline, 5), (fg, 2)):
                if variant == 0:
                    cv2.line(img, (cx + 4, cy + 4), (cx + 34, cy + 4), color, thick)
                    cv2.line(img, (cx + 19, cy + 4), (cx + 19, cy + 28), color, thick)
                    cv2.line(img, (cx + 4, cy + 28), (cx + 34, cy + 28), color, thick)
                elif variant == 1:
                    cv2.rectangle(img, (cx + 5, cy + 4), (cx + 33, cy + 26), color, thick)
                    cv2.line(img, (cx + 5, cy + 15), (cx + 33, cy + 15), color, thick)
                elif variant == 2:
                    cv2.line(img, (cx + 6, cy + 4), (cx + 32, cy + 27), color, thick)
                    cv2.line(img, (cx + 32, cy + 4), (cx + 6, cy + 27), color, thick)
                elif variant == 3:
                    cv2.circle(img, (cx + 19, cy + 15), 11, color, thick)
                    cv2.line(img, (cx + 19, cy + 2), (cx + 19, cy + 28), color, thick)
                elif variant == 4:
                    cv2.line(img, (cx + 6, cy + 4), (cx + 6, cy + 28), color, thick)
                    cv2.line(img, (cx + 6, cy + 28), (cx + 34, cy + 28), color, thick)
                    cv2.line(img, (cx + 18, cy + 4), (cx + 34, cy + 16), color, thick)
                else:
                    cv2.line(img, (cx + 4, cy + 10), (cx + 34, cy + 10), color, thick)
                    cv2.line(img, (cx + 12, cy + 3), (cx + 12, cy + 28), color, thick)
                    cv2.line(img, (cx + 26, cy + 3), (cx + 26, cy + 28), color, thick)

    # Encode frame_idx in top-left corner pixel (far outside _FIXED_BOX) for mock Det lookup
    img[0, 0] = (frame_idx & 0xFF, (frame_idx >> 8) & 0xFF, 0)
    return img


class _BoxLookupEngine(OcrEngine):
    """Test double returning exact boxes per frame_idx without grid padding."""

    def __init__(self, frame_boxes: dict[int, list[tuple[float, float, float, float]]]):
        super().__init__(provider="cpu", ocr_callable=lambda _img: (None, None))
        self._frame_boxes = frame_boxes
        self.batch_calls: list[int] = []

    def detect_grid_batch(self, rois: list[np.ndarray]) -> list[list[tuple[float, float, float, float]]]:
        self.batch_calls.append(len(rois))
        out: list[list[tuple[float, float, float, float]]] = []
        for roi in rois:
            fidx = int(roi[0, 0, 0]) | (int(roi[0, 0, 1]) << 8)
            out.append(list(self._frame_boxes.get(fidx, [])))
        return out


def test_ring_buffer_refines_onset_offset_to_exact_frame_zero_drift():
    # 25fps (40ms/frame), 40 frames: cues on exact inclusive frames [5..18] and [22..35]
    frame_boxes: dict[int, list[tuple[float, float, float, float]]] = {}
    frames: list[FrameSlice] = []
    for i in range(40):
        if 5 <= i <= 18:
            pat = 0
            frame_boxes[i] = [_FIXED_BOX]
        elif 22 <= i <= 35:
            pat = 1
            frame_boxes[i] = [_FIXED_BOX]
        else:
            pat = None
        frames.append(FrameSlice(i, i * 40, _make_patterned_roi(i, pat)))

    for step in (2, 3):
        eng = _BoxLookupEngine(frame_boxes)
        cfg = ExtractConfig(coarse_step=step, min_duration_ms=120)
        spans, frame_pts_ms = detect_subtitle_spans(frames, fps=25.0, start_pts_ms=0, ocr_engine=eng, config=cfg)

        assert frame_pts_ms == [i * 40 for i in range(40)]
        assert len(spans) == 2
        assert spans[0].start_idx == 5
        assert spans[0].end_idx == 18
        assert spans[0].start_ms == 200
        assert spans[0].end_ms == 760
        assert spans[1].start_idx == 22
        assert spans[1].end_idx == 35
        assert spans[1].start_ms == 880
        assert spans[1].end_ms == 1440


def test_zero_ms_back_to_back_equal_width_cues_split_at_exact_cut_frame():
    # 3 consecutive equal-width cues [4..15], [16..27], [28..40] with 0ms gap & moving background
    frame_boxes: dict[int, list[tuple[float, float, float, float]]] = {}
    frames: list[FrameSlice] = []
    for i in range(45):
        if 4 <= i <= 15:
            pat = 0
            frame_boxes[i] = [_FIXED_BOX]
        elif 16 <= i <= 27:
            pat = 1
            frame_boxes[i] = [_FIXED_BOX]
        elif 28 <= i <= 40:
            pat = 2
            frame_boxes[i] = [_FIXED_BOX]
        else:
            pat = None
        frames.append(FrameSlice(i, i * 40, _make_patterned_roi(i, pat, moving_bg=True)))

    eng = _BoxLookupEngine(frame_boxes)
    cfg = ExtractConfig(coarse_step=2, min_duration_ms=120)
    spans, frame_pts_ms = detect_subtitle_spans(frames, fps=25.0, start_pts_ms=0, ocr_engine=eng, config=cfg)

    assert len(frame_pts_ms) == 45
    assert len(spans) == 3
    assert (spans[0].start_idx, spans[0].end_idx) == (4, 15)
    assert (spans[1].start_idx, spans[1].end_idx) == (16, 27)
    assert (spans[2].start_idx, spans[2].end_idx) == (28, 40)
    assert spans[0].start_ms == 160
    assert spans[0].end_ms == spans[1].start_ms == 640
    assert spans[1].end_ms == spans[2].start_ms == 1120
    assert spans[2].end_ms == 1640

    # Also test odd-frame cut boundary ([4..14] -> [15..26])
    frame_boxes_odd: dict[int, list[tuple[float, float, float, float]]] = {}
    frames_odd: list[FrameSlice] = []
    for i in range(32):
        if 4 <= i <= 14:
            pat = 0
            frame_boxes_odd[i] = [_FIXED_BOX]
        elif 15 <= i <= 26:
            pat = 1
            frame_boxes_odd[i] = [_FIXED_BOX]
        else:
            pat = None
        frames_odd.append(FrameSlice(i, i * 40, _make_patterned_roi(i, pat, moving_bg=True)))

    spans_odd, _ = detect_subtitle_spans(
        frames_odd, fps=25.0, start_pts_ms=0, ocr_engine=_BoxLookupEngine(frame_boxes_odd), config=cfg
    )
    assert len(spans_odd) == 2
    assert (spans_odd[0].start_idx, spans_odd[0].end_idx) == (4, 14)
    assert (spans_odd[1].start_idx, spans_odd[1].end_idx) == (15, 26)
    assert spans_odd[0].end_ms == spans_odd[1].start_ms == 600


def test_short_160ms_cue_all_phases_caught_and_40ms_flash_dropped():
    cfg = ExtractConfig(coarse_step=2, min_duration_ms=120)

    for phase in (0, 1):
        cue_start = 6 + phase
        cue_end = cue_start + 3  # 4 inclusive frames = 160ms at 25fps
        flash_idx = 24 + phase   # 1 isolated frame = 40ms
        frame_boxes: dict[int, list[tuple[float, float, float, float]]] = {}
        frames: list[FrameSlice] = []
        for i in range(34):
            if cue_start <= i <= cue_end:
                pat = 0
                frame_boxes[i] = [_FIXED_BOX]
            elif i == flash_idx:
                pat = 1
                frame_boxes[i] = [_FIXED_BOX]
            else:
                pat = None
            frames.append(FrameSlice(i, i * 40, _make_patterned_roi(i, pat)))

        spans, _ = detect_subtitle_spans(
            frames, fps=25.0, start_pts_ms=0, ocr_engine=_BoxLookupEngine(frame_boxes), config=cfg
        )
        assert len(spans) == 1
        assert spans[0].start_idx == cue_start
        assert spans[0].end_idx == cue_end
        assert spans[0].end_ms - spans[0].start_ms == 160


def test_vfr_timeline_uses_actual_next_frame_pts_no_overlap():
    # Mixed 12fps (~83ms), 25fps (40ms), and 60fps (~17ms) VFR PTS timeline
    pts: list[int] = [0]
    for i in range(1, 40):
        if i <= 14:
            step = 83  # ~12fps
        elif i <= 26:
            step = 40  # 25fps
        else:
            step = 17  # ~60fps
        pts.append(pts[-1] + step)

    frame_boxes: dict[int, list[tuple[float, float, float, float]]] = {}
    frames: list[FrameSlice] = []
    for i, t in enumerate(pts):
        if 3 <= i <= 11:
            pat = 0
            frame_boxes[i] = [_FIXED_BOX]
        elif 12 <= i <= 21:
            pat = 1
            frame_boxes[i] = [_FIXED_BOX]
        elif 24 <= i <= 35:
            pat = 2
            frame_boxes[i] = [_FIXED_BOX]
        else:
            pat = None
        frames.append(FrameSlice(i, t, _make_patterned_roi(i, pat)))

    cfg = ExtractConfig(coarse_step=2, min_duration_ms=120)
    spans, frame_pts_ms = detect_subtitle_spans(
        frames, fps=25.0, start_pts_ms=0, ocr_engine=_BoxLookupEngine(frame_boxes), config=cfg
    )

    assert frame_pts_ms == pts
    assert len(spans) == 3
    assert spans[0].start_ms == pts[3]
    assert spans[0].end_ms == pts[12]
    assert spans[1].start_ms == pts[12]
    assert spans[1].end_ms == pts[22]
    assert spans[2].start_ms == pts[24]
    assert spans[2].end_ms == pts[36]
    assert spans[0].end_ms <= spans[1].start_ms
    assert spans[1].end_ms <= spans[2].start_ms


def test_span_sample_reservoir_bounded_at_24_with_full_tail_coverage():
    frame_boxes = {i: [_FIXED_BOX] for i in range(2, 102)}
    frames = [
        FrameSlice(i, i * 40, _make_patterned_roi(i, 0 if 2 <= i <= 101 else None, moving_bg=True))
        for i in range(106)
    ]
    cfg = ExtractConfig(coarse_step=2, min_duration_ms=120)
    spans, _ = detect_subtitle_spans(
        frames, fps=25.0, start_pts_ms=0, ocr_engine=_BoxLookupEngine(frame_boxes), config=cfg
    )

    assert len(spans) == 1
    sp = spans[0]
    assert (sp.start_idx, sp.end_idx) == (2, 101)
    assert len(sp.samples) == 24
    assert sp.samples[0].frame_idx == 2
    assert sp.samples[-1].frame_idx == 101
    # Verify uniform interior coverage (no huge tail gap like old stride *= 2)
    sample_idxs = [s.frame_idx for s in sp.samples]
    assert sample_idxs == sorted(sample_idxs)
    max_gap = max(b - a for a, b in zip(sample_idxs, sample_idxs[1:]))
    assert max_gap <= 10
