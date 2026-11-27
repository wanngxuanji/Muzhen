"""Unit tests for the 2-Stage Deep Learning extractor with Majority Voting."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from transapp.core.deep_extractor import (
    TightSignature,
    _DetectedSpan,
    _FrameVote,
    _choose_majority_vote,
    _compute_tight_signature,
    _find_sandwich_noise,
    _is_strict_same_cue,
    _tight_signatures_same_cue,
    detect_subtitle_spans,
    extract_subtitles_deep,
    recognize_spans_with_voting,
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

    votes_clean = [
        _FrameVote(100, "好N", 0.73, img),
        _FrameVote(260, "太好了", 0.99, img),
        _FrameVote(420, "太好了", 0.98, img),
        _FrameVote(580, "太好了", 0.99, img),
    ]
    txt, conf, _ = _choose_majority_vote(votes_clean, dummy_engine, cfg)
    assert txt == "太好了"
    assert conf == pytest.approx(0.99)

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

    class ScriptedDet:
        def __call__(self, canvas: np.ndarray):
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
            if fidx == 2:
                return ("场战斗结束一年后#", 0.76)
            if fidx < 10:
                return ("场战斗结束一年后", 0.98)
            if fidx == 16:
                return ("那场战斗结束一年后#", 0.76)
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
        cue_end = cue_start + 3
        flash_idx = 24 + phase
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
    pts: list[int] = [0]
    for i in range(1, 40):
        if i <= 14:
            step = 83
        elif i <= 26:
            step = 40
        else:
            step = 17
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
    sample_idxs = [s.frame_idx for s in sp.samples]
    assert sample_idxs == sorted(sample_idxs)
    max_gap = max(b - a for a, b in zip(sample_idxs, sample_idxs[1:]))
    assert max_gap <= 10


def _make_encoded_span(
    frame_indices: list[int],
    pts_ms_list: list[int],
    end_ms: int,
    *,
    luma_overrides: dict[int, int] | None = None,
) -> _DetectedSpan:
    """Create a _DetectedSpan whose samples encode their frame_idx in a lossless-safe corner block."""
    span = _DetectedSpan(
        start_idx=frame_indices[0],
        end_idx=frame_indices[-1],
        start_ms=pts_ms_list[0],
        end_ms=end_ms,
        union_rect=_FIXED_BOX,
    )
    for fidx, pts in zip(frame_indices, pts_ms_list):
        base = (luma_overrides or {}).get(fidx, 100)
        img = np.full((76, 480, 3), (base, base, base), dtype=np.uint8)
        img[0:16, 0:16, 0] = min(248, fidx * 8)
        span.add_sample(fidx, pts, img)
    return span


def _decode_encoded_fidx(bgr: np.ndarray) -> int:
    return int(round(float(bgr[4:12, 4:12, 0].mean()) / 8.0))


def test_stage2_early_exit_saves_ocr_calls_on_clean_consensus():
    cfg = ExtractConfig(conf_threshold=0.85, min_duration_ms=120)
    frame_pts_ms = [i * 180 for i in range(11)]
    span = _make_encoded_span(list(range(10)), frame_pts_ms[:10], end_ms=1800)

    called_frames: list[int] = []

    class CleanEngine(OcrEngine):
        def __init__(self):
            super().__init__(provider="cpu", ocr_callable=lambda _img: (None, None))

        def recognize_roi(self, bgr_roi: np.ndarray, *, brightness_filter: bool = False):
            fidx = _decode_encoded_fidx(bgr_roi)
            called_frames.append(fidx)
            if fidx in (0, 9):
                return ("G太好了", 0.72)
            return ("太好了", 0.98)

    eng = CleanEngine()
    segs = recognize_spans_with_voting([span], frame_pts_ms, eng, cfg)

    assert len(segs) == 1
    assert segs[0].text == "太好了"
    assert segs[0].start_ms == 0
    assert segs[0].end_ms == 1800
    assert len(called_frames) == 3
    assert 0 not in called_frames
    assert 9 not in called_frames


def test_stage2_binary_search_splits_one_char_minimal_pair_to_exact_frame():
    cfg = ExtractConfig(conf_threshold=0.85, min_duration_ms=120, flash_bridge_max_ms=320)
    frame_pts_ms = [i * 40 for i in range(25)]
    span = _make_encoded_span(list(range(24)), frame_pts_ms[:24], end_ms=frame_pts_ms[24])

    called_frames: list[int] = []

    class MinimalPairEngine(OcrEngine):
        def __init__(self):
            super().__init__(provider="cpu", ocr_callable=lambda _img: (None, None))

        def recognize_roi(self, bgr_roi: np.ndarray, *, brightness_filter: bool = False):
            fidx = _decode_encoded_fidx(bgr_roi)
            called_frames.append(fidx)
            if fidx <= 11:
                return ("我不会放过你的", 0.98)
            return ("我不会放过他的", 0.98)

    eng = MinimalPairEngine()
    segs = recognize_spans_with_voting([span], frame_pts_ms, eng, cfg)

    assert len(segs) == 2
    assert segs[0].text == "我不会放过你的"
    assert segs[1].text == "我不会放过他的"
    assert segs[0].start_ms == 0
    assert abs(segs[0].end_ms - frame_pts_ms[12]) <= 40
    assert segs[1].start_ms == segs[0].end_ms
    assert segs[1].end_ms == frame_pts_ms[24]
    assert len(called_frames) < 16


def test_stage2_weighted_consensus_pools_varied_edge_noise_and_symbol_artifacts():
    cfg = ExtractConfig(conf_threshold=0.85)
    dummy_engine = OcrEngine(provider="cpu", ocr_callable=lambda _img: (None, None))
    img1 = _solid_roi(11)
    img2 = _solid_roi(22)

    votes_varied = [
        _FrameVote(100, "G我们是救世", 0.78, img1, frame_idx=1),
        _FrameVote(200, "Y我们是救世", 0.76, img1, frame_idx=2),
        _FrameVote(300, "我们是救世", 0.99, img2, frame_idx=3),
    ]
    txt, conf, _ = _choose_majority_vote(votes_varied, dummy_engine, cfg)
    assert txt == "我们是救世"
    assert conf == pytest.approx(0.99)

    votes_2v1 = [
        _FrameVote(100, "G我们是救世", 0.78, img1, frame_idx=1),
        _FrameVote(200, "G我们是救世", 0.78, img1, frame_idx=2),
        _FrameVote(300, "我们是救世", 0.95, img2, frame_idx=3),
    ]
    txt_2v1, conf_2v1, _ = _choose_majority_vote(votes_2v1, dummy_engine, cfg)
    assert txt_2v1 == "我们是救世"
    assert conf_2v1 == pytest.approx(0.95)

    class SymbolRetryEngine(OcrEngine):
        def __init__(self):
            super().__init__(provider="cpu", ocr_callable=lambda _img: (None, None))
            self.retry_calls = 0

        def recognize_roi(self, bgr_roi: np.ndarray, *, brightness_filter: bool = False):
            if brightness_filter:
                self.retry_calls += 1
                if self.retry_calls == 1 and bgr_roi is img1:
                    return ("#一人", 0.62)
                return ("一人", 0.95)
            return ("#一人", 0.62)

    eng_single = SymbolRetryEngine()
    votes_sym = [_FrameVote(100, "#一人", 0.62, img2, frame_idx=1)]
    txt_sym, conf_sym, _ = _choose_majority_vote(votes_sym, eng_single, cfg)
    assert eng_single.retry_calls == 1
    assert txt_sym == "一人"
    assert conf_sym == pytest.approx(0.95)

    eng_two = SymbolRetryEngine()
    votes_two = [
        _FrameVote(100, "#一人", 0.62, img1, frame_idx=1),
        _FrameVote(200, "#一人", 0.60, img2, frame_idx=2),
    ]
    txt_two, conf_two, _ = _choose_majority_vote(votes_two, eng_two, cfg)
    assert eng_two.retry_calls == 2
    assert txt_two == "一人"
    assert conf_two == pytest.approx(0.95)


def test_stage2_middle_noise_vote_does_not_shatter_single_cue():
    cfg = ExtractConfig(conf_threshold=0.85, min_duration_ms=120)
    frame_pts_ms = [i * 200 for i in range(6)]
    span = _make_encoded_span(list(range(5)), frame_pts_ms[:5], end_ms=frame_pts_ms[5])

    class MiddleNoiseEngine(OcrEngine):
        def __init__(self):
            super().__init__(provider="cpu", ocr_callable=lambda _img: (None, None))

        def recognize_roi(self, bgr_roi: np.ndarray, *, brightness_filter: bool = False):
            fidx = _decode_encoded_fidx(bgr_roi)
            if fidx == 2:
                return ("乱码杂音", 0.65)
            return ("战斗开始", 0.98)

    segs = recognize_spans_with_voting([span], frame_pts_ms, MiddleNoiseEngine(), cfg)

    assert len(segs) == 1
    assert segs[0].text == "战斗开始"
    assert segs[0].start_ms == 0
    assert segs[0].end_ms == 1000


def test_stage2_char_level_consensus_and_clean_fuller_containment_single_support():
    cfg = ExtractConfig(conf_threshold=0.85)
    dummy_engine = OcrEngine(provider="cpu", ocr_callable=lambda _img: (None, None))
    img = _solid_roi(33)

    votes_char = [
        _FrameVote(100, "你们一起去公园", 0.85, img, frame_idx=1),
        _FrameVote(200, "我们一起去公院", 0.86, img, frame_idx=2),
        _FrameVote(300, "我们一七去公园", 0.84, img, frame_idx=3),
    ]
    txt_char, _, _ = _choose_majority_vote(votes_char, dummy_engine, cfg)
    assert txt_char == "我们一起去公园"

    votes_fuller = [
        _FrameVote(100, "个蕴藏着神秘力量的", 0.98, img, frame_idx=1),
        _FrameVote(200, "个蕴藏着神秘力量的", 0.98, img, frame_idx=2),
        _FrameVote(300, "个蕴藏着神秘力量的", 0.98, img, frame_idx=3),
        _FrameVote(400, "一个蕴藏着神秘力量的", 0.94, img, frame_idx=4),
    ]
    txt_fuller, conf_fuller, _ = _choose_majority_vote(votes_fuller, dummy_engine, cfg)
    assert txt_fuller == "一个蕴藏着神秘力量的"
    assert conf_fuller == pytest.approx(0.94)


def test_tight_signature_empty_boxes_and_opposite_metric_floor():
    img = _solid_roi(40)
    empty_sig = _compute_tight_signature(img, [])
    assert empty_sig.boxes == []
    assert empty_sig.union_rect == (0.0, 0.0, 0.0, 0.0)
    assert not np.any(empty_sig.edge_mask)
    assert not np.any(empty_sig.luma_patch)

    edge = np.zeros((16, 64), dtype=np.uint8)
    edge[4:12, 10:50] = 1
    la = np.zeros((16, 64), dtype=np.float32)
    lb = np.zeros((16, 64), dtype=np.float32)
    la[4:12, 10:30] = 1.0
    la /= float(np.linalg.norm(la))
    lb[4:12, 10:30] = -1.0
    lb /= float(np.linalg.norm(lb))
    sig_a = TightSignature(boxes=[_FIXED_BOX], union_rect=_FIXED_BOX, edge_mask=edge, luma_patch=la)
    sig_b = TightSignature(boxes=[_FIXED_BOX], union_rect=_FIXED_BOX, edge_mask=edge, luma_patch=lb)
    assert _tight_signatures_same_cue(sig_a, sig_b) is False


def test_stage2_glare_dodge_empty_early_exit_and_short_ghost_guard():
    cfg = ExtractConfig(conf_threshold=0.85, min_duration_ms=120)

    frame_pts_ms = [i * 150 for i in range(11)]
    span_glare = _make_encoded_span(
        list(range(10)), frame_pts_ms[:10], end_ms=1500, luma_overrides={4: 250}
    )
    glare_calls: list[int] = []

    class GlareEngine(OcrEngine):
        def __init__(self):
            super().__init__(provider="cpu", ocr_callable=lambda _img: (None, None))

        def recognize_roi(self, bgr_roi: np.ndarray, *, brightness_filter: bool = False):
            fidx = _decode_encoded_fidx(bgr_roi)
            glare_calls.append(fidx)
            if fidx == 4:
                return ("G闪光错字", 0.60)
            return ("清晰字幕", 0.98)

    segs_glare = recognize_spans_with_voting([span_glare], frame_pts_ms, GlareEngine(), cfg)
    assert len(segs_glare) == 1
    assert segs_glare[0].text == "清晰字幕"
    assert len(glare_calls) == 3
    assert 4 not in glare_calls

    empty_pts = [i * 100 for i in range(9)]
    span_empty = _make_encoded_span(list(range(8)), empty_pts[:8], end_ms=800)
    empty_calls: list[int] = []

    class EmptyEngine(OcrEngine):
        def __init__(self):
            super().__init__(provider="cpu", ocr_callable=lambda _img: (None, None))

        def recognize_roi(self, bgr_roi: np.ndarray, *, brightness_filter: bool = False):
            empty_calls.append(_decode_encoded_fidx(bgr_roi))
            return ("", 0.0)

    segs_empty = recognize_spans_with_voting([span_empty], empty_pts, EmptyEngine(), cfg)
    assert segs_empty == []
    assert len(empty_calls) == 3

    ghost_pts = [0, 160, 320]
    span_ghost = _make_encoded_span([0, 1], [0, 160], end_ms=320)
    span_real = _make_encoded_span([0, 1], [0, 160], end_ms=320)

    class FixedEngine(OcrEngine):
        def __init__(self, text: str, conf: float):
            super().__init__(provider="cpu", ocr_callable=lambda _img: (None, None))
            self._text = text
            self._conf = conf

        def recognize_roi(self, bgr_roi: np.ndarray, *, brightness_filter: bool = False):
            return (self._text, self._conf)

    assert recognize_spans_with_voting([span_ghost], ghost_pts, FixedEngine("生", 0.55), cfg) == []
    kept = recognize_spans_with_voting([span_real], ghost_pts, FixedEngine("嗯", 0.95), cfg)
    assert len(kept) == 1 and kept[0].text == "嗯"


def test_stage2_streams_pending_cluster_inline_and_honors_cancel_on_flush():
    from transapp.core.models import ExtractionCancelled

    cfg = ExtractConfig(conf_threshold=0.85, min_duration_ms=120, flash_bridge_max_ms=200)
    frame_pts_ms = [i * 100 for i in range(40)]
    span1 = _make_encoded_span([0, 1, 2, 3, 4], [0, 100, 200, 300, 400], end_ms=500)
    span2 = _make_encoded_span([10, 11, 12, 13, 14], [1000, 1100, 1200, 1300, 1400], end_ms=1500)
    span3 = _make_encoded_span([20, 21, 22, 23, 24], [2000, 2100, 2200, 2300, 2400], end_ms=2500)

    timeline: list[str] = []

    class StreamingTrackEngine(OcrEngine):
        def __init__(self):
            super().__init__(provider="cpu", ocr_callable=lambda _img: (None, None))

        def recognize_roi(self, bgr_roi: np.ndarray, *, brightness_filter: bool = False):
            fidx = _decode_encoded_fidx(bgr_roi)
            if fidx < 10:
                timeline.append("ocr_span1")
                return ("第一句字幕", 0.98)
            if fidx < 20:
                timeline.append("ocr_span2")
                return ("第二句字幕", 0.98)
            timeline.append("ocr_span3")
            return ("第三句字幕", 0.98)

        def encode_thumbnail_png(self, bgr_roi: np.ndarray) -> bytes:
            timeline.append("flush_thumb")
            return super().encode_thumbnail_png(bgr_roi)

    segs = recognize_spans_with_voting(
        [span1, span2, span3], frame_pts_ms, StreamingTrackEngine(), cfg
    )
    assert [s.text for s in segs] == ["第一句字幕", "第二句字幕", "第三句字幕"]
    first_flush_pos = timeline.index("flush_thumb")
    first_span3_ocr_pos = timeline.index("ocr_span3")
    assert first_flush_pos < first_span3_ocr_pos

    span_cancel = _make_encoded_span([0, 1, 2, 3, 4], [0, 100, 200, 300, 400], end_ms=500)
    ocr_done = {"count": 0}

    class CancelOnFlushEngine(OcrEngine):
        def __init__(self):
            super().__init__(provider="cpu", ocr_callable=lambda _img: (None, None))

        def recognize_roi(self, bgr_roi: np.ndarray, *, brightness_filter: bool = False):
            ocr_done["count"] += 1
            return ("第一句字幕", 0.98)

    with pytest.raises(ExtractionCancelled):
        recognize_spans_with_voting(
            [span_cancel],
            frame_pts_ms,
            CancelOnFlushEngine(),
            cfg,
            cancel_check=lambda: ocr_done["count"] >= 3,
        )


def test_stage2_binary_search_handles_none_midpoint_after_true_cut():
    cfg = ExtractConfig(conf_threshold=0.85, min_duration_ms=120, flash_bridge_max_ms=320)
    frame_pts_ms = [i * 40 for i in range(25)]
    span = _make_encoded_span(list(range(24)), frame_pts_ms[:24], end_ms=frame_pts_ms[24])

    class BlankMidEngine(OcrEngine):
        def __init__(self):
            super().__init__(provider="cpu", ocr_callable=lambda _img: (None, None))

        def recognize_roi(self, bgr_roi: np.ndarray, *, brightness_filter: bool = False):
            fidx = _decode_encoded_fidx(bgr_roi)
            if 8 <= fidx <= 12:
                return ("", 0.0)
            if fidx <= 5:
                return ("我不会放过你的", 0.98)
            return ("我不会放过他的", 0.98)

    segs = recognize_spans_with_voting([span], frame_pts_ms, BlankMidEngine(), cfg)
    assert len(segs) == 2
    assert segs[0].text == "我不会放过你的"
    assert segs[1].text == "我不会放过他的"
    assert abs(segs[0].end_ms - frame_pts_ms[6]) <= 40
    assert segs[1].start_ms == segs[0].end_ms


def test_stage2_absorbed_high_conf_sandwich_noise_excluded_from_voting():
    cfg = ExtractConfig(conf_threshold=0.85, min_duration_ms=120)
    frame_pts_ms = [0, 200, 400, 600]
    span = _make_encoded_span([0, 1, 2], [0, 200, 400], end_ms=600)

    class HighConfGlitchEngine(OcrEngine):
        def __init__(self):
            super().__init__(provider="cpu", ocr_callable=lambda _img: (None, None))

        def recognize_roi(self, bgr_roi: np.ndarray, *, brightness_filter: bool = False):
            fidx = _decode_encoded_fidx(bgr_roi)
            if fidx == 0:
                return ("个蕴藏着神秘力量的", 0.88)
            if fidx == 1:
                return ("乱码错字", 0.92)
            return ("一个蕴藏着神秘力量的", 0.89)

    segs = recognize_spans_with_voting([span], frame_pts_ms, HighConfGlitchEngine(), cfg)
    assert len(segs) == 1
    assert segs[0].text == "一个蕴藏着神秘力量的"


def test_brightness_retry_cleaned_edge_rejects_severely_truncated_result():
    cfg = ExtractConfig(conf_threshold=0.85)
    img = _solid_roi(55)

    class TruncatingRetryEngine(OcrEngine):
        def __init__(self):
            super().__init__(provider="cpu", ocr_callable=lambda _img: (None, None))
            self.called = False

        def recognize_roi(self, bgr_roi: np.ndarray, *, brightness_filter: bool = False):
            if brightness_filter:
                self.called = True
                return ("世", 0.95)
            return ("G我们是救世", 0.78)

    eng = TruncatingRetryEngine()
    votes = [_FrameVote(100, "G我们是救世", 0.78, img, frame_idx=1)]
    txt, conf, _ = _choose_majority_vote(votes, eng, cfg)
    assert eng.called is True
    assert txt == "G我们是救世"
    assert conf == pytest.approx(0.78)


def test_tight_signature_invariant_to_dbnet_box_jitter_on_short_two_char_cue():
    """Stationary 2-char hardsub must match under +/-2% X / +/-6% Y DBNet box jitter,
    while a different 2-char cue at the same box still splits."""
    h, w = 92, 820
    short_box = (0.44, 0.22, 0.56, 0.80)

    def _draw_two_char_roi(variant: int, bg_shift: int) -> np.ndarray:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for x in range(0, w, 24):
            c = 185 if ((x + bg_shift) // 24) % 2 == 0 else 45
            cv2.rectangle(img, (x, 0), (min(w, x + 12), h), (c, 90, 210 - c // 2), -1)
        x0 = int(short_box[0] * w) + 6
        y0 = int(short_box[1] * h) + 6
        for idx in range(2):
            cx = x0 + idx * 44
            for thick, color in ((6, (0, 0, 0)), (2, (255, 255, 255))):
                if variant == 0:
                    if idx == 0:
                        cv2.line(img, (cx + 2, y0 + 6), (cx + 34, y0 + 6), color, thick)
                        cv2.line(img, (cx + 18, y0 + 6), (cx + 18, y0 + 38), color, thick)
                        cv2.line(img, (cx + 18, y0 + 14), (cx + 4, y0 + 32), color, thick)
                        cv2.line(img, (cx + 18, y0 + 14), (cx + 32, y0 + 32), color, thick)
                    else:
                        cv2.rectangle(img, (cx + 10, y0 + 8), (cx + 34, y0 + 30), color, thick)
                        cv2.line(img, (cx + 2, y0 + 36), (cx + 36, y0 + 36), color, thick)
                else:
                    if idx == 0:
                        cv2.circle(img, (cx + 18, y0 + 22), 14, color, thick)
                    else:
                        cv2.line(img, (cx + 4, y0 + 38), (cx + 18, y0 + 4), color, thick)
                        cv2.line(img, (cx + 18, y0 + 4), (cx + 34, y0 + 38), color, thick)
        return img

    roi_a1 = _draw_two_char_roi(variant=0, bg_shift=0)
    roi_a2 = _draw_two_char_roi(variant=0, bg_shift=13)
    roi_b = _draw_two_char_roi(variant=1, bg_shift=0)

    sig_base = _compute_tight_signature(roi_a1, [short_box])
    for dx in (-0.02, -0.015, 0.0, 0.015, 0.02):
        for dy in (-0.06, 0.0, 0.06):
            jittered = (
                short_box[0] + dx,
                max(0.0, short_box[1] + dy),
                short_box[2] + dx,
                min(1.0, short_box[3] + dy),
            )
            sig_jit = _compute_tight_signature(roi_a2, [jittered])
            assert _tight_signatures_same_cue(sig_base, sig_jit), (
                f"Expected same cue under box jitter dx={dx}, dy={dy}"
            )

    sig_other = _compute_tight_signature(roi_b, [short_box])
    assert not _tight_signatures_same_cue(sig_base, sig_other)


def test_adjacent_one_frame_spans_are_kept_while_isolated_40ms_flash_is_dropped():
    """Adjacent 1-frame (40ms) spans from jitter/cuts must survive Stage 1 for Stage 2 merging,
    while truly isolated 1-frame 40ms flashes are still dropped."""
    frame_boxes: dict[int, list[tuple[float, float, float, float]]] = {}
    frames: list[FrameSlice] = []
    for i in range(40):
        if i == 2:
            pat = 0
            frame_boxes[i] = [_FIXED_BOX]
        elif 20 <= i <= 27:
            pat = i % 3
            frame_boxes[i] = [_FIXED_BOX]
        else:
            pat = None
        frames.append(FrameSlice(i, i * 40, _make_patterned_roi(i, pat)))

    eng = _BoxLookupEngine(frame_boxes)
    cfg = ExtractConfig(coarse_step=1, min_duration_ms=120, flash_bridge_max_ms=320)
    spans, _pts = detect_subtitle_spans(frames, fps=25.0, start_pts_ms=0, ocr_engine=eng, config=cfg)

    assert all(sp.start_idx != 2 for sp in spans)
    covered = set()
    for sp in spans:
        covered.update(range(sp.start_idx, sp.end_idx + 1))
    assert covered == set(range(20, 28))


def test_compute_tight_signature_skips_unused_fullres_patch_and_uses_640x64_roi_gray(monkeypatch):
    """_compute_tight_signature must skip redundant full-res _extract_patch_features, build a
    640x64 roi_gray with appropriate interpolation, and preserve 1080p 2-char discrimination."""
    import transapp.core.deep_extractor as de

    h, w = 108, 1536
    short_box = (0.45, 0.20, 0.55, 0.82)

    def _draw_1080p_two_char_roi(variant: int, bg_shift: int) -> np.ndarray:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for x in range(0, w, 32):
            c = 180 if ((x + bg_shift) // 32) % 2 == 0 else 42
            cv2.rectangle(img, (x, 0), (min(w, x + 16), h), (c, 95, 210 - c // 2), -1)
        x0 = int(short_box[0] * w) + 12
        y0 = int(short_box[1] * h) + 8
        for idx in range(2):
            cx = x0 + idx * 64
            for thick, color in ((7, (0, 0, 0)), (3, (255, 255, 255))):
                if variant == 0:
                    if idx == 0:
                        cv2.line(img, (cx + 4, y0 + 6), (cx + 48, y0 + 6), color, thick)
                        cv2.line(img, (cx + 26, y0 + 6), (cx + 26, y0 + 48), color, thick)
                        cv2.line(img, (cx + 26, y0 + 18), (cx + 6, y0 + 42), color, thick)
                        cv2.line(img, (cx + 26, y0 + 18), (cx + 46, y0 + 42), color, thick)
                    else:
                        cv2.rectangle(img, (cx + 12, y0 + 10), (cx + 48, y0 + 38), color, thick)
                        cv2.line(img, (cx + 4, y0 + 46), (cx + 50, y0 + 46), color, thick)
                else:
                    if idx == 0:
                        cv2.circle(img, (cx + 26, y0 + 28), 18, color, thick)
                    else:
                        cv2.line(img, (cx + 6, y0 + 48), (cx + 26, y0 + 6), color, thick)
                        cv2.line(img, (cx + 26, y0 + 6), (cx + 48, y0 + 48), color, thick)
        return img

    roi_a1 = _draw_1080p_two_char_roi(variant=0, bg_shift=0)
    roi_a2 = _draw_1080p_two_char_roi(variant=0, bg_shift=19)
    roi_b = _draw_1080p_two_char_roi(variant=1, bg_shift=0)

    extract_calls = {"count": 0}
    orig_extract = de._extract_patch_features

    def _spy_extract(*args, **kwargs):
        extract_calls["count"] += 1
        return orig_extract(*args, **kwargs)

    resize_interps: list[int] = []
    orig_resize = de.cv2.resize

    def _spy_resize(src, dsize, *args, **kwargs):
        if "interpolation" in kwargs:
            resize_interps.append(int(kwargs["interpolation"]))
        elif len(args) >= 3:
            resize_interps.append(int(args[2]))
        return orig_resize(src, dsize, *args, **kwargs)

    monkeypatch.setattr(de, "_extract_patch_features", _spy_extract)
    monkeypatch.setattr(de.cv2, "resize", _spy_resize)

    sig_a = _compute_tight_signature(roi_a1, [short_box])
    assert extract_calls["count"] == 0
    assert sig_a.roi_gray is not None
    assert sig_a.roi_gray.shape == (64, 640)
    assert resize_interps[-1] == cv2.INTER_AREA

    small_roi = _solid_roi(7, h=40, w=320)
    sig_small = _compute_tight_signature(small_roi, [short_box])
    assert extract_calls["count"] == 0
    assert sig_small.roi_gray is not None
    assert sig_small.roi_gray.shape == (64, 640)
    assert resize_interps[-1] == cv2.INTER_LINEAR

    jittered_box = (
        short_box[0] + 0.015,
        short_box[1] - 0.05,
        short_box[2] + 0.015,
        short_box[3] - 0.05,
    )
    sig_a_jit = _compute_tight_signature(roi_a2, [jittered_box])
    sig_b = _compute_tight_signature(roi_b, [short_box])
    assert _tight_signatures_same_cue(sig_a, sig_a_jit) is True
    assert _tight_signatures_same_cue(sig_a, sig_b) is False


def test_onset_and_offset_refinement_retains_faded_boundary_frames_with_overlapping_box():
    """Faded onset/offset frames with overlapping subtitle box must be retained during
    EMPTY <-> TEXT refinement, while detached side false-positive boxes still stop refinement."""
    faded_box = (0.21, 0.25, 0.79, 0.77)
    detached_box = (0.02, 0.24, 0.14, 0.76)

    def _make_faded_roi(frame_idx: int) -> np.ndarray:
        img = np.full((76, 480, 3), (78, 82, 86), dtype=np.uint8)
        cv2.line(img, (110, 38), (370, 38), (94, 98, 102), 2)
        img[0, 0] = (frame_idx & 0xFF, (frame_idx >> 8) & 0xFF, 0)
        return img

    def _make_side_fp_roi(frame_idx: int) -> np.ndarray:
        img = np.full((76, 480, 3), (30, 30, 30), dtype=np.uint8)
        cv2.rectangle(img, (12, 20), (62, 54), (245, 245, 245), 2)
        img[0, 0] = (frame_idx & 0xFF, (frame_idx >> 8) & 0xFF, 0)
        return img

    crisp_ref = _make_patterned_roi(5, 0, moving_bg=False)
    faded_sample = _make_faded_roi(4)
    assert (
        _tight_signatures_same_cue(
            _compute_tight_signature(faded_sample, [faded_box]),
            _compute_tight_signature(crisp_ref, [_FIXED_BOX]),
        )
        is False
    )

    frame_boxes: dict[int, list[tuple[float, float, float, float]]] = {}
    frames: list[FrameSlice] = []
    for i in range(37):
        if i in (4, 19, 26):
            frame_boxes[i] = [faded_box]
            roi = _make_faded_roi(i)
        elif i in (20, 25):
            frame_boxes[i] = [detached_box]
            roi = _make_side_fp_roi(i)
        elif 5 <= i <= 18:
            frame_boxes[i] = [_FIXED_BOX]
            roi = _make_patterned_roi(i, 0, moving_bg=False)
        elif 27 <= i <= 33:
            frame_boxes[i] = [_FIXED_BOX]
            roi = _make_patterned_roi(i, 1, moving_bg=False)
        else:
            roi = _make_patterned_roi(i, None, moving_bg=False)
        frames.append(FrameSlice(i, i * 40, roi))

    eng = _BoxLookupEngine(frame_boxes)
    cfg = ExtractConfig(coarse_step=3, min_duration_ms=120)
    spans, _ = detect_subtitle_spans(frames, fps=25.0, start_pts_ms=0, ocr_engine=eng, config=cfg)

    assert len(spans) == 2
    assert spans[0].start_idx == 4
    assert spans[0].end_idx == 19
    assert spans[1].start_idx == 26
    assert spans[1].end_idx == 33


def test_numeric_and_grade_prefixed_cues_are_not_merged_or_stripped():
    assert _is_strict_same_cue("3天后", "5天后", 0.98, 0.98) is False
    assert _is_strict_same_cue("S级", "A级", 0.96, 0.96) is False
    assert _is_strict_same_cue("距离爆炸还有3秒", "距离爆炸还有2秒", 0.98, 0.98) is False

    cfg = ExtractConfig(conf_threshold=0.85)
    img = _solid_roi(77)

    class SpyRetryEngine(OcrEngine):
        def __init__(self):
            super().__init__(provider="cpu", ocr_callable=lambda _img: (None, None))
            self.retry_called = False

        def recognize_roi(self, bgr_roi: np.ndarray, *, brightness_filter: bool = False):
            if brightness_filter:
                self.retry_called = True
                return ("天后", 0.99)
            return ("3天后", 0.98)

    eng = SpyRetryEngine()
    votes = [
        _FrameVote(100, "3天后", 0.98, img, frame_idx=1),
        _FrameVote(200, "3天后", 0.97, img, frame_idx=2),
        _FrameVote(300, "天后", 0.86, img, frame_idx=3),
    ]
    txt, conf, _ = _choose_majority_vote(votes, eng, cfg)
    assert txt == "3天后"
    assert conf == pytest.approx(0.98)
    assert eng.retry_called is False

    eng_grade = SpyRetryEngine()
    votes_grade = [
        _FrameVote(100, "S级", 0.96, img, frame_idx=1),
        _FrameVote(200, "S级", 0.95, img, frame_idx=2),
    ]
    txt_g, conf_g, _ = _choose_majority_vote(votes_grade, eng_grade, cfg)
    assert txt_g == "S级"
    assert conf_g == pytest.approx(0.96)
    assert eng_grade.retry_called is False


def test_stage2_bounded_stack_splits_three_cues_and_transition_glitch_cleanly():
    cfg = ExtractConfig(conf_threshold=0.85, min_duration_ms=120, flash_bridge_max_ms=320)
    frame_pts_ms = [i * 40 for i in range(25)]

    span_three = _make_encoded_span(list(range(24)), frame_pts_ms[:24], end_ms=frame_pts_ms[24])

    class ThreeCueEngine(OcrEngine):
        def __init__(self):
            super().__init__(provider="cpu", ocr_callable=lambda _img: (None, None))

        def recognize_roi(self, bgr_roi: np.ndarray, *, brightness_filter: bool = False):
            fidx = _decode_encoded_fidx(bgr_roi)
            if fidx <= 5:
                return ("第一句字幕", 0.98)
            if fidx <= 10:
                return ("第二句字幕", 0.98)
            return ("第三句字幕", 0.98)

    segs_three = recognize_spans_with_voting([span_three], frame_pts_ms, ThreeCueEngine(), cfg)
    assert [s.text for s in segs_three] == ["第一句字幕", "第二句字幕", "第三句字幕"]
    assert segs_three[0].start_ms == 0
    assert abs(segs_three[0].end_ms - frame_pts_ms[6]) <= 40
    assert segs_three[1].start_ms == segs_three[0].end_ms
    assert abs(segs_three[1].end_ms - frame_pts_ms[11]) <= 40
    assert segs_three[2].start_ms == segs_three[1].end_ms
    assert segs_three[2].end_ms == frame_pts_ms[24]

    span_three_eq = _make_encoded_span(list(range(24)), frame_pts_ms[:24], end_ms=frame_pts_ms[24])

    class ThreeCueBalancedEngine(OcrEngine):
        def __init__(self):
            super().__init__(provider="cpu", ocr_callable=lambda _img: (None, None))

        def recognize_roi(self, bgr_roi: np.ndarray, *, brightness_filter: bool = False):
            fidx = _decode_encoded_fidx(bgr_roi)
            if fidx <= 7:
                return ("第一句字幕", 0.98)
            if fidx <= 15:
                return ("第二句字幕", 0.98)
            return ("第三句字幕", 0.98)

    segs_eq = recognize_spans_with_voting([span_three_eq], frame_pts_ms, ThreeCueBalancedEngine(), cfg)
    assert [s.text for s in segs_eq] == ["第一句字幕", "第二句字幕", "第三句字幕"]
    assert abs(segs_eq[0].end_ms - frame_pts_ms[8]) <= 40
    assert abs(segs_eq[1].end_ms - frame_pts_ms[16]) <= 40

    span_glitch = _make_encoded_span(list(range(24)), frame_pts_ms[:24], end_ms=frame_pts_ms[24])

    class MidGlitchEngine(OcrEngine):
        def __init__(self):
            super().__init__(provider="cpu", ocr_callable=lambda _img: (None, None))

        def recognize_roi(self, bgr_roi: np.ndarray, *, brightness_filter: bool = False):
            fidx = _decode_encoded_fidx(bgr_roi)
            if fidx <= 11:
                return ("我不会放过你的", 0.98)
            if fidx == 12:
                return ("乱码", 0.65)
            return ("我不会放过他的", 0.98)

    segs_glitch = recognize_spans_with_voting([span_glitch], frame_pts_ms, MidGlitchEngine(), cfg)
    assert [s.text for s in segs_glitch] == ["我不会放过你的", "我不会放过他的"]
    assert abs(segs_glitch[0].end_ms - frame_pts_ms[12]) <= 40
    assert abs(segs_glitch[1].start_ms - frame_pts_ms[13]) <= 40
    assert segs_glitch[1].end_ms == frame_pts_ms[24]


def test_stage2_pending_cluster_preserves_repeated_dialogue_across_240ms_gap():
    cfg = ExtractConfig(
        conf_threshold=0.85,
        min_duration_ms=120,
        merge_gap_ms=160,
        flash_bridge_max_ms=320,
    )
    frame_pts_ms = [i * 40 for i in range(27)]
    span1 = _make_encoded_span(list(range(0, 10)), frame_pts_ms[0:10], end_ms=frame_pts_ms[10])
    span2 = _make_encoded_span(list(range(16, 26)), frame_pts_ms[16:26], end_ms=frame_pts_ms[26])

    class RepeatEngine(OcrEngine):
        def __init__(self):
            super().__init__(provider="cpu", ocr_callable=lambda _img: (None, None))

        def recognize_roi(self, bgr_roi: np.ndarray, *, brightness_filter: bool = False):
            return ("快跑！", 0.98)

    segs = recognize_spans_with_voting([span1, span2], frame_pts_ms, RepeatEngine(), cfg)
    assert len(segs) == 2
    assert [(s.start_ms, s.end_ms, s.text) for s in segs] == [
        (0, 400, "快跑！"),
        (640, 1040, "快跑！"),
    ]

    span1_t = _make_encoded_span(list(range(0, 10)), frame_pts_ms[0:10], end_ms=frame_pts_ms[10])
    span2_t = _make_encoded_span(list(range(16, 26)), frame_pts_ms[16:26], end_ms=frame_pts_ms[26])

    class TruncEngine(OcrEngine):
        def __init__(self):
            super().__init__(provider="cpu", ocr_callable=lambda _img: (None, None))

        def recognize_roi(self, bgr_roi: np.ndarray, *, brightness_filter: bool = False):
            fidx = _decode_encoded_fidx(bgr_roi)
            if fidx < 10:
                return ("个蕴藏着神秘力量的", 0.98)
            return ("一个蕴藏着神秘力量的", 0.98)

    segs_t = recognize_spans_with_voting([span1_t, span2_t], frame_pts_ms, TruncEngine(), cfg)
    assert len(segs_t) == 1
    assert (segs_t[0].start_ms, segs_t[0].end_ms, segs_t[0].text) == (0, 1040, "一个蕴藏着神秘力量的")


def test_short_3char_distinct_dialogue_pairs_are_not_merged():
    assert _is_strict_same_cue("对不起", "真的对不起", 0.99, 0.99) is False
    assert _is_strict_same_cue("快走吧", "我们快走吧", 0.99, 0.99) is False
    assert _is_strict_same_cue("不会放过你", "真的不会放过你", 0.96, 0.95) is False
    assert _is_strict_same_cue("不会放过你", "真的不会放过你", 0.88, 0.96) is True


def test_low_conf_boundary_1char_glitch_clusters_and_valid_160ms_cue_survives():
    cfg = ExtractConfig(conf_threshold=0.85, min_duration_ms=120)

    assert _is_strict_same_cue("战斗开使了", "战斗开始了", 0.74, 0.98) is True
    assert _is_strict_same_cue("战斗开使了", "战斗开始了", 0.92, 0.98) is False

    pts_long = [i * 300 for i in range(12)]
    span_glitch = _make_encoded_span(list(range(11)), pts_long[:11], end_ms=pts_long[11])

    class EdgeTypoEngine(OcrEngine):
        def __init__(self):
            super().__init__(provider="cpu", ocr_callable=lambda _img: (None, None))

        def recognize_roi(self, bgr_roi: np.ndarray, *, brightness_filter: bool = False):
            fidx = _decode_encoded_fidx(bgr_roi)
            if fidx == 0:
                return ("战斗开使了", 0.74)
            return ("战斗开始了", 0.98)

    segs_edge = recognize_spans_with_voting([span_glitch], pts_long, EdgeTypoEngine(), cfg)
    assert len(segs_edge) == 1
    assert segs_edge[0].text == "战斗开始了"
    assert segs_edge[0].start_ms == 0
    assert segs_edge[0].end_ms == pts_long[11]

    pts_short = [0, 160]
    span_160 = _make_encoded_span([0], [0], end_ms=160)

    class SingleVoteEngine(OcrEngine):
        def __init__(self):
            super().__init__(provider="cpu", ocr_callable=lambda _img: (None, None))

        def recognize_roi(self, bgr_roi: np.ndarray, *, brightness_filter: bool = False):
            return ("太好了", 0.86)

    segs_160 = recognize_spans_with_voting([span_160], pts_short, SingleVoteEngine(), cfg)
    assert len(segs_160) == 1
    assert (segs_160[0].start_ms, segs_160[0].end_ms, segs_160[0].text) == (0, 160, "太好了")


def test_stage2_subgroups_stop_at_evaluated_none_gap_and_alternating_sandwich_safe():
    img = _solid_roi(88)
    eval_alt = [
        (0, _FrameVote(0, "第一句台词", 0.98, img, frame_idx=0)),
        (1, _FrameVote(40, "第二句台词", 0.98, img, frame_idx=1)),
        (2, _FrameVote(80, "第一句台词", 0.98, img, frame_idx=2)),
        (3, _FrameVote(120, "第二句台词", 0.98, img, frame_idx=3)),
    ]
    absorbed = _find_sandwich_noise(eval_alt)
    assert not ({1, 2}.issubset(absorbed))
    assert absorbed == {1}

    cfg = ExtractConfig(conf_threshold=0.85, min_duration_ms=120, flash_bridge_max_ms=320)
    frame_pts_ms = [i * 40 for i in range(21)]
    span_gap = _make_encoded_span(list(range(20)), frame_pts_ms[:20], end_ms=frame_pts_ms[20])

    class SilenceGapEngine(OcrEngine):
        def __init__(self):
            super().__init__(provider="cpu", ocr_callable=lambda _img: (None, None))

        def recognize_roi(self, bgr_roi: np.ndarray, *, brightness_filter: bool = False):
            fidx = _decode_encoded_fidx(bgr_roi)
            if fidx <= 5:
                return ("第一句台词", 0.98)
            if fidx <= 11:
                return ("", 0.0)
            return ("第二句台词", 0.98)

    segs = recognize_spans_with_voting([span_gap], frame_pts_ms, SilenceGapEngine(), cfg)
    assert [s.text for s in segs] == ["第一句台词", "第二句台词"]
    assert segs[0].end_ms <= frame_pts_ms[6]
    assert segs[1].start_ms == frame_pts_ms[12]
