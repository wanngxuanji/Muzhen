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
        # Encode fidx across a 16x16 block in channel 0 (scaled by 8) to survive JPEG Q95 losslessly
        img[0:16, 0:16, 0] = min(248, fidx * 8)
        span.add_sample(fidx, pts, img)
    return span


def _decode_encoded_fidx(bgr: np.ndarray) -> int:
    return int(round(float(bgr[4:12, 4:12, 0].mean()) / 8.0))


def test_stage2_early_exit_saves_ocr_calls_on_clean_consensus():
    cfg = ExtractConfig(conf_threshold=0.85, min_duration_ms=120)
    frame_pts_ms = [i * 180 for i in range(11)]  # 10 samples spanning 0..1800ms (dur=1800ms)
    span = _make_encoded_span(list(range(10)), frame_pts_ms[:10], end_ms=1800)

    called_frames: list[int] = []

    class CleanEngine(OcrEngine):
        def __init__(self):
            super().__init__(provider="cpu", ocr_callable=lambda _img: (None, None))

        def recognize_roi(self, bgr_roi: np.ndarray, *, brightness_filter: bool = False):
            fidx = _decode_encoded_fidx(bgr_roi)
            called_frames.append(fidx)
            # Edge frames 0 and 9 would be noisy if probed; interior frames are clean
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
    # 25fps (40ms/frame), single 24-sample span: frames 0..11 = A, frames 12..23 = B
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
    # Binary search must not brute-force all 24 frames
    assert len(called_frames) < 16


def test_stage2_weighted_consensus_pools_varied_edge_noise_and_symbol_artifacts():
    cfg = ExtractConfig(conf_threshold=0.85)
    dummy_engine = OcrEngine(provider="cpu", ocr_callable=lambda _img: (None, None))
    img1 = _solid_roi(11)
    img2 = _solid_roi(22)

    # Varied edge-noise prefixes pool weight onto clean vote "我们是救世"
    votes_varied = [
        _FrameVote(100, "G我们是救世", 0.78, img1, frame_idx=1),
        _FrameVote(200, "Y我们是救世", 0.76, img1, frame_idx=2),
        _FrameVote(300, "我们是救世", 0.99, img2, frame_idx=3),
    ]
    txt, conf, _ = _choose_majority_vote(votes_varied, dummy_engine, cfg)
    assert txt == "我们是救世"
    assert conf == pytest.approx(0.99)

    # Even 2 identical noisy votes lose to 1 clean vote once pooled
    votes_2v1 = [
        _FrameVote(100, "G我们是救世", 0.78, img1, frame_idx=1),
        _FrameVote(200, "G我们是救世", 0.78, img1, frame_idx=2),
        _FrameVote(300, "我们是救世", 0.95, img2, frame_idx=3),
    ]
    txt_2v1, conf_2v1, _ = _choose_majority_vote(votes_2v1, dummy_engine, cfg)
    assert txt_2v1 == "我们是救世"
    assert conf_2v1 == pytest.approx(0.95)

    # Stray symbol artifact "#一人" triggers brightness_filter=True retry (and 2nd frame retry if needed)
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

    # 2nd distinct frame retry kicks in when 1st frame retry remains noisy
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
    # 5-sample span: [A, A, GARBAGE, A, A] -> must remain 1 single cue A
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

    # 3 differing 1-char-error votes -> character-level alignment consensus reconstructs exact line
    votes_char = [
        _FrameVote(100, "你们一起去公园", 0.85, img, frame_idx=1),
        _FrameVote(200, "我们一起去公院", 0.86, img, frame_idx=2),
        _FrameVote(300, "我们一七去公园", 0.84, img, frame_idx=3),
    ]
    txt_char, _, _ = _choose_majority_vote(votes_char, dummy_engine, cfg)
    assert txt_char == "我们一起去公园"

    # cnt=1 clean fuller Hanzi containment (>= 0.92) beats 3x truncated votes
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

    # Opposite-metric floor (>= 0.30) rejects single-metric fluke
    edge = np.zeros((16, 64), dtype=np.uint8)
    edge[4:12, 10:50] = 1
    la = np.zeros((16, 64), dtype=np.float32)
    lb = np.zeros((16, 64), dtype=np.float32)
    la[4:12, 10:30] = 1.0
    la /= float(np.linalg.norm(la))
    lb[4:12, 10:30] = -1.0  # Inverted luma -> NCC = -1.0 < 0.30 despite edge_iou == 1.0
    lb /= float(np.linalg.norm(lb))
    sig_a = TightSignature(boxes=[_FIXED_BOX], union_rect=_FIXED_BOX, edge_mask=edge, luma_patch=la)
    sig_b = TightSignature(boxes=[_FIXED_BOX], union_rect=_FIXED_BOX, edge_mask=edge, luma_patch=lb)
    assert _tight_signatures_same_cue(sig_a, sig_b) is False


def test_stage2_glare_dodge_empty_early_exit_and_short_ghost_guard():
    cfg = ExtractConfig(conf_threshold=0.85, min_duration_ms=120)

    # 1) Glare/white-flash dodge on n >= 8: anchor frame 4 is a white flash (luma=250) -> dodged!
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

    # 2) Empty FP early-exit: 8-sample span (dur=800 <= 1200) returning empty stops after 3 probes
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

    # 3) Stage 2 short low-conf ghost guard drops <=2-vote <=2-char conf < 0.72, keeps conf >= 0.72
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
    # 3 distinct cues separated by > flash_bridge_max_ms
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
    # Span 1 must be flushed BEFORE span 3 OCR begins (O(1) streaming finalization)
    first_flush_pos = timeline.index("flush_thumb")
    first_span3_ocr_pos = timeline.index("ocr_span3")
    assert first_flush_pos < first_span3_ocr_pos

    # Also verify cancel_check is respected during final _flush_cluster
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
    # 24-sample span: true cut at sample 6 (frames 0..5 = A, 6..23 = B), blank glare on 8..12
    # so midpoint between lo=4 (A) and hi=15 (B) is None AFTER the true cut at 6.
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
    # 3-sample span: [truncated(0.88), GARBAGE(0.92), fuller(0.89)]
    # Sandwich noise has HIGHER conf than either real vote, so it must be excluded from voting!
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
