"""Unit tests for the 2-Stage Deep Learning extractor with Majority Voting."""

from __future__ import annotations

import numpy as np
import pytest

from transapp.core.deep_extractor import _FrameVote, _choose_majority_vote, extract_subtitles_deep
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
