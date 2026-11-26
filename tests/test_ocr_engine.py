from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

from transapp.core.models import ExtractionCancelled, OcrSegment, RawSegment
from transapp.core.ocr_engine import OcrEngine

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    assert img is not None, f"Failed to decode fixture: {path}"
    return img


def _make_bgr(h: int, w: int, color: tuple[int, int, int] = (20, 20, 20)) -> np.ndarray:
    return np.full((h, w, 3), color, dtype=np.uint8)


def test_ocr_engine_reexported_from_core_package():
    from transapp.core import OcrEngine as ReexportedOcrEngine

    assert ReexportedOcrEngine is OcrEngine


def test_recognize_roi_pads_and_upscales_only_when_height_below_48():
    seen_shapes: list[tuple[int, int, int]] = []

    def fake_ocr(img: np.ndarray):
        seen_shapes.append(img.shape)
        return ([[[[0, 0], [50, 0], [50, 20], [0, 20]], "测试", 0.95]], [0.01])

    engine = OcrEngine(ocr_callable=fake_ocr)

    small_roi = _make_bgr(28, 160)
    text_small, conf_small = engine.recognize_roi(small_roi)
    assert (text_small, conf_small) == ("测试", pytest.approx(0.95))
    # Upscaled to >= 48 height plus 10px border on top and bottom (>= 68)
    assert seen_shapes[0][0] >= 68
    assert seen_shapes[0][1] > 160

    normal_roi = _make_bgr(64, 160)
    text_norm, conf_norm = engine.recognize_roi(normal_roi)
    assert (text_norm, conf_norm) == ("测试", pytest.approx(0.95))
    assert seen_shapes[1] == (64, 160, 3)


def test_recognize_roi_sorts_multiline_boxes_and_uses_min_confidence():
    def fake_ocr(_img: np.ndarray):
        items = [
            # Bottom line, right box
            [[[200.0, 60.0], [300.0, 60.0], [300.0, 80.0], [200.0, 80.0]], "world", 0.95],
            # Top line, right box
            [[[150.0, 12.0], [250.0, 12.0], [250.0, 32.0], [150.0, 32.0]], "B", 0.91],
            # Bottom line, left box (within 0.5 * line_height of y=70, lowest score)
            [[[20.0, 63.0], [120.0, 63.0], [120.0, 83.0], [20.0, 83.0]], "hello", 0.73],
            # Top line, left box
            [[[20.0, 10.0], [100.0, 10.0], [100.0, 30.0], [20.0, 30.0]], "A", 0.99],
        ]
        return (items, [0.01])

    engine = OcrEngine(ocr_callable=fake_ocr)
    text, conf = engine.recognize_roi(_make_bgr(96, 360))

    assert text == "A B\nhello world"
    assert conf == pytest.approx(0.73)


def test_recognize_roi_empty_and_confidence_clipping():
    empty_engine = OcrEngine(ocr_callable=lambda _img: (None, None))
    assert empty_engine.recognize_roi(_make_bgr(64, 160)) == ("", 0.0)
    assert empty_engine.recognize_roi(np.empty((0, 0, 3), dtype=np.uint8)) == ("", 0.0)

    high_conf_engine = OcrEngine(
        ocr_callable=lambda _img: (
            [[[[0, 0], [10, 0], [10, 20], [0, 20]], "好", 1.15]],
            [0.01],
        )
    )
    assert high_conf_engine.recognize_roi(_make_bgr(64, 160)) == ("好", 1.0)

    low_conf_engine = OcrEngine(
        ocr_callable=lambda _img: (
            [[[[0, 0], [10, 0], [10, 20], [0, 20]], "好", -0.2]],
            [0.01],
        )
    )
    assert low_conf_engine.recognize_roi(_make_bgr(64, 160)) == ("好", 0.0)


def test_encode_thumbnail_png_valid_and_empty_raises():
    engine = OcrEngine(ocr_callable=lambda _img: (None, None))
    roi = _make_bgr(50, 120, color=(40, 120, 200))

    png_bytes = engine.encode_thumbnail_png(roi)
    assert isinstance(png_bytes, bytes)
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    decoded = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == roi.shape

    with pytest.raises(ValueError):
        engine.encode_thumbnail_png(np.empty((0, 0, 3), dtype=np.uint8))


def test_process_segments_emits_progress_and_respects_cancellation():
    calls = 0

    def fake_ocr(_img: np.ndarray):
        nonlocal calls
        calls += 1
        return ([[[[0, 0], [40, 0], [40, 20], [0, 20]], f"line{calls}", 0.9]], [0.01])

    engine = OcrEngine(ocr_callable=fake_ocr)
    segments = [
        RawSegment(start_ms=0, end_ms=500, peak_bgr=_make_bgr(56, 160), frame_index=0),
        RawSegment(start_ms=500, end_ms=1000, peak_bgr=_make_bgr(56, 160), frame_index=12),
        RawSegment(start_ms=1000, end_ms=1500, peak_bgr=_make_bgr(56, 160), frame_index=24),
    ]
    progress: list[tuple[int, int]] = []

    out = engine.process_segments(
        segments,
        progress_cb=lambda cur, tot: progress.append((cur, tot)),
    )
    assert len(out) == 3
    assert all(isinstance(seg, OcrSegment) for seg in out)
    assert [seg.text for seg in out] == ["line1", "line2", "line3"]
    assert all(seg.thumbnail_png.startswith(b"\x89PNG\r\n\x1a\n") for seg in out)
    assert progress == [(1, 3), (2, 3), (3, 3)]

    # Cancellation before second segment
    calls = 0
    cancel_progress: list[tuple[int, int]] = []
    check_count = 0

    def cancel_on_second() -> bool:
        nonlocal check_count
        check_count += 1
        return check_count >= 2

    with pytest.raises(ExtractionCancelled):
        engine.process_segments(
            segments,
            progress_cb=lambda cur, tot: cancel_progress.append((cur, tot)),
            cancel_check=cancel_on_second,
        )
    assert calls == 1
    assert cancel_progress == [(1, 3)]


def test_available_providers_and_directml_fallback(monkeypatch, caplog):
    import onnxruntime
    import rapidocr_onnxruntime

    monkeypatch.setattr(
        onnxruntime,
        "get_available_providers",
        lambda: ["AzureExecutionProvider", "CPUExecutionProvider"],
    )
    assert OcrEngine.available_providers() == ["cpu"]

    monkeypatch.setattr(
        onnxruntime,
        "get_available_providers",
        lambda: ["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    assert OcrEngine.available_providers() == ["cpu", "directml"]

    # When DmlExecutionProvider is unavailable, directml falls back to CPU with warning
    init_kwargs: list[dict] = []

    class FakeRapidOCR:
        def __init__(self, **kwargs):
            init_kwargs.append(kwargs)
            if kwargs.get("det_use_dml"):
                raise RuntimeError("DirectML init failed")

        def __call__(self, _img):
            return ([[[[0, 0], [30, 0], [30, 20], [0, 20]], "回退成功", 0.92]], [0.01])

    monkeypatch.setattr(rapidocr_onnxruntime, "RapidOCR", FakeRapidOCR)
    monkeypatch.setattr(
        onnxruntime,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )

    caplog.set_level(logging.WARNING)
    engine = OcrEngine(provider="directml")
    assert init_kwargs == []  # lazy init
    text, conf = engine.recognize_roi(_make_bgr(64, 160))
    assert (text, conf) == ("回退成功", pytest.approx(0.92))
    assert init_kwargs == [{"det_limit_type": "max", "det_limit_side_len": 960}]
    assert any("directml" in rec.message.lower() for rec in caplog.records)

    # When DmlExecutionProvider is listed but RapidOCR init fails, still fall back to CPU with warning
    caplog.clear()
    init_kwargs.clear()
    monkeypatch.setattr(
        onnxruntime,
        "get_available_providers",
        lambda: ["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    engine2 = OcrEngine(provider="directml")
    text2, conf2 = engine2.recognize_roi(_make_bgr(64, 160))
    assert (text2, conf2) == ("回退成功", pytest.approx(0.92))
    assert init_kwargs == [
        {
            "det_limit_type": "max",
            "det_limit_side_len": 960,
            "det_use_dml": True,
            "cls_use_dml": True,
            "rec_use_dml": True,
        },
        {"det_limit_type": "max", "det_limit_side_len": 960},
    ]
    assert any("directml" in rec.message.lower() for rec in caplog.records)

    # Default CPU provider also passes calibrated det_limit_* kwargs
    init_kwargs.clear()
    cpu_engine = OcrEngine(provider="cpu")
    assert cpu_engine.recognize_roi(_make_bgr(64, 160)) == ("回退成功", pytest.approx(0.92))
    assert init_kwargs == [{"det_limit_type": "max", "det_limit_side_len": 960}]


@pytest.mark.parametrize(
    ("fixture_name", "expected_text"),
    [
        ("sample_dark_bg.png", "不用亲眼看见即将开始的地狱"),
        ("sample_bright_bg.png", "大家都怎么样了"),
    ],
)
def test_real_rapidocr_recognizes_sample_fixtures(fixture_name: str, expected_text: str):
    roi = _load_bgr(FIXTURES_DIR / fixture_name)
    engine = OcrEngine(provider="cpu")

    text, conf = engine.recognize_roi(roi)

    assert text == expected_text
    assert conf >= 0.80
