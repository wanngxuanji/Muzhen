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


def test_prepare_roi_for_ocr_normalizes_scale_without_synthetic_borders():
    from transapp.core.ocr_engine import _prepare_roi_for_ocr

    small_roi = _make_bgr(28, 160, color=(200, 180, 160))
    prep_small = _prepare_roi_for_ocr(small_roi)
    # Upscaled to >= 56 height without adding black top/bottom borders
    assert prep_small.shape[0] == 56
    assert prep_small.shape[1] == 320
    assert int(prep_small[0, 0, 0]) == 200

    normal_roi = _make_bgr(64, 160, color=(120, 120, 120))
    prep_norm = _prepare_roi_for_ocr(normal_roi)
    assert prep_norm.shape == (64, 160, 3)

    wide_roi = _make_bgr(170, 1536, color=(80, 80, 80))
    prep_wide = _prepare_roi_for_ocr(wide_roi)
    assert prep_wide.shape[1] == 960
    assert prep_wide.shape[0] == 106


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


def test_recognize_roi_parses_rapidocr_v3_output_object():
    class FakeRapidOCROutput:
        def __init__(self, boxes, txts, scores):
            self.boxes = boxes
            self.txts = txts
            self.scores = scores

    def fake_v3_ocr(_img: np.ndarray):
        boxes = np.array(
            [
                [[120.0, 50.0], [220.0, 50.0], [220.0, 74.0], [120.0, 74.0]],
                [[10.0, 52.0], [100.0, 52.0], [100.0, 76.0], [10.0, 76.0]],
            ],
            dtype=np.float32,
        )
        return FakeRapidOCROutput(boxes=boxes, txts=("米娜", "筱语"), scores=(0.98, 0.94))

    engine = OcrEngine(ocr_callable=fake_v3_ocr)
    text, conf = engine.recognize_roi(_make_bgr(64, 300))
    assert text == "筱语 米娜"
    assert conf == pytest.approx(0.94)

    empty_v3 = OcrEngine(ocr_callable=lambda _img: FakeRapidOCROutput(None, None, None))
    assert empty_v3.recognize_roi(_make_bgr(64, 300)) == ("", 0.0)


def test_ocr_engine_rejects_invalid_model_preset():
    with pytest.raises(ValueError, match="Unsupported OCR model preset"):
        OcrEngine(provider="cpu", ocr_model="unknown")


def test_available_providers_and_directml_fallback(monkeypatch, caplog):
    import onnxruntime
    import rapidocr

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
    init_params: list[dict] = []

    class FakeRapidOCR:
        def __init__(self, params=None):
            init_params.append(dict(params or {}))
            if (params or {}).get("EngineConfig.onnxruntime.use_dml"):
                raise RuntimeError("DirectML init failed")

        def __call__(self, _img):
            return ([[[[0, 0], [30, 0], [30, 20], [0, 20]], "回退成功", 0.92]], [0.01])

    monkeypatch.setattr(rapidocr, "RapidOCR", FakeRapidOCR)
    monkeypatch.setattr(
        onnxruntime,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )

    caplog.set_level(logging.WARNING)
    engine = OcrEngine(provider="directml")
    assert init_params == []  # lazy init
    text, conf = engine.recognize_roi(_make_bgr(64, 160))
    assert (text, conf) == ("回退成功", pytest.approx(0.92))
    assert len(init_params) == 1
    assert init_params[0]["Global.use_cls"] is False
    assert init_params[0]["Global.use_vertical_padding"] is False
    assert init_params[0]["Global.width_height_ratio"] == -1
    assert init_params[0]["Det.ocr_version"] == rapidocr.OCRVersion.PPOCRV5
    assert init_params[0]["Det.model_type"] == rapidocr.ModelType.SERVER
    assert init_params[0]["Rec.model_type"] == rapidocr.ModelType.SERVER
    assert init_params[0]["Det.lang_type"] == rapidocr.LangDet.CH
    assert init_params[0]["Rec.lang_type"] == rapidocr.LangRec.CH
    assert init_params[0]["Det.limit_type"] == "max"
    assert init_params[0]["Det.limit_side_len"] == 960
    assert "EngineConfig.onnxruntime.use_dml" not in init_params[0]
    assert any("directml" in rec.message.lower() for rec in caplog.records)

    # When DmlExecutionProvider is listed but RapidOCR init fails, still fall back to CPU with warning
    caplog.clear()
    init_params.clear()
    monkeypatch.setattr(
        onnxruntime,
        "get_available_providers",
        lambda: ["DmlExecutionProvider", "CPUExecutionProvider"],
    )
    engine2 = OcrEngine(provider="directml", ocr_model="fast")
    text2, conf2 = engine2.recognize_roi(_make_bgr(64, 160))
    assert (text2, conf2) == ("回退成功", pytest.approx(0.92))
    assert len(init_params) == 2
    assert init_params[0]["EngineConfig.onnxruntime.use_dml"] is True
    assert init_params[0]["Det.model_type"] == rapidocr.ModelType.MOBILE
    assert init_params[0]["Rec.model_type"] == rapidocr.ModelType.MOBILE
    assert "EngineConfig.onnxruntime.use_dml" not in init_params[1]
    assert init_params[1]["Det.model_type"] == rapidocr.ModelType.MOBILE
    assert any("directml" in rec.message.lower() for rec in caplog.records)

    # Default CPU provider passes ch_PP-OCRv5 server params with use_cls disabled
    init_params.clear()
    cpu_engine = OcrEngine(provider="cpu", ocr_model="accurate")
    assert cpu_engine.recognize_roi(_make_bgr(64, 160)) == ("回退成功", pytest.approx(0.92))
    assert len(init_params) == 1
    assert init_params[0]["Det.model_type"] == rapidocr.ModelType.SERVER
    assert init_params[0]["Rec.model_type"] == rapidocr.ModelType.SERVER
    assert "EngineConfig.onnxruntime.use_dml" not in init_params[0]


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


def test_group_boxes_drops_detached_low_conf_non_hanzi_noise_and_keeps_grid_mapping():
    # Wall inscription "1X1- TO X1-" (0.76) beside strong Hanzi subtitle "停下来了" (0.99) -> dropped
    def fake_noisy(_img: np.ndarray):
        items = [
            [[[120.0, 20.0], [240.0, 20.0], [240.0, 50.0], [120.0, 50.0]], "停下来了", 0.99],
            [[[360.0, 22.0], [490.0, 22.0], [490.0, 50.0], [360.0, 50.0]], "1X1- TO X1-", 0.76],
        ]
        return (items, [0.01])

    engine = OcrEngine(ocr_callable=fake_noisy)
    text, conf = engine.recognize_roi(_make_bgr(76, 512))
    assert text == "停下来了"
    assert conf == pytest.approx(0.99)

    # Grid detection batch maps boxes back to their respective cell indices
    class FakeDetGrid:
        def __call__(self, canvas: np.ndarray):
            class _DetOut:
                # Cell 0 occupies y in [20, 84], cell 2 occupies y in [188, 252]
                boxes = np.array(
                    [
                        [[50.0, 28.0], [250.0, 28.0], [250.0, 68.0], [50.0, 68.0]],
                        [[80.0, 190.0], [300.0, 190.0], [300.0, 226.0], [80.0, 226.0]],
                    ],
                    dtype=np.float32,
                )

            return _DetOut()

    det_only_engine = OcrEngine(det_callable=FakeDetGrid())
    rois = [_make_bgr(64, 400) for _ in range(3)]
    mapped = det_only_engine.detect_grid_batch(rois)
    assert len(mapped) == 3
    assert len(mapped[0]) == 1
    assert mapped[1] == []
    assert len(mapped[2]) == 1
