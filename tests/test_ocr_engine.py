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


def test_det_mobile_params_and_batch_size(monkeypatch):
    import onnxruntime
    import rapidocr
    from transapp.core.ocr_engine import _build_det_mobile_params

    cpu_params = _build_det_mobile_params(use_dml=False)
    assert cpu_params["Det.box_thresh"] == pytest.approx(0.55)
    assert cpu_params["Det.thresh"] == pytest.approx(0.30)
    assert cpu_params["Det.use_dilation"] is True
    assert cpu_params["Det.score_mode"] == "fast"
    assert "EngineConfig.onnxruntime.use_dml" not in cpu_params

    dml_params = _build_det_mobile_params(use_dml=True)
    assert dml_params["Det.box_thresh"] == pytest.approx(0.55)
    assert dml_params["Det.thresh"] == pytest.approx(0.30)
    assert dml_params["Det.use_dilation"] is True
    assert dml_params["Det.score_mode"] == "fast"
    assert dml_params["EngineConfig.onnxruntime.use_dml"] is True

    assert OcrEngine(provider="directml").det_batch_size == 16
    assert OcrEngine(provider="cpu").det_batch_size == 8

    # When DirectML falls back to CPU on Det init, det_batch_size drops to 8,
    # and a fake RapidOCR without _load_det_model falls back to per-ROI det_engine(roi) calls.
    seen_shapes: list[tuple[int, ...]] = []

    class FakePerRoiRapidOCR:
        def __init__(self, params=None):
            self.params = params or {}

        def __call__(self, img: np.ndarray):
            seen_shapes.append(img.shape)

            class _Out:
                boxes = np.array(
                    [[[20.0, 10.0], [180.0, 10.0], [180.0, 40.0], [20.0, 40.0]]],
                    dtype=np.float32,
                )

            return _Out()

    monkeypatch.setattr(rapidocr, "RapidOCR", FakePerRoiRapidOCR)
    monkeypatch.setattr(
        onnxruntime,
        "get_available_providers",
        lambda: ["CPUExecutionProvider"],
    )
    fallback_engine = OcrEngine(provider="directml")
    assert fallback_engine.det_batch_size == 16
    rois = [_make_bgr(64, 320), _make_bgr(64, 320)]
    out = fallback_engine.detect_grid_batch(rois)
    assert fallback_engine.det_batch_size == 8
    assert seen_shapes == [(64, 320, 3), (64, 320, 3)]
    assert len(out) == 2
    assert len(out[0]) == 1 and len(out[1]) == 1


def test_detect_batch_4d_native_tensor_and_empty_map_short_circuit():
    dark_roi = _load_bgr(FIXTURES_DIR / "sample_dark_bg.png")
    bright_roi = _load_bgr(FIXTURES_DIR / "sample_bright_bg.png")
    # Bright colored blank anime ROI (orange BGR) that previously triggered black-grid edge false positives
    blank_bright_orange = _make_bgr(dark_roi.shape[0], dark_roi.shape[1], color=(40, 165, 255))

    engine = OcrEngine(provider="cpu")
    det_engine = engine._ensure_det_engine()
    det_model = det_engine._load_det_model()

    session_shapes: list[tuple[int, ...]] = []
    orig_session = det_model.session

    def spy_session(batch_tensor: np.ndarray) -> np.ndarray:
        session_shapes.append(tuple(batch_tensor.shape))
        return orig_session(batch_tensor)

    postprocess_calls = 0
    orig_postprocess = det_model.postprocess_op

    def spy_postprocess(pred: np.ndarray, ori_shape: tuple[int, int]):
        nonlocal postprocess_calls
        postprocess_calls += 1
        return orig_postprocess(pred, ori_shape)

    det_model.session = spy_session
    det_model.postprocess_op = spy_postprocess

    results = engine.detect_grid_batch([dark_roi, blank_bright_orange, bright_roi])

    assert len(results) == 3
    assert len(results[0]) >= 1, "Expected subtitle box on sample_dark_bg.png"
    assert results[1] == [], "Blank bright orange ROI must produce zero boxes"
    assert len(results[2]) >= 1, "Expected subtitle box on sample_bright_bg.png"
    # Must execute as a single 4D NCHW batch (N=3, C=3, rh, rw)
    assert len(session_shapes) == 1
    assert len(session_shapes[0]) == 4
    assert session_shapes[0][0] == 3
    assert session_shapes[0][1] == 3
    # Empty-map short-circuit must skip postprocess_op for the blank frame
    assert postprocess_calls == 2


def test_detect_batch_preserves_small_subtitle_in_tall_roi():
    from transapp.core.ocr_engine import _polys_to_normalized_boxes

    # Exact 3% height box is kept (previously dropped by > 0.05); <= 2% tiny sliver is dropped
    poly_3pct = np.array([[100.0, 900.0], [500.0, 900.0], [500.0, 930.0], [100.0, 930.0]], dtype=np.float32)
    poly_1pct = np.array([[100.0, 100.0], [500.0, 100.0], [500.0, 115.0], [100.0, 115.0]], dtype=np.float32)
    kept = _polys_to_normalized_boxes([poly_3pct, poly_1pct], orig_w=1000, orig_h=1000)
    assert len(kept) == 1
    assert (kept[0][3] - kept[0][1]) == pytest.approx(0.03)

    # Real Det Mobile on a tall ROI (height=1800) where a 1-line subtitle occupies ~4% (< 5%) of ROI height
    dark_roi = _load_bgr(FIXTURES_DIR / "sample_dark_bg.png")
    h_sub, w_sub = dark_roi.shape[:2]
    bg_color = tuple(int(c) for c in np.median(dark_roi[0, :, :], axis=0))
    tall_roi = _make_bgr(1800, w_sub, color=bg_color)
    y_top = 1800 - h_sub - 40
    tall_roi[y_top : y_top + h_sub, 0:w_sub] = dark_roi

    engine = OcrEngine(provider="cpu")
    results = engine.detect_grid_batch([tall_roi])

    assert len(results) == 1
    assert len(results[0]) >= 1
    assert any(0.02 < (y2 - y1) < 0.05 for (_x1, y1, _x2, y2) in results[0])


def test_det_directml_lazy_load_failure_and_silent_cpu_fallback(monkeypatch, caplog):
    import onnxruntime
    import rapidocr

    monkeypatch.setattr(
        onnxruntime,
        "get_available_providers",
        lambda: ["DmlExecutionProvider", "CPUExecutionProvider"],
    )

    # Case 1: RapidOCR.__init__ succeeds with use_dml=True, but lazy _load_det_model() fails
    init_flags: list[bool] = []

    class FakeLazyFailRapidOCR:
        def __init__(self, params=None):
            self._use_dml = bool((params or {}).get("EngineConfig.onnxruntime.use_dml"))
            init_flags.append(self._use_dml)

        def _load_det_model(self):
            if self._use_dml:
                raise RuntimeError("DirectML session creation failed lazily")

            class _FakeOrtSession:
                def get_providers(self):
                    return ["CPUExecutionProvider"]

            class _FakeSession:
                session = _FakeOrtSession()

                def __call__(self, tensor: np.ndarray) -> np.ndarray:
                    n, _c, h, w = tensor.shape
                    return np.zeros((n, 1, h, w), dtype=np.float32)

            class _FakeDetModel:
                mean = [0.5, 0.5, 0.5]
                std = [0.5, 0.5, 0.5]
                session = _FakeSession()

                @staticmethod
                def postprocess_op(_pred, _shape):
                    return np.empty((0, 4, 2), dtype=np.float32), []

            return _FakeDetModel()

    monkeypatch.setattr(rapidocr, "RapidOCR", FakeLazyFailRapidOCR)
    caplog.set_level(logging.WARNING)
    eng1 = OcrEngine(provider="directml")
    assert eng1.det_batch_size == 16
    assert eng1.detect_grid_batch([_make_bgr(64, 320)]) == [[]]
    assert init_flags == [True, False]
    assert eng1.det_batch_size == 8
    assert any("directml" in rec.message.lower() for rec in caplog.records)

    # Case 2: _load_det_model() succeeds without raising, but ORT silently dropped DirectML to CPU
    class FakeSilentCpuFallbackRapidOCR:
        def __init__(self, params=None):
            self.params = params or {}

        def _load_det_model(self):
            class _FakeOrtSession:
                def get_providers(self):
                    return ["CPUExecutionProvider"]

            class _FakeSession:
                session = _FakeOrtSession()

                def __call__(self, tensor: np.ndarray) -> np.ndarray:
                    n, _c, h, w = tensor.shape
                    return np.zeros((n, 1, h, w), dtype=np.float32)

            class _FakeDetModel:
                mean = [0.5, 0.5, 0.5]
                std = [0.5, 0.5, 0.5]
                session = _FakeSession()

                @staticmethod
                def postprocess_op(_pred, _shape):
                    return np.empty((0, 4, 2), dtype=np.float32), []

            return _FakeDetModel()

    monkeypatch.setattr(rapidocr, "RapidOCR", FakeSilentCpuFallbackRapidOCR)
    eng2 = OcrEngine(provider="directml")
    assert eng2.det_batch_size == 16
    assert eng2.detect_grid_batch([_make_bgr(64, 320)]) == [[]]
    assert eng2.det_batch_size == 8


def test_detect_batch_empty_zero_size_dtype_and_portrait_height_cap():
    dark_roi = _load_bgr(FIXTURES_DIR / "sample_dark_bg.png")
    empty_roi = np.empty((0, 0, 3), dtype=np.uint8)
    float_roi = dark_roi.astype(np.float32)

    engine = OcrEngine(provider="cpu")
    assert engine.detect_grid_batch([]) == []

    # Mixed zero-size ROI + uint8 ROI + float32 ROI in a real 4D batch
    mixed = engine.detect_grid_batch([empty_roi, dark_roi, float_roi])
    assert len(mixed) == 3
    assert mixed[0] == []
    assert len(mixed[1]) >= 1
    assert len(mixed[2]) >= 1

    # Pathological portrait ROI (w <= 640, h >> 960) stays capped at rh <= 960
    det_model = engine._ensure_det_engine()._load_det_model()
    seen_shapes: list[tuple[int, ...]] = []
    orig_session = det_model.session

    def spy_session(batch_tensor: np.ndarray) -> np.ndarray:
        seen_shapes.append(tuple(batch_tensor.shape))
        return orig_session(batch_tensor)

    det_model.session = spy_session
    portrait_roi = _make_bgr(2400, 400, color=(20, 20, 20))
    assert engine.detect_grid_batch([portrait_roi]) == [[]]
    assert len(seen_shapes) == 1
    assert seen_shapes[0][2] <= 960

    # Injected det_callable path also tolerates all-empty or leading-empty ROIs
    called = 0

    def fake_det(_canvas: np.ndarray):
        nonlocal called
        called += 1

        class _Out:
            boxes = None

        return _Out()

    det_mock = OcrEngine(det_callable=fake_det)
    assert det_mock.detect_grid_batch([empty_roi]) == [[]]
    assert called == 0
    assert det_mock.detect_grid_batch([empty_roi, _make_bgr(64, 320)]) == [[], []]
    assert called == 1


def test_ocr_engine_drops_offband_tiny_hanzi_and_serializes_concurrent_calls():
    import threading
    import time

    # 1) Off-band low-conf 1-char Hanzi & tiny low-conf 1-char Hanzi are dropped when strong Hanzi (>=0.90) exists
    def fake_spatial_outliers(_img: np.ndarray):
        items = [
            # Strong main subtitle line: cy=50, h=30, score=0.98
            [[[80.0, 35.0], [340.0, 35.0], [340.0, 65.0], [80.0, 65.0]], "不用亲眼看见", 0.98],
            # Detached off-band 1-char low-conf Hanzi: cy=10 (|10-50|=40 > 0.8*30=24), score=0.62 -> drop
            [[[20.0, 0.0], [45.0, 0.0], [45.0, 20.0], [20.0, 20.0]], "生", 0.62],
            # Tiny 1-char low-conf Hanzi on same band: h=12 (< 0.55*30=16.5), score=0.70 -> drop
            [[[360.0, 44.0], [375.0, 44.0], [375.0, 56.0], [360.0, 56.0]], "人", 0.70],
        ]
        return (items, [0.01])

    outlier_engine = OcrEngine(ocr_callable=fake_spatial_outliers)
    text, conf = outlier_engine.recognize_roi(_make_bgr(80, 420))
    assert text == "不用亲眼看见"
    assert conf == pytest.approx(0.98)

    # 2) Genuine 2-line subtitle where both lines are full-height & high-score MUST be preserved
    def fake_two_line(_img: np.ndarray):
        items = [
            [[[80.0, 10.0], [320.0, 10.0], [320.0, 38.0], [80.0, 38.0]], "第一行台词", 0.96],
            [[[180.0, 46.0], [210.0, 46.0], [210.0, 74.0], [180.0, 74.0]], "嗯", 0.93],
        ]
        return (items, [0.01])

    two_line_engine = OcrEngine(ocr_callable=fake_two_line)
    text2, conf2 = two_line_engine.recognize_roi(_make_bgr(84, 420))
    assert text2 == "第一行台词\n嗯"
    assert conf2 == pytest.approx(0.93)

    # 3) DirectML / ONNX thread safety: _inference_lock serializes concurrent recognize_roi & detect_grid_batch
    active = 0
    max_active = 0
    state_lock = threading.Lock()
    barrier = threading.Barrier(2)

    def slow_ocr(_img: np.ndarray):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.05)
            return ([[[[10, 10], [80, 10], [80, 34], [10, 34]], "并发测试", 0.97]], [0.05])
        finally:
            with state_lock:
                active -= 1

    def slow_det(_canvas: np.ndarray):
        nonlocal active, max_active
        with state_lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.05)

            class _Out:
                boxes = np.array(
                    [[[20.0, 24.0], [180.0, 24.0], [180.0, 52.0], [20.0, 52.0]]],
                    dtype=np.float32,
                )

            return _Out()
        finally:
            with state_lock:
                active -= 1

    shared_engine = OcrEngine(ocr_callable=slow_ocr, det_callable=slow_det)
    assert hasattr(shared_engine, "_inference_lock")

    results: list[object] = []

    def run_rec():
        barrier.wait()
        results.append(shared_engine.recognize_roi(_make_bgr(64, 240)))

    def run_det():
        barrier.wait()
        results.append(shared_engine.detect_grid_batch([_make_bgr(64, 240)]))

    t1 = threading.Thread(target=run_rec)
    t2 = threading.Thread(target=run_det)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert max_active == 1, f"Expected serialized ONNX/DirectML calls, saw {max_active} overlapping"
    assert len(results) == 2



