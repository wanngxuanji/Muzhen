from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest

from transapp.core.models import (
    ExtractConfig,
    ExtractionCancelled,
    ExtractResult,
    FrameTestResult,
    RoiRect,
    SubtitleEvent,
)
from transapp.core.ocr_engine import OcrEngine
from transapp.core.subtitle_mask import mask_fg_ratio
from transapp.core.video_reader import VideoReader
from transapp.exporters import export_ass, export_srt


def _load_cjk_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _make_synthetic_subtitle_mp4(
    path: Path,
    *,
    width: int = 640,
    height: int = 360,
    fps: int = 25,
    duration_sec: float = 3.0,
    start_pts_ms: int = 0,
    cues: tuple[tuple[int, int, str], ...] = (
        (200, 1100, "你好世界"),
        (1450, 2500, "战斗开始"),
    ),
) -> None:
    """Create a synthetic MP4 with outlined white text over a changing colored background."""
    total_frames = int(round(duration_sec * fps))
    font = _load_cjk_font(size=34)

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)

        for idx in range(total_frames):
            pts_ms = int(round(idx * 1000.0 / fps))
            # Vary background color and draw moving shapes behind ROI so background changes constantly.
            bg_r = 40 + (idx * 7) % 160
            bg_g = 90 + (idx * 5) % 130
            bg_b = 180 - (idx * 4) % 140
            img = Image.new("RGB", (width, height), (bg_r, bg_g, bg_b))
            draw = ImageDraw.Draw(img)

            bar_x = (idx * 14) % (width - 100)
            draw.rectangle(
                [bar_x, int(height * 0.72), bar_x + 100, int(height * 0.96)],
                fill=(220, 140, 45),
            )

            active_text = next(
                (text for start_ms, end_ms, text in cues if start_ms <= pts_ms < end_ms),
                None,
            )
            if active_text is not None:
                draw.text(
                    (width // 2, int(height * 0.85)),
                    active_text,
                    font=font,
                    anchor="mm",
                    fill=(250, 250, 250),
                    stroke_width=3,
                    stroke_fill=(8, 8, 8),
                )

            bgr = np.asarray(img, dtype=np.uint8)[:, :, ::-1].copy()
            frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
            frame.pts = start_pts_ms + pts_ms
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)


def test_pipeline_and_cancellation_reexports():
    from transapp.core import (
        ExtractionCancelled as CoreCancelled,
        ExtractionPipeline as CorePipeline,
    )
    from transapp.core.pipeline import (
        ExtractionCancelled as PipeCancelled,
        ExtractionPipeline as PipePipeline,
    )

    assert PipeCancelled is ExtractionCancelled
    assert CoreCancelled is ExtractionCancelled
    assert CorePipeline is PipePipeline


def test_single_frame_computes_mask_and_normalizes_ocr_text(tmp_path: Path):
    from transapp.core.pipeline import ExtractionPipeline

    video_path = tmp_path / "single_frame_测试.mp4"
    _make_synthetic_subtitle_mp4(video_path, duration_sec=1.5, cues=((200, 1200, "测试字幕"),))

    seen_roi_shapes: list[tuple[int, int, int]] = []

    def fake_ocr(img: np.ndarray):
        seen_roi_shapes.append(img.shape)
        return ([[[[0, 0], [80, 0], [80, 24], [0, 24]], " 你 好 \t 世 界 ", 0.91]], [0.01])

    pipeline = ExtractionPipeline(ocr_engine=OcrEngine(ocr_callable=fake_ocr))
    roi = RoiRect(0.10, 0.74, 0.80, 0.20)
    config = ExtractConfig()

    res = pipeline.test_single_frame(video_path, 600, roi, config)

    assert isinstance(res, FrameTestResult)
    assert res.frame_index >= 0
    assert abs(res.pts_ms - 600) <= 60
    x1, y1, x2, y2 = roi.to_pixels(640, 360)
    assert res.mask.shape == (y2 - y1, x2 - x1)
    assert res.mask.dtype == np.uint8
    assert set(np.unique(res.mask).tolist()).issubset({0, 255})
    assert mask_fg_ratio(res.mask) > config.fg_ratio_min
    assert seen_roi_shapes == [(y2 - y1, x2 - x1, 3)]
    assert res.text == "你好世界"
    assert res.confidence == pytest.approx(0.91)


def test_pipeline_end_to_end_synthetic_mp4_and_exporters(tmp_path: Path):
    from transapp.core.pipeline import ExtractionPipeline

    video_path = tmp_path / "synthetic_3s_字幕.mp4"
    _make_synthetic_subtitle_mp4(
        video_path,
        width=640,
        height=360,
        fps=25,
        duration_sec=3.0,
        cues=(
            (200, 1120, "你好世界"),
            (1480, 2480, "战斗开始"),
        ),
    )

    roi = RoiRect(0.10, 0.74, 0.80, 0.20)
    config = ExtractConfig(provider="cpu")
    pipeline = ExtractionPipeline()

    # Single-frame preview on cue 1
    frame_preview = pipeline.test_single_frame(video_path, 600, roi, config)
    assert isinstance(frame_preview, FrameTestResult)
    assert mask_fg_ratio(frame_preview.mask) > config.fg_ratio_min
    assert "你好" in frame_preview.text or "世界" in frame_preview.text
    assert frame_preview.confidence >= 0.70

    # Full pipeline extraction with progress tracking
    progress_calls: list[tuple[str, int, int]] = []
    result = pipeline.run(
        video_path,
        roi,
        config,
        progress_cb=lambda stage, cur, tot: progress_calls.append((stage, cur, tot)),
    )

    assert isinstance(result, ExtractResult)
    assert result.video_path == video_path
    assert result.width == 640
    assert result.height == 360
    assert result.duration_ms >= 2900
    assert result.elapsed_sec > 0.0
    assert len(result.frame_pts_ms) >= 70
    assert result.frame_pts_ms[0] == 0
    assert result.frame_pts_ms == sorted(result.frame_pts_ms)

    assert len(result.events) == 2
    assert all(isinstance(ev, SubtitleEvent) for ev in result.events)
    assert [ev.index for ev in result.events] == [1, 2]

    ev1, ev2 = result.events
    assert 120 <= ev1.start_ms < ev1.end_ms <= 1300
    assert 1350 <= ev2.start_ms < ev2.end_ms <= result.duration_ms + 100
    assert ev1.end_ms <= ev2.start_ms
    assert "你好" in ev1.text and "世界" in ev1.text
    assert "战斗" in ev2.text and "开始" in ev2.text
    assert all(ev.thumbnail_png.startswith(b"\x89PNG\r\n\x1a\n") for ev in result.events)

    scan_calls = [c for c in progress_calls if c[0] == "scan"]
    ocr_calls = [c for c in progress_calls if c[0] == "ocr"]
    done_calls = [c for c in progress_calls if c[0] == "done"]

    assert len(scan_calls) > 0
    assert all(tot >= 1 and 0 <= cur <= tot for _, cur, tot in scan_calls)
    assert [cur for _, cur, _ in scan_calls] == sorted(cur for _, cur, _ in scan_calls)
    assert ocr_calls == [(("ocr", 1, 2)), (("ocr", 2, 2))]
    assert done_calls == [("done", 2, 2)]

    # Exporters consume ExtractResult cleanly
    ass_path = tmp_path / "out.ass"
    srt_path = tmp_path / "out.srt"
    export_ass(result.events, ass_path, video_width=result.width, video_height=result.height)
    export_srt(result.events, srt_path)

    ass_text = ass_path.read_text(encoding="utf-8")
    srt_text = srt_path.read_text(encoding="utf-8")
    assert ass_text.count("Dialogue:") == 2
    assert ev1.text in ass_text and ev2.text in ass_text
    assert ev1.text in srt_text and ev2.text in srt_text
    assert "\n2\n" in srt_text


def test_pipeline_cancellation_raises_and_closes_video_reader(tmp_path: Path, monkeypatch):
    import transapp.core.pipeline as pipeline_mod
    from transapp.core.pipeline import ExtractionPipeline

    video_path = tmp_path / "cancel_test.mp4"
    _make_synthetic_subtitle_mp4(
        video_path,
        duration_sec=2.0,
        cues=((200, 900, "第一句"), (1100, 1800, "第二句")),
    )

    tracker = {"opened": 0, "closed": 0, "open_now": 0}

    class TrackedVideoReader(VideoReader):
        def __init__(self, path: Path | str) -> None:
            super().__init__(path)
            tracker["opened"] += 1
            tracker["open_now"] += 1

        def close(self) -> None:
            was_open = self._container is not None
            super().close()
            if was_open:
                tracker["closed"] += 1
                tracker["open_now"] -= 1

    monkeypatch.setattr(pipeline_mod, "VideoReader", TrackedVideoReader)

    roi = RoiRect(0.10, 0.74, 0.80, 0.20)
    config = ExtractConfig()

    # 1) Cancel mid-scan
    scan_checks = 0

    def cancel_mid_scan() -> bool:
        nonlocal scan_checks
        scan_checks += 1
        return scan_checks >= 6

    pipeline = ExtractionPipeline(
        ocr_engine=OcrEngine(ocr_callable=lambda _img: ([[[[0, 0], [20, 0], [20, 20], [0, 20]], "测试", 0.9]], [0.01]))
    )
    with pytest.raises(ExtractionCancelled):
        pipeline.run(video_path, roi, config, cancel_check=cancel_mid_scan)

    assert tracker["opened"] == 1
    assert tracker["closed"] == 1
    assert tracker["open_now"] == 0

    # 2) Cancel during OCR stage (reader must already be closed!)
    seen_stages: list[str] = []

    def progress_cb(stage: str, cur: int, tot: int) -> None:
        seen_stages.append(stage)

    ocr_cancel_trigger = {"cancel": False}

    def tracking_ocr(_img: np.ndarray):
        assert tracker["open_now"] == 0
        ocr_cancel_trigger["cancel"] = True
        return ([[[[0, 0], [20, 0], [20, 20], [0, 20]], "第一句", 0.95]], [0.01])

    pipeline_ocr = ExtractionPipeline(ocr_engine=OcrEngine(ocr_callable=tracking_ocr))
    with pytest.raises(ExtractionCancelled):
        pipeline_ocr.run(
            video_path,
            roi,
            config,
            progress_cb=progress_cb,
            cancel_check=lambda: ocr_cancel_trigger["cancel"],
        )

    assert tracker["opened"] == 2
    assert tracker["closed"] == 2
    assert tracker["open_now"] == 0
    assert "done" not in seen_stages

    # 3) Cancel immediately before opening reader
    with pytest.raises(ExtractionCancelled):
        pipeline.run(video_path, roi, config, cancel_check=lambda: True)
    assert tracker["open_now"] == 0


def test_pipeline_empty_video_produces_zero_events_and_closes_reader(tmp_path: Path):
    from transapp.core.pipeline import ExtractionPipeline

    video_path = tmp_path / "empty_subtitles.mp4"
    _make_synthetic_subtitle_mp4(video_path, duration_sec=1.0, cues=())

    ocr_calls = 0

    def counting_ocr(_img: np.ndarray):
        nonlocal ocr_calls
        ocr_calls += 1
        return (None, None)

    pipeline = ExtractionPipeline(ocr_engine=OcrEngine(ocr_callable=counting_ocr))
    stages: list[tuple[str, int, int]] = []
    result = pipeline.run(
        video_path,
        RoiRect(0.10, 0.74, 0.80, 0.20),
        ExtractConfig(),
        progress_cb=lambda s, c, t: stages.append((s, c, t)),
    )

    assert result.events == []
    assert ocr_calls == 0
    assert stages[-1] == ("done", 0, 0)


def test_pipeline_skips_ocr_for_segments_below_min_duration(tmp_path: Path):
    from transapp.core.pipeline import ExtractionPipeline

    video_path = tmp_path / "short_flash.mp4"
    # 1 frame / 40ms flash at 25fps -> raw segment duration ~40ms < min_duration_ms=120
    _make_synthetic_subtitle_mp4(video_path, duration_sec=0.6, cues=((200, 230, "闪"),))

    ocr_calls = 0

    def counting_ocr(_img: np.ndarray):
        nonlocal ocr_calls
        ocr_calls += 1
        return ([[[[0, 0], [20, 0], [20, 20], [0, 20]], "闪", 0.99]], [0.01])

    pipeline = ExtractionPipeline(ocr_engine=OcrEngine(ocr_callable=counting_ocr))
    result = pipeline.run(
        video_path,
        RoiRect(0.10, 0.74, 0.80, 0.20),
        ExtractConfig(coarse_step=1, min_duration_ms=120),
    )

    assert ocr_calls == 0
    assert result.events == []


def test_pipeline_scan_progress_normalized_for_nonzero_start_pts(tmp_path: Path):
    from transapp.core.pipeline import ExtractionPipeline

    video_path = tmp_path / "nonzero_start_pts.mp4"
    _make_synthetic_subtitle_mp4(
        video_path,
        duration_sec=0.5,
        start_pts_ms=5000,
        cues=((100, 400, "偏移测试"),),
    )

    pipeline = ExtractionPipeline(
        ocr_engine=OcrEngine(
            ocr_callable=lambda _img: (
                [[[[0, 0], [80, 0], [80, 24], [0, 24]], "偏移测试", 0.95]],
                [0.01],
            )
        )
    )
    progress_calls: list[tuple[str, int, int]] = []
    result = pipeline.run(
        video_path,
        RoiRect(0.10, 0.74, 0.80, 0.20),
        ExtractConfig(coarse_step=1),
        progress_cb=lambda stage, cur, tot: progress_calls.append((stage, cur, tot)),
    )

    scan_calls = [c for c in progress_calls if c[0] == "scan"]
    assert len(scan_calls) >= 5
    # Frame 0 must start at 0, not jump straight to total_ms even though start_pts_ms >= duration_ms.
    assert scan_calls[0][1] == 0
    assert scan_calls[-1][1] > 0
    assert all(0 <= cur <= tot for _, cur, tot in scan_calls)
    assert [cur for _, cur, _ in scan_calls] == sorted(cur for _, cur, _ in scan_calls)
    # Exported event timestamps must also be normalized relative to start_pts_ms (aligning with Preview).
    assert len(result.events) == 1
    assert 0 <= result.events[0].start_ms <= 200
    assert result.events[0].end_ms <= 600


def test_pipeline_test_frame_bgr_and_engine_lock(monkeypatch):
    import threading
    import time
    import transapp.core.pipeline as pipeline_mod
    from transapp.core.pipeline import ExtractionPipeline

    full_bgr = np.zeros((360, 640, 3), dtype=np.uint8)
    roi = RoiRect(0.10, 0.74, 0.80, 0.20)
    x1, y1, x2, y2 = roi.to_pixels(640, 360)
    full_bgr[y1 + 20 : y2 - 20, x1 + 40 : x2 - 40] = (245, 245, 245)

    seen_shapes: list[tuple[int, int, int]] = []

    def fake_ocr(img: np.ndarray):
        seen_shapes.append(img.shape)
        return ([[[[0, 0], [60, 0], [60, 20], [0, 20]], " 你 好 ", 0.88]], [0.01])

    pipeline = ExtractionPipeline(ocr_engine=OcrEngine(ocr_callable=fake_ocr))
    res = pipeline.test_frame_bgr(full_bgr, frame_index=7, pts_ms=1234, roi=roi, config=ExtractConfig())
    assert isinstance(res, FrameTestResult)
    assert res.frame_index == 7
    assert res.pts_ms == 1234
    assert res.mask.shape == (y2 - y1, x2 - x1)
    assert seen_shapes == [(y2 - y1, x2 - x1, 3)]
    assert res.text == "你好"
    assert res.confidence == pytest.approx(0.88)

    # Concurrent _resolve_ocr_engine calls must initialize OcrEngine only once under _engine_lock.
    init_calls = 0

    class _SlowFakeOcrEngine:
        def __init__(self, provider: str = "cpu", ocr_model: str = "fast"):
            nonlocal init_calls
            time.sleep(0.05)
            init_calls += 1
            self.provider = provider
            self.ocr_model = ocr_model

    monkeypatch.setattr(pipeline_mod, "OcrEngine", _SlowFakeOcrEngine)
    shared_pipeline = ExtractionPipeline()
    assert hasattr(shared_pipeline, "_engine_lock")

    barrier = threading.Barrier(4)
    resolved: list[object] = []
    lock = threading.Lock()

    def worker():
        barrier.wait()
        eng = shared_pipeline._resolve_ocr_engine(ExtractConfig(provider="cpu", ocr_model="fast"))
        with lock:
            resolved.append(eng)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert init_calls == 1
    assert len(resolved) == 4
    assert all(eng is resolved[0] for eng in resolved)

    # Changing ocr_model must rebuild cached engine
    accurate_eng = shared_pipeline._resolve_ocr_engine(
        ExtractConfig(provider="cpu", ocr_model="accurate")
    )
    assert init_calls == 2
    assert accurate_eng is not resolved[0]
    assert accurate_eng.ocr_model == "accurate"


def test_pipeline_run_clamps_event_end_ms_to_video_duration(tmp_path: Path, monkeypatch):
    import transapp.core.pipeline as pipeline_mod
    from transapp.core.models import OcrSegment
    from transapp.core.pipeline import ExtractionPipeline

    video_path = tmp_path / "clamp_duration.mp4"
    _make_synthetic_subtitle_mp4(
        video_path,
        duration_sec=1.0,
        fps=25,
        cues=((200, 1000, "结尾测试"),),
    )

    # Simulate an OCR segment whose end_ms overshoots the container duration_ms
    def fake_recognize(spans, frame_pts_ms, ocr, config, **kwargs):
        return [
            OcrSegment(
                start_ms=200,
                end_ms=5000,
                text="结尾测试",
                confidence=0.96,
                thumbnail_png=b"\x89PNG\r\n\x1a\n",
            )
        ]

    monkeypatch.setattr(pipeline_mod, "recognize_spans_with_voting", fake_recognize)

    pipeline = ExtractionPipeline(
        ocr_engine=OcrEngine(
            ocr_callable=lambda _img: (
                [[[[0, 0], [80, 0], [80, 24], [0, 24]], "结尾测试", 0.96]],
                [0.01],
            )
        )
    )
    result = pipeline.run(
        video_path,
        RoiRect(0.10, 0.74, 0.80, 0.20),
        ExtractConfig(coarse_step=1),
    )

    assert len(result.events) == 1
    assert result.duration_ms > 0
    assert result.events[0].end_ms <= result.duration_ms
    assert result.events[0].end_ms == result.duration_ms


def test_pipeline_trim_in_out_skips_op_ed_and_preserves_absolute_timestamps(tmp_path: Path):
    from transapp.core.pipeline import ExtractionPipeline

    video_path = tmp_path / "trim_op_ed.mp4"
    _make_synthetic_subtitle_mp4(
        video_path,
        width=640,
        height=360,
        fps=25,
        duration_sec=3.2,
        cues=(
            (160, 760, "片头字幕"),
            (1200, 1800, "正文台词"),
            (2320, 3000, "片尾字幕"),
        ),
    )

    roi = RoiRect(0.10, 0.74, 0.80, 0.20)
    pipeline = ExtractionPipeline()

    # Invalid trim bounds must raise ValueError before scanning
    with pytest.raises(ValueError, match="trim_start_ms"):
        pipeline.run(video_path, roi, ExtractConfig(provider="cpu", trim_start_ms=2000, trim_end_ms=2000))
    with pytest.raises(ValueError, match="trim_start_ms"):
        pipeline.run(video_path, roi, ExtractConfig(provider="cpu", trim_start_ms=2400, trim_end_ms=1000))
    with pytest.raises(ValueError, match="trim_start_ms"):
        pipeline.run(video_path, roi, ExtractConfig(provider="cpu", trim_start_ms=None, trim_end_ms=0))

    # Case 1: Trim [900ms, 2000ms] skips OP & ED, keeps ONLY middle cue with absolute ~1200ms start
    progress_calls: list[tuple[str, int, int]] = []
    result = pipeline.run(
        video_path,
        roi,
        ExtractConfig(provider="cpu", coarse_step=1, trim_start_ms=900, trim_end_ms=2000),
        progress_cb=lambda stage, cur, tot: progress_calls.append((stage, cur, tot)),
    )

    assert len(result.events) == 1
    ev = result.events[0]
    assert "正文" in ev.text or "台词" in ev.text
    assert 1120 <= ev.start_ms <= 1280  # absolute video-relative ms, NOT shifted to ~300ms
    assert 1720 <= ev.end_ms <= 1920
    assert result.frame_pts_ms[0] >= 900
    assert result.frame_pts_ms[-1] <= 2000

    scan_calls = [c for c in progress_calls if c[0] == "scan"]
    assert len(scan_calls) >= 2
    assert all(tot == 1100 and 0 <= cur <= 1100 for _, cur, tot in scan_calls)
    assert scan_calls[0][1] <= 80
    assert scan_calls[-1][1] >= 1000

    # Case 2: Cue still active at trim_end_ms=1600 must be clamped so end_ms <= 1600
    clamped_result = pipeline.run(
        video_path,
        roi,
        ExtractConfig(provider="cpu", coarse_step=1, trim_start_ms=900, trim_end_ms=1600),
    )
    assert len(clamped_result.events) == 1
    clamped_ev = clamped_result.events[0]
    assert 1120 <= clamped_ev.start_ms <= 1280
    assert clamped_ev.end_ms <= 1600
    assert clamped_ev.end_ms >= 1520

    # Case 3: trim_end_ms > duration_ms still clamps scan progress total to duration_ms (~100%)
    overshoot_progress: list[tuple[str, int, int]] = []
    overshoot_result = pipeline.run(
        video_path,
        roi,
        ExtractConfig(provider="cpu", coarse_step=1, trim_start_ms=900, trim_end_ms=99_999),
        progress_cb=lambda stage, cur, tot: overshoot_progress.append((stage, cur, tot)),
    )
    overshoot_scans = [c for c in overshoot_progress if c[0] == "scan"]
    assert len(overshoot_scans) >= 2
    expected_span = max(1, overshoot_result.duration_ms - 900)
    assert all(tot == expected_span and 0 <= cur <= expected_span for _, cur, tot in overshoot_scans)
    assert overshoot_scans[-1][1] >= expected_span - 80


def test_test_single_frame_uses_zero_based_rel_ms_on_nonzero_start_pts(tmp_path: Path, monkeypatch):
    import transapp.core.pipeline as pipeline_mod
    from transapp.core.pipeline import ExtractionPipeline

    video_path = tmp_path / "nonzero_single_frame.mp4"
    _make_synthetic_subtitle_mp4(
        video_path,
        duration_sec=1.5,
        start_pts_ms=5000,
        cues=((200, 1200, "偏移单帧"),),
    )

    seen_targets: list[int] = []

    class SpyVideoReader(VideoReader):
        def read_frame_at_ms(self, target_ms: int):
            seen_targets.append(target_ms)
            return super().read_frame_at_ms(target_ms)

    monkeypatch.setattr(pipeline_mod, "VideoReader", SpyVideoReader)

    pipeline = ExtractionPipeline(
        ocr_engine=OcrEngine(
            ocr_callable=lambda _img: (
                [[[[0, 0], [80, 0], [80, 24], [0, 24]], "偏移单帧", 0.94]],
                [0.01],
            )
        )
    )
    roi = RoiRect(0.10, 0.74, 0.80, 0.20)
    res = pipeline.test_single_frame(video_path, 600, roi, ExtractConfig())

    assert len(seen_targets) == 1
    assert seen_targets[0] == 5600  # start_pts_ms (5000) + rel frame_ms (600)
    assert abs(res.pts_ms - 600) <= 60  # returned FrameTestResult.pts_ms is 0-based rel_ms
    assert mask_fg_ratio(res.mask) > ExtractConfig().fg_ratio_min
    assert res.text == "偏移单帧"


def test_pipeline_rejects_trim_start_at_or_beyond_duration_and_preserves_tail_over_short_header(
    tmp_path: Path, monkeypatch
):
    from dataclasses import replace as dc_replace
    import transapp.core.pipeline as pipeline_mod
    from transapp.core.models import OcrSegment
    from transapp.core.pipeline import ExtractionPipeline

    video_path = tmp_path / "trim_and_tail.mp4"
    _make_synthetic_subtitle_mp4(
        video_path,
        duration_sec=1.0,
        fps=25,
        cues=((200, 1000, "尾帧测试"),),
    )
    roi = RoiRect(0.10, 0.74, 0.80, 0.20)

    # 1. Missing video path must NOT evict _cached_ocr_engine even when config changes provider/model
    cached_pipeline = ExtractionPipeline()
    sentinel_engine = OcrEngine(provider="cpu", ocr_model="fast")
    cached_pipeline._cached_ocr_engine = sentinel_engine
    with pytest.raises(Exception):
        cached_pipeline.run(
            tmp_path / "does_not_exist.mp4",
            roi,
            ExtractConfig(provider="directml", ocr_model="accurate"),
        )
    assert cached_pipeline._cached_ocr_engine is sentinel_engine

    # 2. trim_start_ms >= meta.duration_ms must raise ValueError (without evicting cached engine)
    with VideoReader(video_path) as probe_reader:
        actual_duration_ms = probe_reader.metadata.duration_ms
    assert actual_duration_ms > 0

    with pytest.raises(ValueError, match="trim_start_ms"):
        cached_pipeline.run(
            video_path,
            roi,
            ExtractConfig(provider="directml", ocr_model="accurate", trim_start_ms=actual_duration_ms),
        )
    assert cached_pipeline._cached_ocr_engine is sentinel_engine

    with pytest.raises(ValueError, match="trim_start_ms"):
        cached_pipeline.run(
            video_path,
            roi,
            ExtractConfig(provider="cpu", trim_start_ms=actual_duration_ms + 500),
        )

    # 3. Under-reported container header duration_ms must not clip valid tail cue when trim_end is None
    class ShortHeaderVideoReader(VideoReader):
        @property
        def metadata(self):
            base = super().metadata
            # Under-report header duration by 2 frames (80ms) relative to real 1000ms video
            return dc_replace(base, duration_ms=max(100, base.duration_ms - 80))

    monkeypatch.setattr(pipeline_mod, "VideoReader", ShortHeaderVideoReader)

    def fake_recognize(spans, frame_pts_ms, ocr, config, **kwargs):
        tail_end = frame_pts_ms[-1] + 40
        return [
            OcrSegment(
                start_ms=200,
                end_ms=tail_end,
                text="尾帧测试",
                confidence=0.96,
                thumbnail_png=b"\x89PNG\r\n\x1a\n",
            )
        ]

    monkeypatch.setattr(pipeline_mod, "recognize_spans_with_voting", fake_recognize)

    tail_pipeline = ExtractionPipeline(
        ocr_engine=OcrEngine(
            ocr_callable=lambda _img: (
                [[[[0, 0], [80, 0], [80, 24], [0, 24]], "尾帧测试", 0.96]],
                [0.01],
            )
        )
    )
    result = tail_pipeline.run(video_path, roi, ExtractConfig(coarse_step=1))
    assert len(result.events) == 1
    expected_tail = result.frame_pts_ms[-1] + 40
    assert result.duration_ms < expected_tail  # header genuinely under-reported
    assert result.events[0].end_ms == expected_tail  # tail cue preserved intact




