from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import threading
import time

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QPixmap

from transapp.core.models import ExtractResult, RoiRect, SubtitleEvent
from transapp.core.ocr_engine import OcrEngine
from transapp.core.pipeline import ExtractionPipeline
from transapp.core.presets import PresetStore, RoiPreset
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
    duration_sec: float = 1.6,
    start_pts_ms: int = 0,
    cues: tuple[tuple[int, int, str], ...] = ((200, 1200, "你好世界"),),
) -> None:
    total_frames = int(round(duration_sec * fps))
    font = _load_cjk_font(size=32)

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)

        for idx in range(total_frames):
            pts_ms = int(round(idx * 1000.0 / fps))
            img = Image.new("RGB", (width, height), (35, 70, 120))
            draw = ImageDraw.Draw(img)
            active_text = next(
                (text for start_ms, end_ms, text in cues if start_ms <= pts_ms < end_ms),
                None,
            )
            if active_text is not None:
                draw.text(
                    (width // 2, int(height * 0.84)),
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


def _make_png_bytes(color_bgr: tuple[int, int, int] = (240, 240, 240)) -> bytes:
    img = np.full((32, 120, 3), color_bgr, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_roi_canvas_drag_normalizes_clamps_and_handles_set_roi(qtbot):
    from transapp.ui.roi_canvas import RoiCanvas

    canvas = RoiCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(640, 360)
    canvas.show()

    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[280:330, 100:540] = (200, 200, 200)
    canvas.set_frame(frame)

    # set_roi clamps out-of-bounds values cleanly
    canvas.set_roi(RoiRect(-0.2, 0.75, 1.5, 0.5))
    clamped = canvas.roi()
    assert clamped.x == pytest.approx(0.0)
    assert clamped.y == pytest.approx(0.75)
    assert clamped.w == pytest.approx(1.0)
    assert clamped.h == pytest.approx(0.25)

    # Set ROI to bottom region so top-area drag draws a fresh ROI
    canvas.set_roi(RoiRect(0.10, 0.80, 0.80, 0.15))

    emitted: list[RoiRect] = []
    canvas.roi_changed.connect(emitted.append)

    start_pos = QPoint(64, 36)   # (0.10, 0.10) outside current bottom ROI
    end_pos = QPoint(720, 180)   # past right edge -> x+w clamps to 1.0, y=0.50
    qtbot.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start_pos)
    qtbot.mouseMove(canvas, pos=end_pos)
    qtbot.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=end_pos)

    dragged = canvas.roi()
    assert 0.0 <= dragged.x <= 1.0
    assert 0.0 <= dragged.y <= 1.0
    assert dragged.x + dragged.w <= 1.0 + 1e-6
    assert dragged.y + dragged.h <= 1.0 + 1e-6
    assert dragged.x == pytest.approx(0.10, abs=0.02)
    assert dragged.y == pytest.approx(0.10, abs=0.02)
    assert dragged.w == pytest.approx(0.90, abs=0.02)
    assert dragged.h == pytest.approx(0.40, abs=0.02)
    assert len(emitted) >= 1
    assert emitted[-1] == dragged

    # Mask overlay inside ROI renders cleanly without error
    x1, y1, x2, y2 = dragged.to_pixels(640, 360)
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    mask[:, :10] = 255
    canvas.set_frame(frame, mask=mask)
    canvas.repaint()


def test_review_table_filtering_thumbnail_and_inline_edits(qtbot, tmp_path: Path):
    from transapp.ui.review_table import ReviewTable

    table = ReviewTable()
    qtbot.addWidget(table)
    table.resize(900, 420)
    table.show()

    ev1 = SubtitleEvent(
        index=1,
        start_ms=500,
        end_ms=1200,
        text="第一句",
        confidence=0.96,
        flagged=False,
        thumbnail_png=_make_png_bytes((40, 200, 40)),
    )
    ev2 = SubtitleEvent(
        index=2,
        start_ms=1500,
        end_ms=2600,
        text="第二句误识",
        confidence=0.72,
        flagged=True,
        thumbnail_png=_make_png_bytes((40, 40, 220)),
    )

    table.set_events([ev1, ev2])
    assert table.table.rowCount() == 2

    # Filtering to flagged_only hides normal rows
    table.set_filter_mode("flagged_only")
    assert table.table.rowCount() == 1
    assert table.table.item(0, 4).text() == "第二句误识"

    # Selecting visible flagged row renders thumbnail PNG from memory
    table.table.selectRow(0)
    pixmap = table.preview_label.pixmap()
    assert isinstance(pixmap, QPixmap)
    assert not pixmap.isNull()

    # Inline edit text, valid SRT/ASS timestamps, and reject invalid timestamp
    table.table.item(0, 4).setText("第二句已修正")
    table.table.item(0, 1).setText("00:00:01,650")  # SRT format -> 1650 ms
    table.table.item(0, 2).setText("0:00:02.85")    # ASS format -> 2850 ms

    # Invalid time input must revert cleanly without crashing or corrupting state
    before_start = table.table.item(0, 1).text()
    table.table.item(0, 1).setText("99:99:99.99")
    assert table.table.item(0, 1).text() == before_start

    updated_events = table.events()
    assert len(updated_events) == 2
    assert updated_events[0].text == "第一句"
    assert updated_events[1].text == "第二句已修正"
    assert updated_events[1].start_ms == 1650
    assert updated_events[1].end_ms == 2850

    # Exporting updated events reflects edits in ASS and SRT
    ass_path = tmp_path / "edited.ass"
    srt_path = tmp_path / "edited.srt"
    export_ass(updated_events, ass_path, video_width=640, video_height=360)
    export_srt(updated_events, srt_path)
    assert "第二句已修正" in ass_path.read_text(encoding="utf-8")
    assert "00:00:01,650 --> 00:00:02,850\n第二句已修正" in srt_path.read_text(encoding="utf-8")

    # Clearing removes events and thumbnail pixmap from memory
    table.clear()
    assert table.events() == []
    assert table.table.rowCount() == 0
    cleared_pm = table.preview_label.pixmap()
    assert cleared_pm is None or cleared_pm.isNull()


def test_main_window_overwrite_guard_and_video_switch_clears_memory(qtbot, tmp_path: Path):
    from transapp.ui.main_window import MainWindow

    ep1 = tmp_path / "ep01_测试.mp4"
    ep2 = tmp_path / "ep02_测试.mp4"
    _make_synthetic_subtitle_mp4(ep1, duration_sec=1.5, cues=((200, 1200, "你好世界"),))
    _make_synthetic_subtitle_mp4(ep2, duration_sec=1.0, cues=((200, 800, "下集预告"),))

    existing_ass = tmp_path / "ep01_测试.ass"
    existing_srt = tmp_path / "ep01_测试.srt"
    existing_ass.write_text("ORIGINAL ASS\n", encoding="utf-8")
    existing_srt.write_text("ORIGINAL SRT\n", encoding="utf-8")

    fake_ocr = OcrEngine(
        ocr_callable=lambda _img: (
            [[[[0, 0], [80, 0], [80, 24], [0, 24]], "你好世界", 0.93]],
            [0.01],
        )
    )
    pipeline = ExtractionPipeline(ocr_engine=fake_ocr)
    store = PresetStore(config_dir=tmp_path / "config")

    prompt_calls: list[list[Path]] = []
    confirm_answer = {"allow": False}

    def fake_confirm(paths: list[Path]) -> bool:
        prompt_calls.append(list(paths))
        return confirm_answer["allow"]

    window = MainWindow(
        preset_store=store,
        pipeline=pipeline,
        overwrite_confirm_fn=fake_confirm,
    )
    qtbot.addWidget(window)
    window.show()

    window.load_video(ep1)
    window.roi_canvas.set_roi(RoiRect(0.10, 0.72, 0.80, 0.22))

    # Run extraction via QThread worker
    window.btn_start.click()
    qtbot.waitUntil(lambda: len(window.review_table.events()) > 0, timeout=10000)
    qtbot.waitUntil(lambda: window.btn_export.isEnabled(), timeout=5000)

    # First export attempt: guard declines overwrite -> files stay untouched
    confirm_answer["allow"] = False
    exported = window.export_subtitles()
    assert exported is False
    assert len(prompt_calls) == 1
    assert set(prompt_calls[0]) == {existing_ass, existing_srt}
    assert existing_ass.read_text(encoding="utf-8") == "ORIGINAL ASS\n"
    assert existing_srt.read_text(encoding="utf-8") == "ORIGINAL SRT\n"

    # Second export attempt: guard allows overwrite -> files updated
    confirm_answer["allow"] = True
    exported = window.export_subtitles()
    assert exported is True
    assert len(prompt_calls) == 2
    assert "Dialogue:" in existing_ass.read_text(encoding="utf-8")
    assert "你好世界" in existing_srt.read_text(encoding="utf-8")

    # Loading another video immediately clears old events and result from memory
    window.load_video(ep2)
    assert window.review_table.events() == []
    assert window.review_table.table.rowCount() == 0
    assert window.current_result is None
    window.close()


def test_main_window_presets_test_frame_and_worker_cancel(qtbot, tmp_path: Path):
    from transapp.ui.main_window import MainWindow

    video_path = tmp_path / "cancel_ep.mp4"
    _make_synthetic_subtitle_mp4(
        video_path,
        duration_sec=2.0,
        cues=((200, 900, "第一句"), (1100, 1800, "第二句")),
    )

    ocr_started = threading.Event()
    release_ocr = threading.Event()

    def slow_ocr(_img: np.ndarray):
        ocr_started.set()
        release_ocr.wait(timeout=3.0)
        return ([[[[0, 0], [80, 0], [80, 24], [0, 24]], "第一句", 0.91]], [0.01])

    pipeline = ExtractionPipeline(ocr_engine=OcrEngine(ocr_callable=slow_ocr))
    store = PresetStore(config_dir=tmp_path / "cfg")
    store.save(
        RoiPreset(
            name="Donghua_Test",
            roi=RoiRect(0.10, 0.72, 0.80, 0.22),
            conf_threshold=0.80,
        )
    )

    window = MainWindow(preset_store=store, pipeline=pipeline)
    qtbot.addWidget(window)
    window.show()
    window.load_video(video_path)

    # Apply saved preset
    idx = window.preset_combo.findText("Donghua_Test")
    assert idx >= 0
    window.preset_combo.setCurrentIndex(idx)
    assert window.roi_canvas.roi().y == pytest.approx(0.72)
    assert window.spin_conf.value() == pytest.approx(0.80)

    # Test 1 frame updates diagnostic label cleanly
    window.seek_to_ms(500)
    release_ocr.set()
    window.btn_test_frame.click()
    assert "第一句" in window.test_result_label.text()

    # Reset release flag so extraction pauses inside OCR for cancel verification
    release_ocr.clear()
    ocr_started.clear()
    window.btn_start.click()
    assert ocr_started.wait(timeout=5.0)
    window.btn_stop.click()
    release_ocr.set()

    qtbot.waitUntil(lambda: window.btn_start.isEnabled(), timeout=5000)
    assert window.current_result is None
    assert window.review_table.events() == []
    window.close()


def test_roi_canvas_letterbox_move_and_handle_resize(qtbot):
    from transapp.ui.roi_canvas import RoiCanvas

    canvas = RoiCanvas()
    qtbot.addWidget(canvas)
    # 800x360 widget displaying a 640x360 frame -> pillarboxed at x=[80..720], w=640, h=360
    canvas.resize(800, 360)
    canvas.show()
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    canvas.set_frame(frame)

    canvas.set_roi(RoiRect(0.25, 0.25, 0.50, 0.50))
    # Inside ROI in widget coords: x = 80 + 0.5*640 = 400, y = 0.5*360 = 180
    qtbot.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(400, 180))
    qtbot.mouseMove(canvas, pos=QPoint(464, 216))  # +0.10 x, +0.10 y
    qtbot.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(464, 216))

    moved = canvas.roi()
    assert moved.x == pytest.approx(0.35, abs=0.02)
    assert moved.y == pytest.approx(0.35, abs=0.02)
    assert moved.w == pytest.approx(0.50, abs=0.01)
    assert moved.h == pytest.approx(0.50, abs=0.01)

    # Drag bottom-right corner handle (se) beyond frame edge -> clamped at 1.0
    se_x = round(80 + (moved.x + moved.w) * 640)
    se_y = round((moved.y + moved.h) * 360)
    qtbot.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(se_x, se_y))
    qtbot.mouseMove(canvas, pos=QPoint(795, 355))
    qtbot.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(795, 355))

    resized = canvas.roi()
    assert resized.x == pytest.approx(moved.x, abs=0.02)
    assert resized.x + resized.w == pytest.approx(1.0, abs=0.01)
    assert resized.y + resized.h == pytest.approx(1.0, abs=0.02)


def test_review_table_readonly_cols_and_switch_back_to_all(qtbot):
    from transapp.ui.review_table import ReviewTable

    table = ReviewTable()
    qtbot.addWidget(table)
    ev1 = SubtitleEvent(1, 100, 500, "普通", 0.95, False, _make_png_bytes())
    ev2 = SubtitleEvent(2, 600, 1200, "可疑", 0.70, True, _make_png_bytes())
    table.set_events([ev1, ev2])

    assert not (table.table.item(0, 0).flags() & Qt.ItemFlag.ItemIsEditable)
    assert not (table.table.item(0, 3).flags() & Qt.ItemFlag.ItemIsEditable)

    table.set_filter_mode("flagged_only")
    table.table.item(0, 4).setText("已修正")
    table.set_filter_mode("all")
    assert table.table.rowCount() == 2
    assert table.table.item(1, 4).text() == "已修正"


def test_main_window_custom_dir_ass_template_presets_scrubber_and_close_while_running(
    qtbot, tmp_path: Path, monkeypatch
):
    from PySide6.QtWidgets import QApplication
    import transapp.app as app_mod
    from transapp.ui.main_window import MainWindow

    video_path = tmp_path / "ep_tpl.mp4"
    _make_synthetic_subtitle_mp4(video_path, duration_sec=1.2, cues=((200, 1000, "模板测试"),))

    tpl_path = tmp_path / "style_template.ass"
    tpl_path.write_text(
        "\ufeff[Script Info]\r\nTitle: CustomStyle\r\nPlayResX: 1920\r\nPlayResY: 1080\r\n\r\n"
        "[V4+ Styles]\r\nFormat: Name, Fontname, Fontsize\r\n"
        "Style: AnimeFont,Microsoft YaHei,54\r\n\r\n[Events]\r\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "exported_subs"

    ocr_entered = threading.Event()
    release_ocr = threading.Event()

    def controllable_ocr(_img: np.ndarray):
        ocr_entered.set()
        release_ocr.wait(timeout=3.0)
        return ([[[[0, 0], [80, 0], [80, 24], [0, 24]], "模板测试", 0.94]], [0.01])

    pipeline = ExtractionPipeline(ocr_engine=OcrEngine(ocr_callable=controllable_ocr))
    store = PresetStore(config_dir=tmp_path / "preset_cfg")
    window = MainWindow(preset_store=store, pipeline=pipeline)
    qtbot.addWidget(window)
    window.show()
    window.load_video(video_path)

    # Scrubber buttons advance and rewind frames cleanly
    window.btn_step_fwd.click()
    assert window._current_frame_ms > 0
    window.btn_seek_start.click()
    assert window._current_frame_ms == 0
    window.btn_seek_end.click()
    assert window._current_frame_ms > 0

    # Save & delete ROI preset
    window.roi_canvas.set_roi(RoiRect(0.15, 0.70, 0.70, 0.20))
    window.radio_ass_template.setChecked(True)
    window.ass_template_edit.setText(str(tpl_path))
    assert window.save_preset("MyAnime") is True
    assert "MyAnime" in store.list_names()
    assert window.delete_current_preset() is True
    assert "MyAnime" not in store.list_names()

    # Extract and verify preview reader is released before QThread starts
    release_ocr.set()
    window.btn_start.click()
    assert window._preview_reader is None
    qtbot.waitUntil(lambda: window.btn_export.isEnabled(), timeout=8000)

    # Export to custom directory with ASS template preserved
    window.radio_dir_custom.setChecked(True)
    window.custom_dir_edit.setText(str(out_dir))
    window.radio_ass_template.setChecked(True)
    window.ass_template_edit.setText(str(tpl_path))
    assert window.export_subtitles() is True

    ass_out = (out_dir / "ep_tpl.ass").read_text(encoding="utf-8")
    srt_out = (out_dir / "ep_tpl.srt").read_text(encoding="utf-8")
    assert "Style: AnimeFont,Microsoft YaHei,54" in ass_out
    assert ",AnimeFont,,0,0,0,,模板测试" in ass_out
    assert "模板测试" in srt_out

    # Closing window while worker is running cancels and shuts down QThread cleanly
    release_ocr.clear()
    ocr_entered.clear()
    window.btn_start.click()
    assert ocr_entered.wait(timeout=5.0)
    release_ocr.set()
    window.close()
    assert window._worker_thread is None
    assert window._preview_reader is None
    assert window.current_result is None

    # Smoke-test transapp.app.main() without blocking on event loop (isolated LOCALAPPDATA)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "isolated_appdata"))
    monkeypatch.setattr(QApplication, "exec", lambda _self: 0)
    before_widgets = set(QApplication.topLevelWidgets())
    assert app_mod.main() == 0
    created_windows = [
        w
        for w in QApplication.topLevelWidgets()
        if w not in before_widgets and isinstance(w, MainWindow)
    ]
    assert len(created_windows) == 1
    assert created_windows[0]._preset_store.config_dir == tmp_path / "isolated_appdata" / "TransApp"
    created_windows[0].close()


def test_main_window_load_video_during_active_extraction_stops_fast_and_ignores_stale_worker(
    qtbot, tmp_path: Path, monkeypatch
):
    from PySide6.QtWidgets import QMessageBox
    from transapp.ui.main_window import MainWindow, _ExtractionWorker

    ep1 = tmp_path / "active_ep1.mp4"
    ep2 = tmp_path / "active_ep2.mp4"
    _make_synthetic_subtitle_mp4(ep1, duration_sec=1.2, cues=((200, 900, "第一集"),))
    _make_synthetic_subtitle_mp4(ep2, duration_sec=1.0, cues=((200, 800, "第二集"),))

    ocr_entered = threading.Event()

    def quick_cancel_ocr(_img: np.ndarray):
        ocr_entered.set()
        time.sleep(0.05)
        return ([[[[0, 0], [80, 0], [80, 24], [0, 24]], "第一集", 0.90]], [0.01])

    pipeline = ExtractionPipeline(ocr_engine=OcrEngine(ocr_callable=quick_cancel_ocr))
    window = MainWindow(preset_store=PresetStore(config_dir=tmp_path / "cfg"), pipeline=pipeline)
    qtbot.addWidget(window)
    window.show()
    window.load_video(ep1)

    window.btn_start.click()
    assert ocr_entered.wait(timeout=5.0)

    # Switching video while worker is active must quit & join QThread quickly (< 2s, never 5s stall)
    t0 = time.perf_counter()
    window.load_video(ep2)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0
    assert window._worker_thread is None
    assert window._worker is None
    assert window.current_result is None
    assert window.review_table.events() == []
    assert window.status_label.text() == "Sẵn sàng"

    # Queued/stale signals from an old worker must be ignored when sender is not self._worker
    critical_calls: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, _title, text: critical_calls.append(text),
    )
    stale_worker = _ExtractionWorker(
        pipeline=pipeline,
        video_path=ep1,
        roi=RoiRect(0.1, 0.7, 0.8, 0.2),
        config=window.build_config(),
        cancel_event=threading.Event(),
    )
    stale_worker.progress.connect(window._on_worker_progress)
    stale_worker.finished.connect(window._on_worker_finished)
    stale_worker.cancelled.connect(window._on_worker_cancelled)
    stale_worker.failed.connect(window._on_worker_failed)

    stale_result = ExtractResult(
        video_path=ep1,
        width=640,
        height=360,
        duration_ms=1000,
        events=[SubtitleEvent(1, 100, 500, "STALE", 0.99, False, _make_png_bytes())],
        elapsed_sec=0.1,
    )
    stale_worker.progress.emit("ocr", 1, 1)
    stale_worker.finished.emit(stale_result)
    stale_worker.cancelled.emit()
    stale_worker.failed.emit("stale error")

    assert window.current_result is None
    assert window.review_table.events() == []
    assert window.status_label.text() == "Sẵn sàng"
    assert critical_calls == []
    window.close()


def test_main_window_export_oserror_shows_warning_without_crashing(
    qtbot, tmp_path: Path, monkeypatch
):
    from PySide6.QtWidgets import QMessageBox
    import transapp.ui.main_window as mw_mod
    from transapp.ui.main_window import MainWindow

    video_path = tmp_path / "locked_ep.mp4"
    _make_synthetic_subtitle_mp4(video_path, duration_sec=1.2, cues=((200, 900, "锁定测试"),))

    pipeline = ExtractionPipeline(
        ocr_engine=OcrEngine(
            ocr_callable=lambda _img: (
                [[[[0, 0], [80, 0], [80, 24], [0, 24]], "锁定测试", 0.95]],
                [0.01],
            )
        )
    )
    window = MainWindow(preset_store=PresetStore(config_dir=tmp_path / "cfg"), pipeline=pipeline)
    qtbot.addWidget(window)
    window.show()
    window.load_video(video_path)

    window.btn_start.click()
    qtbot.waitUntil(lambda: window.btn_export.isEnabled(), timeout=8000)

    def raise_locked(*_args, **_kwargs):
        raise PermissionError("Locked by another process")

    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(mw_mod, "export_srt", raise_locked)
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, title, text: warnings.append((title, text)),
    )

    assert window.export_subtitles() is False
    assert len(warnings) == 1
    assert "Locked by another process" in warnings[0][1]
    assert "Locked by another process" in window.status_label.text()
    window.close()


def test_review_table_rejects_inverted_start_end_and_rescales_preview_on_resize(qtbot):
    from transapp.ui.review_table import ReviewTable

    table = ReviewTable()
    qtbot.addWidget(table)
    table.resize(620, 320)
    table.show()

    ev = SubtitleEvent(
        index=1,
        start_ms=1000,
        end_ms=2000,
        text="台词",
        confidence=0.95,
        flagged=False,
        thumbnail_png=_make_png_bytes((20, 180, 220)),
    )
    table.set_events([ev])
    qtbot.waitUntil(
        lambda: table.preview_label.pixmap() is not None and not table.preview_label.pixmap().isNull(),
        timeout=2000,
    )
    initial_w = table.preview_label.pixmap().width()

    emitted_count = 0
    table.events_changed.connect(lambda: nonlocal_inc())

    def nonlocal_inc():
        nonlocal emitted_count
        emitted_count += 1

    # Start >= End must revert cleanly without corrupting event or emitting events_changed
    orig_start_text = table.table.item(0, 1).text()
    table.table.item(0, 1).setText("00:00:02,000")
    assert table.table.item(0, 1).text() == orig_start_text
    assert table.events()[0].start_ms == 1000
    table.table.item(0, 1).setText("00:00:02,500")
    assert table.table.item(0, 1).text() == orig_start_text
    assert table.events()[0].start_ms == 1000

    # End <= Start must revert cleanly as well
    orig_end_text = table.table.item(0, 2).text()
    table.table.item(0, 2).setText("00:00:01,000")
    assert table.table.item(0, 2).text() == orig_end_text
    assert table.events()[0].end_ms == 2000
    table.table.item(0, 2).setText("00:00:00,500")
    assert table.table.item(0, 2).text() == orig_end_text
    assert table.events()[0].end_ms == 2000
    assert emitted_count == 0

    # Resizing ReviewTable rescales _raw_pixmap smoothly to the new preview_label size
    table.resize(1200, 620)
    qtbot.waitUntil(
        lambda: table.preview_label.pixmap() is not None
        and table.preview_label.pixmap().width() > initial_w,
        timeout=2000,
    )


def test_main_window_nonzero_start_pts_scrubbing_slider_tracking_and_single_decode(
    qtbot, tmp_path: Path, monkeypatch
):
    from transapp.core.video_reader import VideoReader
    from transapp.ui.main_window import MainWindow

    video_path = tmp_path / "nonzero_start.mp4"
    _make_synthetic_subtitle_mp4(
        video_path,
        duration_sec=0.4,
        fps=25,
        start_pts_ms=5000,
        cues=((40, 360, "偏移测试"),),
    )

    pipeline = ExtractionPipeline(
        ocr_engine=OcrEngine(
            ocr_callable=lambda _img: (
                [[[[0, 0], [80, 0], [80, 24], [0, 24]], "偏移测试", 0.94]],
                [0.01],
            )
        )
    )
    window = MainWindow(preset_store=PresetStore(config_dir=tmp_path / "cfg"), pipeline=pipeline)
    qtbot.addWidget(window)
    window.show()
    window.load_video(video_path)

    assert window._start_pts_ms == 5000
    assert window._duration_ms > 0
    assert window.frame_slider.hasTracking() is False
    assert window.frame_slider.minimum() == 0
    assert window.frame_slider.maximum() == window._duration_ms
    assert window._current_frame_ms == 5000
    assert window.time_label.text().startswith("0:00:00.00 / ")

    # Spy on VideoReader.read_frame_at_ms, iter_full_frames_from_ms, and VideoReader.__init__
    read_calls: list[int] = []
    iter_calls: list[int] = []
    init_calls: list[Path] = []
    real_read = VideoReader.read_frame_at_ms
    real_iter = VideoReader.iter_full_frames_from_ms
    real_init = VideoReader.__init__

    def spy_read(self, target_ms: int):
        read_calls.append(target_ms)
        return real_read(self, target_ms)

    def spy_iter(self, start_raw_ms: int):
        iter_calls.append(start_raw_ms)
        return real_iter(self, start_raw_ms)

    def spy_init(self, path):
        init_calls.append(Path(path))
        return real_init(self, path)

    monkeypatch.setattr(VideoReader, "read_frame_at_ms", spy_read)
    monkeypatch.setattr(VideoReader, "iter_full_frames_from_ms", spy_iter)
    monkeypatch.setattr(VideoReader, "__init__", spy_init)

    # sliderMoved updates time_label live WITHOUT decoding frames
    window.frame_slider.sliderMoved.emit(120)
    assert read_calls == []
    assert window.time_label.text().startswith("0:00:00.12 / ")

    # valueChanged (commit/release) decodes ONCE at start_pts_ms + offset
    window.frame_slider.setValue(120)
    assert read_calls == [5120]
    assert window._current_frame_ms == 5120
    assert window.time_label.text().startswith("0:00:00.12 / ")

    # +1f, >|, |< work relative to [start_pts_ms, start_pts_ms + duration_ms]
    read_calls.clear()
    iter_calls.clear()
    window.btn_step_fwd.click()
    assert window._current_frame_ms > 5120
    assert len(read_calls) + len(iter_calls) == 1
    # Consecutive +1f reuses sequential stream without re-seeking
    prev_ms = window._current_frame_ms
    window.btn_step_fwd.click()
    assert window._current_frame_ms > prev_ms
    assert len(read_calls) + len(iter_calls) == 1

    window.btn_seek_end.click()
    assert window._current_frame_ms >= 5160
    assert window.frame_slider.value() > 0

    window.btn_seek_start.click()
    assert window._current_frame_ms == 5000
    assert window.frame_slider.value() == 0

    # Clamping outside [start_pts_ms, start_pts_ms + duration_ms]
    read_calls.clear()
    window.seek_to_ms(0)
    assert read_calls[-1] == 5000
    window.seek_to_ms(999_999)
    assert read_calls[-1] == 5000 + window._duration_ms

    # test_current_frame uses cached frame / at most 1 decode without double VideoReader open
    init_calls.clear()
    read_calls.clear()
    res = window.test_current_frame()
    assert res is not None
    assert "偏移测试" in window.test_result_label.text()
    assert len(init_calls) == 0
    assert len(read_calls) == 0

    # Even if preview reader & cached frame are cleared, test_current_frame opens/reads at most once
    window._close_preview_reader()
    window._current_frame_bgr = None
    res2 = window.test_current_frame()
    assert res2 is not None
    assert len(init_calls) == 1
    assert len(read_calls) == 1

    # seek_to_ms surfaces missing video file error inside try block without crashing
    window._close_preview_reader()
    video_path.unlink()
    window.seek_to_ms(5040)
    assert "Lỗi seek frame:" in window.status_label.text()
    window.close()


def test_main_window_retires_slow_qthread_and_waits_on_close(
    qtbot, tmp_path: Path, monkeypatch
):
    from transapp.ui.main_window import MainWindow

    ep1 = tmp_path / "slow_ep1.mp4"
    ep2 = tmp_path / "slow_ep2.mp4"
    _make_synthetic_subtitle_mp4(ep1, duration_sec=1.0, cues=((200, 800, "第一集"),))
    _make_synthetic_subtitle_mp4(ep2, duration_sec=1.0, cues=((200, 800, "第二集"),))

    ocr_entered = threading.Event()
    release_ocr = threading.Event()

    def slow_ocr(_img: np.ndarray):
        ocr_entered.set()
        release_ocr.wait(timeout=5.0)
        return ([[[[0, 0], [80, 0], [80, 24], [0, 24]], "第一集", 0.92]], [0.01])

    pipeline = ExtractionPipeline(ocr_engine=OcrEngine(ocr_callable=slow_ocr))
    window = MainWindow(preset_store=PresetStore(config_dir=tmp_path / "cfg"), pipeline=pipeline)
    qtbot.addWidget(window)
    window.show()
    window.load_video(ep1)

    window.btn_start.click()
    assert ocr_entered.wait(timeout=5.0)

    old_thread = window._worker_thread
    old_worker = window._worker
    assert old_thread is not None and old_worker is not None

    delete_events: list[tuple[str, bool]] = []
    orig_thread_delete = old_thread.deleteLater
    orig_worker_delete = old_worker.deleteLater
    orig_wait = old_thread.wait

    monkeypatch.setattr(
        old_thread,
        "deleteLater",
        lambda: (delete_events.append(("thread", old_thread.isRunning())), orig_thread_delete())[1],
    )
    monkeypatch.setattr(
        old_worker,
        "deleteLater",
        lambda: (delete_events.append(("worker", old_thread.isRunning())), orig_worker_delete())[1],
    )
    monkeypatch.setattr(
        old_thread,
        "wait",
        lambda *args: False if args == (5000,) else orig_wait(*args),
    )

    # Switching video when wait(5000) times out must NOT call deleteLater on running thread
    window.load_video(ep2)
    assert window._worker_thread is None
    assert window._worker is None
    assert old_thread in window._retiring_threads
    assert delete_events == []

    # Releasing the old worker allows finished signal to retire & deleteLater safely
    import shiboken6

    release_ocr.set()
    qtbot.waitUntil(lambda: old_thread not in window._retiring_threads, timeout=5000)
    assert ("worker", False) in delete_events
    assert ("thread", False) in delete_events
    assert (not shiboken6.isValid(old_thread)) or (not old_thread.isRunning())

    # Now verify closeEvent waits on a timed-out worker/retiring thread until finished
    release_ocr.clear()
    ocr_entered.clear()
    window.btn_start.click()
    assert ocr_entered.wait(timeout=5.0)

    second_thread = window._worker_thread
    assert second_thread is not None
    orig_second_wait = second_thread.wait
    monkeypatch.setattr(
        second_thread,
        "wait",
        lambda *args: False if args == (5000,) else orig_second_wait(*args),
    )

    timer = threading.Timer(0.15, release_ocr.set)
    timer.start()
    try:
        window.close()
    finally:
        timer.cancel()
        release_ocr.set()

    assert (not shiboken6.isValid(second_thread)) or (not second_thread.isRunning())
    assert window._retiring_threads == []


def test_main_window_browse_buttons_screen_clamp_and_corrupt_presets(
    qtbot, tmp_path: Path, monkeypatch
):
    from PySide6.QtCore import QRect
    from PySide6.QtGui import QScreen
    from transapp.ui.main_window import MainWindow

    cfg_dir = tmp_path / "corrupt_cfg"
    cfg_dir.mkdir()
    (cfg_dir / "presets.json").write_text('{"bad": ', encoding="utf-8")

    # Simulate a 1080p laptop at 150% DPI scaling (available geometry 1280x672)
    monkeypatch.setattr(QScreen, "availableGeometry", lambda _self: QRect(0, 0, 1280, 672))

    window = MainWindow(preset_store=PresetStore(config_dir=cfg_dir))
    qtbot.addWidget(window)
    assert window.width() <= 1160
    assert window.height() <= 672

    # Browse buttons stay enabled while not running so clicking them auto-selects custom radio
    assert window.radio_dir_same.isChecked()
    assert window.btn_browse_out_dir.isEnabled() is True
    assert window.radio_ass_default.isChecked()
    assert window.btn_browse_ass_template.isEnabled() is True
    window.close()


def test_review_table_delete_selected_row_and_reindex_under_filter(qtbot, tmp_path: Path):
    from transapp.ui.review_table import ReviewTable

    table = ReviewTable()
    qtbot.addWidget(table)
    table.resize(900, 420)
    table.show()

    assert table.btn_delete_row.isEnabled() is False

    ev1 = SubtitleEvent(1, 500, 1200, "第一句", 0.96, False, _make_png_bytes((40, 200, 40)))
    ev2 = SubtitleEvent(2, 1300, 1600, "VNY", 0.65, True, _make_png_bytes((40, 40, 220)))
    ev3 = SubtitleEvent(3, 1700, 2400, "第三句", 0.78, True, _make_png_bytes((200, 40, 40)))

    emitted: list[int] = []
    table.events_changed.connect(lambda: emitted.append(len(table.events())))

    table.set_events([ev1, ev2, ev3])
    table.set_filter_mode("flagged_only")
    assert table.table.rowCount() == 2
    assert table.btn_delete_row.isEnabled() is True

    # Delete first visible flagged row (ev2: "VNY")
    table.table.selectRow(0)
    table.btn_delete_row.click()

    remaining = table.events()
    assert [(e.index, e.text) for e in remaining] == [(1, "第一句"), (2, "第三句")]
    assert table.radio_all.text() == "Tất cả (2)"
    assert table.radio_flagged.text() == "Chỉ dòng nghi ngờ (1)"
    assert table.table.rowCount() == 1
    assert table.table.item(0, 0).text() == "2 !"
    assert len(emitted) == 1

    # Delete remaining visible flagged row via Delete key shortcut
    qtbot.keyClick(table.table, Qt.Key.Key_Delete)
    assert [(e.index, e.text) for e in table.events()] == [(1, "第一句")]
    assert table.table.rowCount() == 0
    assert table.btn_delete_row.isEnabled() is False

    # Export reflects deletion
    srt_path = tmp_path / "after_delete.srt"
    export_srt(table.events(), srt_path)
    srt_text = srt_path.read_text(encoding="utf-8")
    assert "第一句" in srt_text
    assert "VNY" not in srt_text
    assert "第三句" not in srt_text


def test_main_window_prefers_directml_when_available(qtbot, tmp_path: Path, monkeypatch):
    from transapp.ui.main_window import MainWindow

    monkeypatch.setattr(OcrEngine, "available_providers", staticmethod(lambda: ["cpu", "directml"]))
    window = MainWindow(preset_store=PresetStore(config_dir=tmp_path / "cfg"))
    qtbot.addWidget(window)

    assert window.provider_combo.count() == 2
    assert window.provider_combo.currentData() == "directml"
    assert window.build_config().provider == "directml"

    cpu_idx = window.provider_combo.findData("cpu")
    assert cpu_idx >= 0
    window.provider_combo.setCurrentIndex(cpu_idx)
    assert window.build_config().provider == "cpu"

    assert window.ocr_model_combo.count() == 2
    assert window.ocr_model_combo.currentData() == "accurate"
    assert window.build_config().ocr_model == "accurate"
    fast_idx = window.ocr_model_combo.findData("fast")
    assert fast_idx >= 0
    window.ocr_model_combo.setCurrentIndex(fast_idx)
    assert window.build_config().ocr_model == "fast"
    window.close()


def test_studio_theme_icons_and_video_meta_badge(qtbot, tmp_path: Path):
    from PySide6.QtWidgets import QApplication
    from transapp.app import load_app_icon
    from transapp.ui.icons import ICON_NAMES, make_icon
    from transapp.ui.main_window import MainWindow
    from transapp.ui.theme import STUDIO_DARK_QSS, apply_studio_theme

    app = QApplication.instance()
    assert app is not None
    assert len(STUDIO_DARK_QSS.strip()) > 200
    apply_studio_theme(app)
    assert "QMainWindow" in app.styleSheet()

    app_icon = load_app_icon()
    assert not app_icon.isNull()
    assert not app_icon.pixmap(32, 32).isNull()

    for name in ICON_NAMES:
        icon = make_icon(name)
        assert not icon.isNull()
        pm = icon.pixmap(16, 16)
        assert not pm.isNull()

    video = tmp_path / "badge_test.mp4"
    _make_synthetic_subtitle_mp4(video, width=640, height=360, fps=25, duration_sec=1.0)

    window = MainWindow(preset_store=PresetStore(config_dir=tmp_path / "cfg"))
    qtbot.addWidget(window)
    assert window.video_meta_badge.isVisible() is False

    window.load_video(video)
    assert "640" in window.video_meta_badge.text()
    assert "360" in window.video_meta_badge.text()
    assert "25" in window.video_meta_badge.text()
    window.close()


def test_main_window_trim_in_out_controls_and_reset_on_video_switch(
    qtbot, tmp_path: Path, monkeypatch
):
    from PySide6.QtWidgets import QSlider
    import transapp.ui.main_window as mw_mod
    from transapp.ui.main_window import MainWindow, TrimRangeSlider

    ep1 = tmp_path / "trim_ep1.mp4"
    ep2 = tmp_path / "trim_ep2.mp4"
    _make_synthetic_subtitle_mp4(ep1, width=640, height=360, fps=25, duration_sec=2.0, cues=((200, 1600, "测试字幕"),))
    _make_synthetic_subtitle_mp4(ep2, width=640, height=360, fps=25, duration_sec=1.2, cues=((200, 800, "下集字幕"),))

    fake_ocr = OcrEngine(
        ocr_callable=lambda _img: (
            [[[[0, 0], [80, 0], [80, 24], [0, 24]], "测试字幕", 0.95]],
            [0.01],
        )
    )
    pipeline = ExtractionPipeline(ocr_engine=fake_ocr)
    window = MainWindow(preset_store=PresetStore(config_dir=tmp_path / "cfg"), pipeline=pipeline)
    qtbot.addWidget(window)
    window.show()

    assert isinstance(window.frame_slider, QSlider)
    assert isinstance(window.frame_slider, TrimRangeSlider)
    assert window.frame_slider.maximum() == 0
    window.frame_slider.set_trim_range(100, 500)
    window.frame_slider.repaint()
    window.frame_slider.set_trim_range(None, None)
    assert window.lbl_trim_range.text() == "Toàn tập"
    assert window.btn_set_in.isEnabled() is False
    assert window.btn_set_out.isEnabled() is False
    assert window.btn_clear_trim.isEnabled() is False

    window.load_video(ep1)
    assert window.btn_set_in.isEnabled() is True
    assert window.btn_set_out.isEnabled() is True
    assert window.btn_clear_trim.isEnabled() is False
    # ROI pill displays pixel dimensions when video metadata is loaded
    assert "px" in window.roi_label.text()
    assert "×" in window.roi_label.text()

    # Set In at 400ms (In-only repaint) and Out at 1400ms via buttons
    window.seek_to_ms(400)
    window.btn_set_in.click()
    cfg = window.build_config()
    assert cfg.trim_start_ms is not None and abs(cfg.trim_start_ms - 400) <= 60
    assert cfg.trim_end_ms is None
    assert window.btn_clear_trim.isEnabled() is True
    assert "→" in window.lbl_trim_range.text()
    window.frame_slider.repaint()

    window.seek_to_ms(1400)
    window.btn_set_out.click()
    cfg = window.build_config()
    assert cfg.trim_start_ms is not None and abs(cfg.trim_start_ms - 400) <= 60
    assert cfg.trim_end_ms is not None and abs(cfg.trim_end_ms - 1400) <= 60
    assert window.frame_slider.trim_range() == (cfg.trim_start_ms, cfg.trim_end_ms)
    window.frame_slider.repaint()

    # Auto-guard: setting In >= Out clears Out so In < Out always holds
    window.seek_to_ms(1600)
    window.btn_set_in.click()
    cfg = window.build_config()
    assert cfg.trim_start_ms is not None and abs(cfg.trim_start_ms - 1600) <= 60
    assert cfg.trim_end_ms is None

    # Auto-guard: setting Out <= In clears In so In < Out always holds (Out-only repaint)
    window.seek_to_ms(800)
    window.btn_set_out.click()
    cfg = window.build_config()
    assert cfg.trim_start_ms is None
    assert cfg.trim_end_ms is not None and abs(cfg.trim_end_ms - 800) <= 60
    window.frame_slider.repaint()

    # Keyboard shortcuts I / O work when focus is outside text inputs
    window.roi_canvas.setFocus()
    window.seek_to_ms(320)
    qtbot.keyClick(window.roi_canvas, Qt.Key.Key_I)
    cfg = window.build_config()
    assert cfg.trim_start_ms is not None and abs(cfg.trim_start_ms - 320) <= 60

    window.seek_to_ms(1200)
    qtbot.keyClick(window.roi_canvas, Qt.Key.Key_O)
    cfg = window.build_config()
    assert cfg.trim_end_ms is not None and abs(cfg.trim_end_ms - 1200) <= 60

    # Typing I / O in QLineEdit, QComboBox, QDoubleSpinBox, or editing table cell must NOT trigger trim
    prev_in, prev_out = cfg.trim_start_ms, cfg.trim_end_ms
    window.radio_dir_custom.setChecked(True)
    window.custom_dir_edit.setFocus()
    window.seek_to_ms(600)
    qtbot.keyClick(window.custom_dir_edit, Qt.Key.Key_I)
    qtbot.keyClick(window.custom_dir_edit, Qt.Key.Key_O)
    assert window.build_config().trim_start_ms == prev_in
    assert window.build_config().trim_end_ms == prev_out
    assert "i" in window.custom_dir_edit.text().lower()
    window.radio_dir_same.setChecked(True)
    window.custom_dir_edit.clear()

    window.provider_combo.setFocus()
    qtbot.keyClick(window.provider_combo, Qt.Key.Key_I)
    qtbot.keyClick(window.provider_combo, Qt.Key.Key_O)
    assert window.build_config().trim_start_ms == prev_in
    assert window.build_config().trim_end_ms == prev_out

    window.spin_conf.setFocus()
    qtbot.keyClick(window.spin_conf, Qt.Key.Key_I)
    qtbot.keyClick(window.spin_conf, Qt.Key.Key_O)
    assert window.build_config().trim_start_ms == prev_in
    assert window.build_config().trim_end_ms == prev_out

    window.review_table.set_events(
        [SubtitleEvent(1, 200, 800, "测试", 0.95, False, _make_png_bytes())]
    )
    edit_item = window.review_table.table.item(0, 4)
    assert edit_item is not None
    window.review_table.table.setCurrentCell(0, 4)
    window.review_table.table.editItem(edit_item)
    editor = window.focusWidget()
    assert editor is not None
    qtbot.keyClick(editor, Qt.Key.Key_I)
    qtbot.keyClick(editor, Qt.Key.Key_O)
    assert window.build_config().trim_start_ms == prev_in
    assert window.build_config().trim_end_ms == prev_out
    qtbot.keyClick(editor, Qt.Key.Key_Escape)
    window.review_table.clear()

    # Clear trim resets state
    window.btn_clear_trim.click()
    assert window.build_config().trim_start_ms is None
    assert window.build_config().trim_end_ms is None
    assert window.lbl_trim_range.text() == "Toàn tập"
    assert window.btn_clear_trim.isEnabled() is False

    # Re-set trim, extract, and verify export_ass receives frame_pts_ms from ExtractResult
    window.seek_to_ms(320)
    window.btn_set_in.click()
    window.seek_to_ms(1400)
    window.btn_set_out.click()
    window.roi_canvas.set_roi(RoiRect(0.10, 0.72, 0.80, 0.22))

    window.btn_start.click()
    qtbot.waitUntil(lambda: window.btn_export.isEnabled(), timeout=10000)
    assert window.current_result is not None
    assert len(window.current_result.frame_pts_ms) > 0

    captured_ass_kwargs: list[dict[str, object]] = []
    real_export_ass = mw_mod.export_ass

    def spy_export_ass(*args, **kwargs):
        captured_ass_kwargs.append(dict(kwargs))
        return real_export_ass(*args, **kwargs)

    monkeypatch.setattr(mw_mod, "export_ass", spy_export_ass)
    assert window.export_subtitles() is True
    assert len(captured_ass_kwargs) == 1
    assert captured_ass_kwargs[0].get("frame_pts_ms") == window.current_result.frame_pts_ms

    # Switching video resets trim back to full video
    window.load_video(ep2)
    assert window.build_config().trim_start_ms is None
    assert window.build_config().trim_end_ms is None
    assert window.lbl_trim_range.text() == "Toàn tập"
    assert window.btn_clear_trim.isEnabled() is False
    assert window.frame_slider.trim_range() == (None, None)
    window.close()


def test_main_window_playback_autorepeat_and_keyboard_scrubbing(qtbot, tmp_path: Path):
    from transapp.ui.main_window import MainWindow

    ep = tmp_path / "playback_test.mp4"
    _make_synthetic_subtitle_mp4(ep, width=640, height=360, fps=25, duration_sec=1.6)
    fake_ocr = OcrEngine(
        ocr_callable=lambda _img: (
            [[[[0, 0], [80, 0], [80, 24], [0, 24]], "你好世界", 0.95]],
            [0.01],
        )
    )
    window = MainWindow(
        preset_store=PresetStore(config_dir=tmp_path / "cfg"),
        pipeline=ExtractionPipeline(ocr_engine=fake_ocr),
    )
    qtbot.addWidget(window)
    window.show()
    window.load_video(ep)

    # Step buttons support press-and-hold auto-repeat and don't steal keyboard focus
    assert window.btn_step_back.autoRepeat() is True
    assert window.btn_step_fwd.autoRepeat() is True
    assert window.btn_play_pause.isEnabled() is True
    assert window.btn_play_pause.focusPolicy() == Qt.FocusPolicy.NoFocus
    assert window.frame_slider.focusPolicy() == Qt.FocusPolicy.NoFocus

    # Keyboard stepping: Right / Period / Greater (+1f) and Left / Comma / Less (-1f)
    start_ms = window._current_frame_ms
    qtbot.keyClick(window, Qt.Key.Key_Right)
    after_right = window._current_frame_ms
    assert after_right > start_ms

    qtbot.keyClick(window, Qt.Key.Key_Period)
    after_period = window._current_frame_ms
    assert after_period > after_right

    qtbot.keyClick(window, Qt.Key.Key_Greater, Qt.KeyboardModifier.ShiftModifier)
    after_gt = window._current_frame_ms
    assert after_gt > after_period

    qtbot.keyClick(window, Qt.Key.Key_Less, Qt.KeyboardModifier.ShiftModifier)
    assert window._current_frame_ms < after_gt

    qtbot.keyClick(window, Qt.Key.Key_Comma)
    qtbot.keyClick(window, Qt.Key.Key_Left)
    assert window._current_frame_ms <= after_right

    # Shift+Right (+1s), Home (start), End (end)
    qtbot.keyClick(window, Qt.Key.Key_Home)
    assert window._current_frame_ms == window._start_pts_ms
    qtbot.keyClick(window, Qt.Key.Key_Right, Qt.KeyboardModifier.ShiftModifier)
    assert window._current_frame_ms >= window._start_pts_ms + 900
    qtbot.keyClick(window, Qt.Key.Key_End)
    end_ms = window._current_frame_ms
    assert end_ms >= window._start_pts_ms + 1000
    qtbot.keyClick(window, Qt.Key.Key_Home)

    # Space toggles Play/Pause and advances sequential frames cleanly
    qtbot.keyClick(window, Qt.Key.Key_Space)
    assert window._is_playing is True
    t_before = window._current_frame_ms
    window._on_playback_tick()
    window._on_playback_tick()
    assert window._current_frame_ms > t_before

    # Second Space pauses cleanly
    qtbot.keyClick(window, Qt.Key.Key_Space)
    assert window._is_playing is False

    # Playing from EOF wraps to start and auto-stops at EOF
    window._seek_to_end()
    window.start_playback()
    assert window._is_playing is True
    for _ in range(60):
        if not window._is_playing:
            break
        window._on_playback_tick()
    assert window._is_playing is False

    # Space / Left / Right must NOT trigger playback or seek while typing in QLineEdit
    window._seek_to_start()
    window.radio_dir_custom.setChecked(True)
    window.custom_dir_edit.setFocus()
    qtbot.keyClick(window.custom_dir_edit, Qt.Key.Key_Space)
    qtbot.keyClick(window.custom_dir_edit, Qt.Key.Key_Period)
    assert window._is_playing is False
    assert window._current_frame_ms == window._start_pts_ms
    window.close()


def test_review_table_jump_sync_nudge_and_flag_reason_auto_clear_in_place(qtbot):
    from transapp.ui.review_table import ReviewTable

    table = ReviewTable()
    qtbot.addWidget(table)
    table.resize(940, 440)
    table.show()

    ev1 = SubtitleEvent(
        index=1,
        start_ms=1000,
        end_ms=2200,
        text="第一行\n第二行",
        confidence=0.96,
        flagged=False,
        thumbnail_png=_make_png_bytes((40, 200, 40)),
    )
    ev2 = SubtitleEvent(
        index=2,
        start_ms=2500,
        end_ms=3700,
        text="可疑字幕",
        confidence=0.68,
        flagged=True,
        thumbnail_png=_make_png_bytes((40, 40, 220)),
        flag_reason="Độ tin cậy thấp (0.68 < 0.85)",
    )
    ev3 = SubtitleEvent(
        index=3,
        start_ms=4000,
        end_ms=4080,
        text="短",
        confidence=0.71,
        flagged=True,
        thumbnail_png=_make_png_bytes((200, 40, 40)),
        flag_reason="",
    )

    jumps: list[int] = []
    snap_starts: list[bool] = []
    snap_ends: list[bool] = []
    modified_hits: list[bool] = []
    table.jump_requested.connect(jumps.append)
    table.snap_start_requested.connect(lambda: snap_starts.append(True))
    table.snap_end_requested.connect(lambda: snap_ends.append(True))
    table.events_modified.connect(lambda: modified_hits.append(True))

    table.set_frame_step_ms(40)
    table.set_events([ev1, ev2, ev3])

    # Uniform HH:MM:SS.mmm display + multi-line \N encoding
    assert table.table.item(0, 1).text() == "00:00:01.000"
    assert table.table.item(0, 2).text() == "00:00:02.200"
    assert table.table.item(0, 4).text() == r"第一行\N第二行"
    assert table.flag_banner.isVisible() is False
    assert "1.20s" in table.lbl_cue_meta.text()

    # Editing multi-line cell with \N preserves real \n in SubtitleEvent.text
    table.table.item(0, 4).setText(r"第一行改\N第二行改")
    assert table.events()[0].text == "第一行改\n第二行改"
    assert len(modified_hits) == 1

    # Flagged row displays flag_reason in side-panel banner and cell tooltip
    table.table.selectRow(1)
    assert table.flag_banner.isVisible() is True
    assert "Độ tin cậy thấp (0.68 < 0.85)" in table.lbl_flag_reason.text()
    assert table.table.item(1, 4).toolTip() == "Độ tin cậy thấp (0.68 < 0.85)"

    # Jump Start / End buttons and Alt+S / Alt+E emit exact start_ms / end_ms
    table.btn_jump_start.click()
    table.btn_jump_end.click()
    assert jumps == [2500, 3700]

    qtbot.keyClick(table.table, Qt.Key.Key_S, Qt.KeyboardModifier.AltModifier)
    qtbot.keyClick(table.table, Qt.Key.Key_E, Qt.KeyboardModifier.AltModifier)
    assert jumps == [2500, 3700, 2500, 3700]

    # Snap In / Out buttons and Ctrl+[ / Ctrl+] emit snap signals, and helpers update timing
    table.btn_snap_in.click()
    table.btn_snap_out.click()
    qtbot.keyClick(table.table, Qt.Key.Key_BracketLeft, Qt.KeyboardModifier.ControlModifier)
    qtbot.keyClick(table.table, Qt.Key.Key_BracketRight, Qt.KeyboardModifier.ControlModifier)
    assert len(snap_starts) == 2
    assert len(snap_ends) == 2

    assert table.apply_playhead_to_selected_start(2600) is True
    assert table.events()[1].start_ms == 2600
    assert table.apply_playhead_to_selected_start(4000) is False  # >= end_ms rejected
    assert table.apply_playhead_to_selected_end(3600) is True
    assert table.events()[1].end_ms == 3600
    assert table.apply_playhead_to_selected_end(2500) is False  # <= start_ms rejected

    # Nudge +1f / -1f shifts (start_ms, end_ms) preserving duration and clamping >= 0
    table.btn_nudge_fwd.click()
    assert (table.events()[1].start_ms, table.events()[1].end_ms) == (2640, 3640)
    table.btn_nudge_back.click()
    assert (table.events()[1].start_ms, table.events()[1].end_ms) == (2600, 3600)
    assert table.nudge_selected_timing(-5000) is True
    assert (table.events()[1].start_ms, table.events()[1].end_ms) == (0, 1000)
    # Restore timing cleanly
    assert table.nudge_selected_timing(2500) is True
    assert (table.events()[1].start_ms, table.events()[1].end_ms) == (2500, 3500)

    # Switch to flagged_only: manual text edit clears flagged & decrements badge IN-PLACE (no eviction)
    table.set_filter_mode("flagged_only")
    assert table.table.rowCount() == 2
    assert table.radio_flagged.text() == "Chỉ dòng nghi ngờ (2)"

    table.table.selectRow(0)
    table.table.item(0, 4).setText("已人工校对")
    assert table.table.rowCount() == 2  # Row stays visible in flagged_only until filter switch
    assert table.radio_flagged.text() == "Chỉ dòng nghi ngờ (1)"
    assert table.events()[1].flagged is False
    assert table.events()[1].flag_reason == ""
    assert table.table.item(0, 0).text() == "2"
    assert table.table.item(0, 4).toolTip() == ""
    assert table.flag_banner.isVisible() is False

    # Selecting second row (empty flag_reason fallback + short cue warning) & clicking [Duyệt]
    table.table.selectRow(1)
    assert table.flag_banner.isVisible() is True
    assert table.lbl_flag_reason.text() == "Cần kiểm tra"
    assert bool(table.lbl_cue_meta.property("warning")) is True
    table.btn_mark_ok.click()
    assert table.table.rowCount() == 2  # Still visible in-place
    assert table.radio_flagged.text() == "Chỉ dòng nghi ngờ (0)"
    assert table.events()[2].flagged is False
    assert table.flag_banner.isVisible() is False

    # Switching tabs refreshes cleanly
    table.set_filter_mode("all")
    table.set_filter_mode("flagged_only")
    assert table.table.rowCount() == 0


def test_main_window_unsaved_edits_guard_dynamic_export_and_drag_drop(
    qtbot, tmp_path: Path, monkeypatch
):
    from PySide6.QtCore import QMimeData, QPointF, QUrl
    from PySide6.QtGui import QDesktopServices, QDragEnterEvent, QDropEvent
    from PySide6.QtWidgets import QMessageBox
    from transapp.ui.main_window import MainWindow

    ep1 = tmp_path / "guard_ep1.mp4"
    ep2 = tmp_path / "guard_ep2.mp4"
    _make_synthetic_subtitle_mp4(
        ep1, width=640, height=360, fps=25, duration_sec=1.6, start_pts_ms=2000, cues=((200, 1200, "第一句"),)
    )
    _make_synthetic_subtitle_mp4(
        ep2, width=640, height=360, fps=25, duration_sec=1.0, cues=((200, 800, "第二集"),)
    )

    fake_ocr = OcrEngine(
        ocr_callable=lambda _img: (
            [[[[0, 0], [80, 0], [80, 24], [0, 24]], "第一句", 0.94]],
            [0.01],
        )
    )
    pipeline = ExtractionPipeline(ocr_engine=fake_ocr)
    store = PresetStore(config_dir=tmp_path / "cfg")

    unsaved_prompts: list[bool] = []
    unsaved_allow = {"value": False}

    def fake_unsaved_confirm() -> bool:
        unsaved_prompts.append(True)
        return unsaved_allow["value"]

    window = MainWindow(
        preset_store=store,
        pipeline=pipeline,
        unsaved_confirm_fn=fake_unsaved_confirm,
    )
    qtbot.addWidget(window)
    window.show()

    # Dynamic btn_export label synced to chk_ass / chk_srt
    assert window.btn_export.text() == "Xuất .ass + .srt"
    assert window.btn_open_out_dir.isEnabled() is False
    window.chk_srt.setChecked(False)
    assert window.btn_export.text() == "Xuất .ass"
    window.chk_ass.setChecked(False)
    assert window.btn_export.isEnabled() is False
    window.chk_srt.setChecked(True)
    assert window.btn_export.text() == "Xuất .srt"
    window.chk_ass.setChecked(True)
    assert window.btn_export.text() == "Xuất .ass + .srt"

    # Drag-and-drop non-video file is ignored; video onto MainWindow loads ep1
    bad_mime = QMimeData()
    bad_mime.setUrls([QUrl.fromLocalFile(str(tmp_path / "notes.txt"))])
    bad_enter = QDragEnterEvent(
        QPointF(20.0, 20.0),
        Qt.DropAction.CopyAction,
        bad_mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    window.dragEnterEvent(bad_enter)
    assert bad_enter.isAccepted() is False

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(ep1))])
    drag_enter = QDragEnterEvent(
        QPointF(20.0, 20.0),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    window.dragEnterEvent(drag_enter)
    assert drag_enter.isAccepted() is True

    drop_event = QDropEvent(
        QPointF(20.0, 20.0),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    window.dropEvent(drop_event)
    assert drop_event.isAccepted() is True
    assert window._video_path == ep1
    assert window._start_pts_ms == 2000

    # Extract so we have events in ReviewTable
    window.roi_canvas.set_roi(RoiRect(0.10, 0.72, 0.80, 0.22))
    window.btn_start.click()
    qtbot.waitUntil(lambda: len(window.review_table.events()) > 0, timeout=10000)
    qtbot.waitUntil(lambda: window.btn_export.isEnabled(), timeout=5000)
    assert window._has_unexported_edits is False

    # Jump sync seeks MainWindow to _start_pts_ms + rel_ms; Ctrl+[ snaps Start to playhead
    ev0 = window.review_table.events()[0]
    window.review_table.jump_requested.emit(ev0.start_ms)
    assert abs(window._current_frame_ms - (2000 + ev0.start_ms)) <= 60

    window.seek_to_ms(2000 + ev0.start_ms + 80)
    snapped_rel = window._current_rel_ms()
    window.review_table.snap_start_requested.emit()
    assert window.review_table.events()[0].start_ms == snapped_rel
    assert window._has_unexported_edits is True

    # Manual edit keeps _has_unexported_edits = True -> blocks load_video, start_extraction, closeEvent
    window.review_table.table.item(0, 4).setText("第一句手动修正")
    assert window._has_unexported_edits is True

    unsaved_allow["value"] = False
    window.load_video(ep2)
    assert window._video_path == ep1  # blocked!
    assert len(unsaved_prompts) == 1

    window.start_extraction()
    assert window._is_running() is False  # blocked!
    assert len(unsaved_prompts) == 2

    window.close()
    assert window.isVisible() is True  # closeEvent ignored!
    assert len(unsaved_prompts) == 3

    # Empty-cue export guard warns and returns False without writing files
    info_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda _parent, title, text: info_calls.append((title, text)),
    )
    window.review_table.table.item(0, 4).setText("   ")
    assert window.export_subtitles() is False
    assert len(info_calls) == 1
    assert not (tmp_path / "guard_ep1.ass").exists()
    assert not (tmp_path / "guard_ep1.srt").exists()

    # Restore text -> export succeeds -> resets _has_unexported_edits and enables btn_open_out_dir
    window.review_table.table.item(0, 4).setText("第一句手动修正")
    assert window.export_subtitles() is True
    assert window._has_unexported_edits is False
    assert window.btn_open_out_dir.isEnabled() is True

    opened_urls: list[QUrl] = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: (opened_urls.append(url), True)[1])
    window.btn_open_out_dir.click()
    assert len(opened_urls) == 1
    assert Path(opened_urls[0].toLocalFile()) == tmp_path

    # With edits exported, drag-and-drop onto RoiCanvas loads ep2 without prompting
    prev_prompts = len(unsaved_prompts)
    mime2 = QMimeData()
    mime2.setUrls([QUrl.fromLocalFile(str(ep2))])
    canvas_enter = QDragEnterEvent(
        QPointF(10.0, 10.0),
        Qt.DropAction.CopyAction,
        mime2,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    window.roi_canvas.dragEnterEvent(canvas_enter)
    assert canvas_enter.isAccepted() is True
    canvas_drop = QDropEvent(
        QPointF(10.0, 10.0),
        Qt.DropAction.CopyAction,
        mime2,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    window.roi_canvas.dropEvent(canvas_drop)
    assert canvas_drop.isAccepted() is True
    assert window._video_path == ep2
    assert len(unsaved_prompts) == prev_prompts
    assert window.btn_open_out_dir.isEnabled() is False

    # Close cleanly
    window.close()


