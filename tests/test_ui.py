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

    # Test 1 frame updates diagnostic label cleanly (asynchronously)
    window.seek_to_ms(500)
    release_ocr.set()
    window.btn_test_frame.click()
    qtbot.waitUntil(lambda: "第一句" in window.test_result_label.text(), timeout=5000)
    qtbot.waitUntil(lambda: window.btn_start.isEnabled(), timeout=5000)

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
    assert window.test_current_frame() is True
    qtbot.waitUntil(lambda: "偏移测试" in window.test_result_label.text(), timeout=5000)
    qtbot.waitUntil(lambda: not window._is_frame_testing(), timeout=5000)
    assert len(init_calls) == 0
    assert len(read_calls) == 0

    # Even if preview reader & cached frame are cleared, test_current_frame opens/reads at most once
    window._close_preview_reader()
    window._current_frame_bgr = None
    assert window.test_current_frame() is True
    qtbot.waitUntil(lambda: "偏移测试" in window.test_result_label.text(), timeout=5000)
    qtbot.waitUntil(lambda: not window._is_frame_testing(), timeout=5000)
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


def test_review_table_multi_delete_merge_split_and_undo_redo(qtbot):
    from PySide6.QtCore import QItemSelectionModel
    from PySide6.QtWidgets import QAbstractItemView
    from transapp.ui.review_table import ReviewTable

    table = ReviewTable()
    qtbot.addWidget(table)
    table.resize(980, 440)
    table.show()

    # ExtendedSelection, no AnyKeyPressed edit trigger, and auto-repeat on nudge buttons
    assert table.table.selectionMode() == QAbstractItemView.SelectionMode.ExtendedSelection
    triggers = table.table.editTriggers()
    assert bool(triggers & QAbstractItemView.EditTrigger.DoubleClicked)
    assert bool(triggers & QAbstractItemView.EditTrigger.EditKeyPressed)
    assert not bool(triggers & QAbstractItemView.EditTrigger.AnyKeyPressed)
    assert table.btn_nudge_back.autoRepeat() is True
    assert table.btn_nudge_fwd.autoRepeat() is True
    assert table.can_undo() is False
    assert table.can_redo() is False
    assert table.btn_undo.isEnabled() is False
    assert table.btn_redo.isEnabled() is False

    thumb1 = _make_png_bytes((30, 180, 30))
    thumb2 = _make_png_bytes((30, 30, 180))
    thumb3 = _make_png_bytes((180, 30, 30))
    thumb4 = _make_png_bytes((180, 180, 30))

    ev1 = SubtitleEvent(1, 500, 1500, "我们今天一起去吃饭", 0.96, False, thumb1)
    ev2 = SubtitleEvent(2, 1550, 2400, "我们今天一起去吃", 0.82, True, thumb2, "Độ tin cậy thấp")
    ev3 = SubtitleEvent(3, 3000, 3900, "第一句", 0.85, False, thumb3)
    ev4 = SubtitleEvent(4, 4000, 5000, "第二句", 0.95, False, thumb4)
    table.set_events([ev1, ev2, ev3, ev4])

    def _select_rows(rows: list[int]) -> None:
        sel = table.table.selectionModel()
        sel.clearSelection()
        flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
        for r in rows:
            idx = table.table.model().index(r, 0)
            sel.select(idx, flags)
        if rows:
            table.table.setCurrentCell(rows[-1], 0, QItemSelectionModel.SelectionFlag.NoUpdate)

    # 1. Non-contiguous multi-row delete -> re-index -> single-step Ctrl+Z / Ctrl+Y round-trip
    _select_rows([0, 2])
    assert table.btn_merge.isEnabled() is False
    assert table.merge_selected_rows() is False  # non-contiguous rejected without mutating
    assert len(table.events()) == 4

    assert table.delete_selected_rows() is True
    assert [(e.index, e.text) for e in table.events()] == [
        (1, "我们今天一起去吃"),
        (2, "第二句"),
    ]
    assert table.can_undo() is True
    assert table.btn_undo.isEnabled() is True

    # Ctrl+Z restores all 4 events with exact thumbnail_png bytes identity
    table.table.setFocus()
    qtbot.keyClick(table.table, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    restored = table.events()
    assert [(e.index, e.text) for e in restored] == [
        (1, "我们今天一起去吃饭"),
        (2, "我们今天一起去吃"),
        (3, "第一句"),
        (4, "第二句"),
    ]
    assert restored[0].thumbnail_png is thumb1
    assert restored[1].thumbnail_png is thumb2
    assert restored[2].thumbnail_png is thumb3
    assert restored[3].thumbnail_png is thumb4
    assert table.can_redo() is True

    # Ctrl+Y re-applies delete, then Ctrl+Z restores again
    qtbot.keyClick(table.table, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
    assert len(table.events()) == 2
    qtbot.keyClick(table.table, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert len(table.events()) == 4

    # 2. Contiguous near-duplicate merge (text_similarity >= 0.80 -> dedup keeping longer/higher-conf)
    _select_rows([0, 1])
    assert table.btn_merge.isEnabled() is True
    qtbot.keyClick(table.table, Qt.Key.Key_M, Qt.KeyboardModifier.ControlModifier)
    after_dup_merge = table.events()
    assert len(after_dup_merge) == 3
    assert after_dup_merge[0].start_ms == 500
    assert after_dup_merge[0].end_ms == 2400
    assert after_dup_merge[0].text == "我们今天一起去吃饭"
    assert after_dup_merge[0].confidence == pytest.approx(0.96)
    assert after_dup_merge[0].thumbnail_png is thumb1
    assert [e.index for e in after_dup_merge] == [1, 2, 3]

    # Undo merge restores exact original 4 rows
    assert table.undo() is True
    assert len(table.events()) == 4
    assert table.events()[1].thumbnail_png is thumb2

    # 3. Contiguous distinct merge (text_similarity < 0.80 -> join with \n, highest-conf thumbnail)
    _select_rows([2, 3])
    assert table.merge_selected_rows() is True
    after_distinct_merge = table.events()
    assert len(after_distinct_merge) == 3
    merged_cue = after_distinct_merge[2]
    assert merged_cue.start_ms == 3000
    assert merged_cue.end_ms == 5000
    assert merged_cue.text == "第一句\n第二句"
    assert merged_cue.confidence == pytest.approx(0.95)
    assert merged_cue.thumbnail_png is thumb4
    assert merged_cue.flagged is False

    # 4. Split multi-line cue at playhead (inside start..end) -> splits lines & shares thumbnail_png
    _select_rows([2])
    assert table.btn_split.isEnabled() is True
    assert table.split_selected_row(playhead_rel_ms=3650) is True
    after_split = table.events()
    assert len(after_split) == 4
    assert (after_split[2].start_ms, after_split[2].end_ms, after_split[2].text) == (3000, 3650, "第一句")
    assert (after_split[3].start_ms, after_split[3].end_ms, after_split[3].text) == (3650, 5000, "第二句")
    assert after_split[2].thumbnail_png is thumb4
    assert after_split[3].thumbnail_png is thumb4

    # 5. Split single-line cue with playhead outside cue -> splits at midpoint & duplicates text
    _select_rows([0])
    assert table.split_selected_row(playhead_rel_ms=100) is True
    after_mid = table.events()
    assert len(after_mid) == 5
    assert (after_mid[0].start_ms, after_mid[0].end_ms, after_mid[0].text) == (500, 1000, "我们今天一起去吃饭")
    assert (after_mid[1].start_ms, after_mid[1].end_ms, after_mid[1].text) == (1000, 1500, "我们今天一起去吃饭")
    assert after_mid[0].thumbnail_png is thumb1
    assert after_mid[1].thumbnail_png is thumb1

    # Ctrl+Z / Ctrl+Shift+Z round-trip preserves thumbnail_png identity
    qtbot.keyClick(table.table, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert len(table.events()) == 4
    qtbot.keyClick(
        table.table,
        Qt.Key.Key_Z,
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
    )
    assert len(table.events()) == 5
    assert table.events()[0].thumbnail_png is thumb1
    assert table.events()[1].thumbnail_png is thumb1

    # 6. Cue with duration < 2ms cannot be split
    table.set_events([SubtitleEvent(1, 1000, 1001, "极短", 0.90, False, thumb1)])
    assert table.btn_split.isEnabled() is False
    assert table.split_selected_row() is False


def test_review_table_find_replace_all_undoable_and_template_style_selector(
    qtbot, tmp_path: Path, monkeypatch
):
    from PySide6.QtWidgets import QAbstractItemView
    import transapp.ui.main_window as mw_mod
    from transapp.ui.main_window import MainWindow
    from transapp.ui.review_table import ReviewTable

    # --- Part A: Collapsible Find & Replace Bar + Single-Step Undo ---
    table = ReviewTable()
    qtbot.addWidget(table)
    table.resize(980, 440)
    table.show()

    assert not bool(table.table.editTriggers() & QAbstractItemView.EditTrigger.AnyKeyPressed)
    assert table.find_bar.isVisible() is False

    thumb = _make_png_bytes((90, 160, 220))
    table.set_events(
        [
            SubtitleEvent(1, 100, 900, "萧炎向前一步", 0.96, False, thumb),
            SubtitleEvent(2, 1000, 1800, "普通对白", 0.95, False, thumb),
            SubtitleEvent(3, 2000, 2900, "萧炎大喝一声", 0.72, True, thumb, "Độ tin cậy thấp"),
            SubtitleEvent(4, 3000, 3900, "又是萧炎", 0.75, True, thumb, "Ký tự lạ"),
        ]
    )

    # Ctrl+F opens collapsible bar
    table.table.setFocus()
    qtbot.keyClick(table.table, Qt.Key.Key_F, Qt.KeyboardModifier.ControlModifier)
    assert table.find_bar.isVisible() is True

    table.find_edit.setText("萧炎")
    assert table.find_next() is True  # row 0 -> next match row 2
    assert table.selected_event() is not None and table.selected_event().index == 3
    assert table.find_next() is True  # row 2 -> row 3
    assert table.selected_event().index == 4
    assert table.find_prev() is True  # row 3 -> row 2
    assert table.selected_event().index == 3

    # Ctrl+H focuses replace_edit; replace_current updates selected row and clears flagged
    qtbot.keyClick(table.table, Qt.Key.Key_H, Qt.KeyboardModifier.ControlModifier)
    assert table.find_bar.isVisible() is True
    table.replace_edit.setText("药老")
    assert table.replace_current() is True
    assert table.events()[2].text == "药老大喝一声"
    assert table.events()[2].flagged is False

    # Undo restores replace_current cleanly
    assert table.undo() is True
    assert table.events()[2].text == "萧炎大喝一声"
    assert table.events()[2].flagged is True

    # replace_all replaces all 3 occurrences in ONE undoable step and clears flagged on modified rows
    modified_calls: list[bool] = []
    table.events_modified.connect(lambda: modified_calls.append(True))
    count = table.replace_all()
    assert count == 3
    assert [e.text for e in table.events()] == [
        "药老向前一步",
        "普通对白",
        "药老大喝一声",
        "又是药老",
    ]
    assert table.events()[2].flagged is False
    assert table.events()[3].flagged is False
    assert len(modified_calls) == 1

    # Single Undo reverts ALL 3 rows and restores their flagged states
    assert table.undo() is True
    assert [e.text for e in table.events()] == [
        "萧炎向前一步",
        "普通对白",
        "萧炎大喝一声",
        "又是萧炎",
    ]
    assert table.events()[2].flagged is True
    assert table.events()[3].flagged is True

    # Escape closes Find/Replace bar
    qtbot.keyClick(table.find_edit, Qt.Key.Key_Escape)
    assert table.find_bar.isVisible() is False

    # --- Part B: MainWindow ASS Template Style Selector & Playhead Split Wiring ---
    video_path = tmp_path / "style_ep.mp4"
    _make_synthetic_subtitle_mp4(
        video_path,
        width=640,
        height=360,
        fps=25,
        duration_sec=1.6,
        cues=((200, 1400, "第一行\n第二行"),),
    )

    single_style_tpl = tmp_path / "single.ass"
    single_style_tpl.write_text(
        "[Script Info]\nPlayResX: 1920\nPlayResY: 1080\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize\n"
        "Style: OnlyStyle,Microsoft YaHei,54\n\n[Events]\n",
        encoding="utf-8",
    )

    multi_style_tpl = tmp_path / "multi_default.ass"
    multi_style_tpl.write_text(
        "[Script Info]\nPlayResX: 1920\nPlayResY: 1080\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize\n"
        "Style: OP_Kara,Microsoft YaHei,48\n"
        "Style: Default,Microsoft YaHei,54\n"
        "Style: SignTop,Microsoft YaHei,44\n\n[Events]\n",
        encoding="utf-8",
    )

    multi_no_default_tpl = tmp_path / "multi_nodefault.ass"
    multi_no_default_tpl.write_text(
        "[Script Info]\nPlayResX: 1920\nPlayResY: 1080\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize\n"
        "Style: MainSub,Microsoft YaHei,54\n"
        "Style: NoteSub,Microsoft YaHei,42\n\n[Events]\n",
        encoding="utf-8",
    )

    fake_ocr = OcrEngine(
        ocr_callable=lambda _img: (
            [[[[0, 0], [80, 0], [80, 24], [0, 24]], "第一行\n第二行", 0.95]],
            [0.01],
        )
    )
    window = MainWindow(
        preset_store=PresetStore(config_dir=tmp_path / "cfg"),
        pipeline=ExtractionPipeline(ocr_engine=fake_ocr),
    )
    qtbot.addWidget(window)
    window.show()

    # Hidden by default (Default ASS radio)
    assert window.ass_style_row_widget.isVisible() is False

    # Hidden when custom template has <= 1 style
    window.radio_ass_template.setChecked(True)
    window.ass_template_edit.setText(str(single_style_tpl))
    assert window.ass_style_row_widget.isVisible() is False

    # Visible when custom template has > 1 styles; pre-selects "Default" if present
    window.ass_template_edit.setText(str(multi_style_tpl))
    assert window.ass_style_row_widget.isVisible() is True
    assert [
        window.ass_style_combo.itemText(i) for i in range(window.ass_style_combo.count())
    ] == ["OP_Kara", "Default", "SignTop"]
    assert window.ass_style_combo.currentText() == "Default"

    # Pre-selects styles[0] when "Default" is absent
    window.ass_template_edit.setText(str(multi_no_default_tpl))
    assert window.ass_style_row_widget.isVisible() is True
    assert window.ass_style_combo.currentText() == "MainSub"

    # Switch back to multi_default and choose "SignTop"
    window.ass_template_edit.setText(str(multi_style_tpl))
    window.ass_style_combo.setCurrentText("SignTop")

    # Switching back to Default ASS hides style row, switching to template restores visibility
    window.radio_ass_default.setChecked(True)
    assert window.ass_style_row_widget.isVisible() is False
    window.radio_ass_template.setChecked(True)
    assert window.ass_style_row_widget.isVisible() is True
    window.ass_style_combo.setCurrentText("SignTop")

    # Load video, extract, test Ctrl+K split at playhead, and verify export_ass receives style_name
    window.load_video(video_path)
    window.roi_canvas.set_roi(RoiRect(0.10, 0.72, 0.80, 0.22))
    window.btn_start.click()
    qtbot.waitUntil(lambda: window.btn_export.isEnabled(), timeout=10000)

    ev0 = window.review_table.events()[0]
    target_split = (ev0.start_ms + ev0.end_ms) // 2 + 80
    if not (ev0.start_ms < target_split < ev0.end_ms):
        target_split = (ev0.start_ms + ev0.end_ms) // 2
    window.seek_to_ms(window._start_pts_ms + target_split)
    actual_rel = window._current_rel_ms()
    assert ev0.start_ms < actual_rel < ev0.end_ms

    window.review_table.table.selectRow(0)
    window.review_table.table.setFocus()
    qtbot.keyClick(window.review_table.table, Qt.Key.Key_K, Qt.KeyboardModifier.ControlModifier)
    split_events = window.review_table.events()
    assert len(split_events) == 2
    assert split_events[0].end_ms == actual_rel
    assert split_events[1].start_ms == actual_rel

    captured_kwargs: list[dict[str, object]] = []
    real_export_ass = mw_mod.export_ass

    def spy_export_ass(*args, **kwargs):
        captured_kwargs.append(dict(kwargs))
        return real_export_ass(*args, **kwargs)

    monkeypatch.setattr(mw_mod, "export_ass", spy_export_ass)
    assert window.export_subtitles() is True
    assert len(captured_kwargs) == 1
    assert captured_kwargs[0].get("style_name") == "SignTop"

    ass_content = (tmp_path / "style_ep.ass").read_text(encoding="utf-8")
    assert ",SignTop," in ass_content
    window.close()


def test_transport_and_trim_shortcuts_work_when_review_table_has_focus(qtbot, tmp_path: Path):
    from PySide6.QtWidgets import QAbstractItemView
    from transapp.ui.main_window import MainWindow

    ep = tmp_path / "focus_transport.mp4"
    _make_synthetic_subtitle_mp4(ep, width=640, height=360, fps=25, duration_sec=2.0)

    window = MainWindow(preset_store=PresetStore(config_dir=tmp_path / "cfg"))
    qtbot.addWidget(window)
    window.show()
    window.load_video(ep)

    # Layout floor invariant
    hint = window.minimumSizeHint()
    assert hint.width() <= 1280
    assert hint.height() <= 672

    # Non-text action controls use NoFocus so Space toggles playback instead of re-firing buttons
    for ctrl in (
        window.btn_test_frame,
        window.btn_browse_video,
        window.btn_export,
        window.review_table.radio_all,
        window.review_table.radio_flagged,
    ):
        assert ctrl.focusPolicy() == Qt.FocusPolicy.NoFocus

    thumb = _make_png_bytes((60, 140, 220))
    window.review_table.set_events(
        [
            SubtitleEvent(1, 200, 600, "第一行", 0.95, False, thumb),
            SubtitleEvent(2, 700, 1100, "第二行", 0.95, False, thumb),
            SubtitleEvent(3, 1200, 1600, "第三行", 0.95, False, thumb),
        ]
    )
    window.review_table.table.selectRow(0)
    window.review_table.table.setFocus()

    # Frame stepping & scrub shortcuts work while ReviewTable.table has focus
    start_ms = window._current_frame_ms
    qtbot.keyClick(window.review_table.table, Qt.Key.Key_Right)
    after_right = window._current_frame_ms
    assert after_right > start_ms

    qtbot.keyClick(window.review_table.table, Qt.Key.Key_Period)
    after_period = window._current_frame_ms
    assert after_period > after_right

    qtbot.keyClick(
        window.review_table.table,
        Qt.Key.Key_Greater,
        Qt.KeyboardModifier.ShiftModifier,
    )
    after_gt = window._current_frame_ms
    assert after_gt > after_period

    qtbot.keyClick(
        window.review_table.table,
        Qt.Key.Key_Less,
        Qt.KeyboardModifier.ShiftModifier,
    )
    assert window._current_frame_ms < after_gt

    qtbot.keyClick(window.review_table.table, Qt.Key.Key_Comma)
    qtbot.keyClick(window.review_table.table, Qt.Key.Key_Left)
    assert window._current_frame_ms <= after_right

    # Shift+Right (+1s), Home, End work from ReviewTable.table
    qtbot.keyClick(window.review_table.table, Qt.Key.Key_Home)
    assert window._current_frame_ms == window._start_pts_ms
    qtbot.keyClick(
        window.review_table.table,
        Qt.Key.Key_Right,
        Qt.KeyboardModifier.ShiftModifier,
    )
    assert window._current_frame_ms >= window._start_pts_ms + 900
    qtbot.keyClick(window.review_table.table, Qt.Key.Key_End)
    assert window._current_frame_ms >= window._start_pts_ms + 1400

    # I / O trim shortcuts work from ReviewTable.table
    window.seek_to_ms(window._start_pts_ms + 400)
    qtbot.keyClick(window.review_table.table, Qt.Key.Key_I)
    assert window.build_config().trim_start_ms is not None
    assert abs(window.build_config().trim_start_ms - 400) <= 60

    window.seek_to_ms(window._start_pts_ms + 1400)
    qtbot.keyClick(window.review_table.table, Qt.Key.Key_O)
    assert window.build_config().trim_end_ms is not None
    assert abs(window.build_config().trim_end_ms - 1400) <= 60

    # Space toggles Play / Pause while ReviewTable.table has focus
    qtbot.keyClick(window.review_table.table, Qt.Key.Key_Home)
    qtbot.keyClick(window.review_table.table, Qt.Key.Key_Space)
    assert window._is_playing is True
    qtbot.keyClick(window.review_table.table, Qt.Key.Key_Space)
    assert window._is_playing is False

    # Up / Down remain owned by ReviewTable for row navigation
    window.review_table.table.selectRow(0)
    qtbot.keyClick(window.review_table.table, Qt.Key.Key_Down)
    assert window.review_table.selected_event() is not None
    assert window.review_table.selected_event().index == 2
    qtbot.keyClick(window.review_table.table, Qt.Key.Key_Up)
    assert window.review_table.selected_event().index == 1

    # Typing Space / i / o inside find_edit does NOT trigger playback or trim
    prev_in = window.build_config().trim_start_ms
    prev_out = window.build_config().trim_end_ms
    window.seek_to_ms(window._start_pts_ms + 800)
    window.review_table.open_find_bar(focus_replace=False)
    qtbot.keyClick(window.review_table.find_edit, Qt.Key.Key_Space)
    qtbot.keyClick(window.review_table.find_edit, Qt.Key.Key_I)
    qtbot.keyClick(window.review_table.find_edit, Qt.Key.Key_O)
    assert window._is_playing is False
    assert window.build_config().trim_start_ms == prev_in
    assert window.build_config().trim_end_ms == prev_out
    assert window.review_table.find_edit.text().lower() == " io"
    window.review_table.close_find_bar()

    # Typing Space / i / o inside an active inline cell editor does NOT trigger playback or trim
    item = window.review_table.table.item(0, 4)
    assert item is not None
    window.review_table.table.setCurrentCell(0, 4)
    window.review_table.table.editItem(item)
    assert window.review_table.table.state() == QAbstractItemView.State.EditingState
    editor = window.focusWidget()
    assert editor is not None
    qtbot.keyClick(editor, Qt.Key.Key_Space)
    qtbot.keyClick(editor, Qt.Key.Key_I)
    qtbot.keyClick(editor, Qt.Key.Key_O)
    assert window._is_playing is False
    assert window.build_config().trim_start_ms == prev_in
    assert window.build_config().trim_end_ms == prev_out
    qtbot.keyClick(editor, Qt.Key.Key_Escape)
    window.close()


def test_nudge_uses_exact_frame_pts_ms_and_exports_1_frame_step_on_ntsc(qtbot, tmp_path: Path):
    from transapp.ui.review_table import ReviewTable, parse_subtitle_time

    # 23.976fps exact frame PTS sequence (alternating 41ms / 42ms cadence)
    frame_pts_ms = [int(round(i * 1000.0 * 1001.0 / 24000.0)) for i in range(60)]

    table = ReviewTable()
    qtbot.addWidget(table)
    table.resize(900, 400)
    table.show()

    table.set_frame_step_ms(42)
    table.set_frame_pts_ms(frame_pts_ms)
    table.set_events(
        [
            SubtitleEvent(
                index=1,
                start_ms=frame_pts_ms[10],
                end_ms=frame_pts_ms[22],
                text="精确帧微调",
                confidence=0.95,
                flagged=False,
                thumbnail_png=_make_png_bytes(),
            )
        ]
    )

    def _exported_visible_frames(ass_file: Path) -> list[int]:
        export_ass(
            table.events(),
            ass_file,
            video_width=1920,
            video_height=1080,
            frame_pts_ms=frame_pts_ms,
        )
        dialogue = next(
            line for line in ass_file.read_text(encoding="utf-8").splitlines()
            if line.startswith("Dialogue:")
        )
        parts = dialogue.split(",", 3)
        start_ms_ass = parse_subtitle_time(parts[1])
        end_ms_ass = parse_subtitle_time(parts[2])
        assert start_ms_ass is not None and end_ms_ass is not None
        return [
            idx
            for idx, pts in enumerate(frame_pts_ms)
            if start_ms_ass <= pts < end_ms_ass
        ]

    assert _exported_visible_frames(tmp_path / "step0.ass") == list(range(10, 22))

    # Each +1f nudge shifts both start & end by EXACTLY 1 frame on 23.976fps (no 0-frame stall or 2-frame jump)
    for step in range(1, 13):
        assert table.nudge_selected_timing(+1) is True
        ev = table.events()[0]
        assert (ev.start_ms, ev.end_ms) == (frame_pts_ms[10 + step], frame_pts_ms[22 + step])
        assert _exported_visible_frames(tmp_path / f"step_{step}.ass") == list(
            range(10 + step, 22 + step)
        )

    # -1f steps back by exact 1 frame
    assert table.nudge_selected_timing(-1) is True
    ev = table.events()[0]
    assert (ev.start_ms, ev.end_ms) == (frame_pts_ms[21], frame_pts_ms[33])
    assert _exported_visible_frames(tmp_path / "step_back.ass") == list(range(21, 33))

    # Tail cue stepping past last_pts and back preserves exact 1-frame step
    table.set_events(
        [
            SubtitleEvent(
                index=1,
                start_ms=frame_pts_ms[-3],
                end_ms=frame_pts_ms[-1],
                text="尾帧测试",
                confidence=0.95,
                flagged=False,
                thumbnail_png=_make_png_bytes(),
            )
        ]
    )
    assert table.nudge_selected_timing(+1) is True
    tail_ev = table.events()[0]
    assert tail_ev.start_ms == frame_pts_ms[-2]
    assert tail_ev.end_ms > frame_pts_ms[-1]
    assert table.nudge_selected_timing(-1) is True
    tail_back = table.events()[0]
    assert (tail_back.start_ms, tail_back.end_ms) == (frame_pts_ms[-3], frame_pts_ms[-1])

    # Clearing frame_pts_ms falls back cleanly to delta_frames * frame_step_ms
    table.set_frame_pts_ms(None)
    before_s, before_e = table.events()[0].start_ms, table.events()[0].end_ms
    assert table.nudge_selected_timing(+1) is True
    assert (table.events()[0].start_ms, table.events()[0].end_ms) == (
        before_s + 42,
        before_e + 42,
    )


def test_active_cell_editor_commits_cleanly_before_nofocus_toolbar_ops_and_close(
    qtbot, tmp_path: Path
):
    from PySide6.QtWidgets import QAbstractItemView, QLineEdit
    from transapp.ui.main_window import MainWindow
    from transapp.ui.review_table import ReviewTable

    # Part A: Standalone ReviewTable toolbar buttons commit/close active editor cleanly
    table = ReviewTable()
    qtbot.addWidget(table)
    table.resize(900, 420)
    table.show()
    table.set_frame_step_ms(40)
    thumb = _make_png_bytes()
    table.set_events(
        [
            SubtitleEvent(1, 500, 1200, "OLD_0", 0.95, False, thumb),
            SubtitleEvent(2, 1500, 2200, "OLD_1", 0.95, False, thumb),
        ]
    )

    def _open_editor_and_type(row: int, new_text: str) -> None:
        item = table.table.item(row, 4)
        assert item is not None
        table.table.setCurrentCell(row, 4)
        table.table.editItem(item)
        assert table.table.state() == QAbstractItemView.State.EditingState
        editors = table.table.viewport().findChildren(QLineEdit)
        assert editors
        editors[-1].setText(new_text)

    # 1a. Undo directly while editor has uncommitted text (even with empty stack) commits first, then undoes
    _open_editor_and_type(0, "EDITED")
    assert table.undo() is True
    assert table.table.state() != QAbstractItemView.State.EditingState
    assert [e.text for e in table.events()] == ["OLD_0", "OLD_1"]
    assert table.redo() is True
    assert [e.text for e in table.events()] == ["EDITED", "OLD_1"]

    # 1b. Clicking NoFocus btn_undo while editor is open commits & undoes cleanly back to "EDITED"
    assert table.btn_undo.isEnabled() is True
    _open_editor_and_type(0, "EDITED_TWICE")
    table.btn_undo.click()
    assert table.table.state() != QAbstractItemView.State.EditingState
    assert [e.text for e in table.events()] == ["EDITED", "OLD_1"]

    # 2. Nudge while editor is open commits text and nudges timing without losing edit
    _open_editor_and_type(0, "EDITED_NUDGE")
    table.btn_nudge_fwd.click()
    assert table.table.state() != QAbstractItemView.State.EditingState
    assert table.events()[0].text == "EDITED_NUDGE"
    assert (table.events()[0].start_ms, table.events()[0].end_ms) == (540, 1240)
    assert table.events()[1].text == "OLD_1"

    # 3. Delete while editor is open closes editor and deletes row 0 without corrupting row 1
    _open_editor_and_type(0, "TO_DELETE")
    table.btn_delete_row.click()
    assert table.table.state() != QAbstractItemView.State.EditingState
    assert [e.text for e in table.events()] == ["OLD_1"]

    # Part B: MainWindow closeEvent & export_subtitles commit active cell editor first
    ep = tmp_path / "editor_guard.mp4"
    _make_synthetic_subtitle_mp4(ep, duration_sec=1.2, cues=((200, 900, "原始字幕"),))
    fake_ocr = OcrEngine(
        ocr_callable=lambda _img: (
            [[[[0, 0], [80, 0], [80, 24], [0, 24]], "原始字幕", 0.95]],
            [0.01],
        )
    )
    unsaved_prompts: list[bool] = []
    unsaved_allow = {"value": False}

    def fake_unsaved_confirm() -> bool:
        unsaved_prompts.append(True)
        return unsaved_allow["value"]

    window = MainWindow(
        preset_store=PresetStore(config_dir=tmp_path / "cfg"),
        pipeline=ExtractionPipeline(ocr_engine=fake_ocr),
        unsaved_confirm_fn=fake_unsaved_confirm,
    )
    qtbot.addWidget(window)
    window.show()
    window.load_video(ep)
    window.roi_canvas.set_roi(RoiRect(0.10, 0.72, 0.80, 0.22))
    window.btn_start.click()
    qtbot.waitUntil(lambda: window.btn_export.isEnabled(), timeout=10000)
    assert window._has_unexported_edits is False

    # Open inline editor without pressing Enter, then try to close -> must commit & prompt unsaved guard
    item = window.review_table.table.item(0, 4)
    assert item is not None
    window.review_table.table.setCurrentCell(0, 4)
    window.review_table.table.editItem(item)
    assert window.review_table.table.state() == QAbstractItemView.State.EditingState
    editors = window.review_table.table.viewport().findChildren(QLineEdit)
    assert editors
    editors[-1].setText("未回车修改")

    window.close()
    assert len(unsaved_prompts) == 1
    assert window.isVisible() is True
    assert window.review_table.table.state() != QAbstractItemView.State.EditingState
    assert window.review_table.events()[0].text == "未回车修改"

    # Open inline editor without pressing Enter, then export -> exported file contains committed text
    item = window.review_table.table.item(0, 4)
    assert item is not None
    window.review_table.table.setCurrentCell(0, 4)
    window.review_table.table.editItem(item)
    editors = window.review_table.table.viewport().findChildren(QLineEdit)
    assert editors
    editors[-1].setText("导出前自动提交")

    assert window.export_subtitles() is True
    assert window.review_table.table.state() != QAbstractItemView.State.EditingState
    assert "导出前自动提交" in (tmp_path / "editor_guard.srt").read_text(encoding="utf-8")
    assert window._has_unexported_edits is False
    window.close()


def test_autorepeat_nudge_coalesces_into_single_undo_step_and_undo_to_clean_clears_dirty(
    qtbot, tmp_path: Path
):
    from transapp.ui.main_window import MainWindow

    ep = tmp_path / "coalesce_ep.mp4"
    _make_synthetic_subtitle_mp4(ep, duration_sec=1.4, cues=((200, 1000, "连按微调"),))
    fake_ocr = OcrEngine(
        ocr_callable=lambda _img: (
            [[[[0, 0], [80, 0], [80, 24], [0, 24]], "连按微调", 0.95]],
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
    window.roi_canvas.set_roi(RoiRect(0.10, 0.72, 0.80, 0.22))
    window.btn_start.click()
    qtbot.waitUntil(lambda: window.btn_export.isEnabled(), timeout=10000)

    dirty_events: list[bool] = []
    window.review_table.dirty_changed.connect(dirty_events.append)

    assert window._has_unexported_edits is False
    orig_start = window.review_table.events()[0].start_ms
    orig_end = window.review_table.events()[0].end_ms
    assert len(window.review_table._undo_stack) == 0

    # 10 consecutive +1f nudges within 500ms coalesce into ONE undo snapshot
    for _ in range(10):
        assert window.review_table.nudge_selected_timing(+1) is True

    assert len(window.review_table._undo_stack) == 1
    assert window.review_table.events()[0].start_ms > orig_start
    assert window._has_unexported_edits is True
    assert dirty_events[-1] is True

    # Single Undo restores original timestamps AND clears dirty state back to clean
    assert window.review_table.undo() is True
    assert (
        window.review_table.events()[0].start_ms,
        window.review_table.events()[0].end_ms,
    ) == (orig_start, orig_end)
    assert window._has_unexported_edits is False
    assert dirty_events[-1] is False

    # Redo marks dirty again; exporting marks clean; undoing past exported clean state marks dirty
    assert window.review_table.redo() is True
    assert window._has_unexported_edits is True
    assert window.export_subtitles() is True
    assert window._has_unexported_edits is False
    assert window.review_table.undo() is True
    assert window._has_unexported_edits is True
    assert window.review_table.redo() is True
    assert window._has_unexported_edits is False
    window.close()


def test_review_table_minor_polish_cloning_find_status_multiselection_and_overlap(qtbot):
    from PySide6.QtCore import QItemSelectionModel
    from transapp.ui.review_table import ReviewTable

    table = ReviewTable()
    qtbot.addWidget(table)
    table.resize(960, 440)
    table.show()

    thumb = _make_png_bytes()
    ev0 = SubtitleEvent(1, 500, 1200, "长剑出鞘", 0.96, False, thumb)
    ev1 = SubtitleEvent(2, 1300, 2000, "普通台词", 0.94, False, thumb)
    ev2 = SubtitleEvent(3, 2100, 2800, "断剑重铸", 0.72, True, thumb, "Độ tin cậy thấp")

    # M1: set_events clones SubtitleEvent objects so edits never mutate caller instances
    table.set_events([ev0, ev1, ev2])
    assert table.events()[0] is not ev0
    table.table.item(0, 4).setText("长剑已改")
    assert ev0.text == "长剑出鞘"
    assert table.events()[0].text == "长剑已改"

    # M2: lbl_find_status refreshes when selection changes while find_bar is visible
    table.open_find_bar(focus_replace=False)
    table.find_edit.setText("剑")
    table.table.selectRow(2)
    assert table.lbl_find_status.text() == "2/2"
    table.table.selectRow(1)
    assert table.lbl_find_status.text() == "0/2"
    table.table.selectRow(0)
    assert table.lbl_find_status.text() == "1/2"
    table.close_find_bar()

    # M6: Single-row timing/QA buttons enabled on 1 row, disabled when 2+ rows selected
    single_row_buttons = (
        table.btn_jump_start,
        table.btn_jump_end,
        table.btn_snap_in,
        table.btn_snap_out,
        table.btn_nudge_back,
        table.btn_nudge_fwd,
        table.btn_mark_ok,
    )
    table.table.selectRow(2)
    assert all(btn.isEnabled() for btn in single_row_buttons)

    sel = table.table.selectionModel()
    sel.clearSelection()
    flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
    sel.select(table.table.model().index(0, 0), flags)
    sel.select(table.table.model().index(1, 0), flags)
    table.table.setCurrentCell(1, 0, QItemSelectionModel.SelectionFlag.NoUpdate)
    assert all(not btn.isEnabled() for btn in single_row_buttons)

    # M7: "Trùng giờ" checks true interval intersection (a.start < b.end and b.start < a.end)
    # Out-of-order non-overlapping intervals must NOT trigger false overlap
    table.set_events(
        [
            SubtitleEvent(1, 5000, 6000, "后段", 0.95, False, thumb),
            SubtitleEvent(2, 1000, 2000, "前段", 0.95, False, thumb),
        ]
    )
    table.table.selectRow(0)
    assert "Trùng giờ" not in table.lbl_cue_meta.text()
    table.table.selectRow(1)
    assert "Trùng giờ" not in table.lbl_cue_meta.text()

    # Touching intervals [1000, 2000] and [2000, 3000] do NOT overlap
    table.set_events(
        [
            SubtitleEvent(1, 1000, 2000, "甲", 0.95, False, thumb),
            SubtitleEvent(2, 2000, 3000, "乙", 0.95, False, thumb),
        ]
    )
    table.table.selectRow(0)
    assert "Trùng giờ" not in table.lbl_cue_meta.text()
    table.table.selectRow(1)
    assert "Trùng giờ" not in table.lbl_cue_meta.text()

    # Non-adjacent intersecting intervals [1000, 2000] and [1500, 2500] DO overlap
    table.set_events(
        [
            SubtitleEvent(1, 1000, 2000, "甲", 0.95, False, thumb),
            SubtitleEvent(2, 5000, 6000, "丙", 0.95, False, thumb),
            SubtitleEvent(3, 1500, 2500, "乙", 0.95, False, thumb),
        ]
    )
    table.table.selectRow(0)
    assert "Trùng giờ" in table.lbl_cue_meta.text()
    table.table.selectRow(1)
    assert "Trùng giờ" not in table.lbl_cue_meta.text()
    table.table.selectRow(2)
    assert "Trùng giờ" in table.lbl_cue_meta.text()


def test_playback_and_step_handle_missing_video_file_gracefully(
    qtbot, tmp_path: Path, monkeypatch
):
    from transapp.ui.main_window import MainWindow

    ep = tmp_path / "missing_ep.mp4"
    _make_synthetic_subtitle_mp4(ep, width=640, height=360, fps=25, duration_sec=0.8)

    window = MainWindow(preset_store=PresetStore(config_dir=tmp_path / "cfg"))
    qtbot.addWidget(window)
    window.show()
    window.load_video(ep)

    window._close_preview_reader()

    def raise_missing():
        raise FileNotFoundError("video moved away")

    monkeypatch.setattr(window, "_ensure_preview_reader", raise_missing)

    # start_playback must not raise; stops playback and reports error to status_label
    window.start_playback()
    assert window._is_playing is False
    assert "video moved away" in window.status_label.text()

    # _step_frame(1) must not raise and reports error to status_label
    window.status_label.setText("Sẵn sàng")
    window._step_frame(1)
    assert window._is_playing is False
    assert "video moved away" in window.status_label.text()

    # _step_frame(-1) must not raise and reports error to status_label
    window.status_label.setText("Sẵn sàng")
    window._step_frame(-1)
    assert window._is_playing is False
    assert "video moved away" in window.status_label.text()
    window.close()


def test_step_back_uses_read_prev_frame_before_ms_and_eof_step_fwd_does_not_double_decode(
    qtbot, tmp_path: Path, monkeypatch
):
    from transapp.core.video_reader import VideoReader
    from transapp.ui.main_window import MainWindow

    ep = tmp_path / "vfr_eof_ep.mp4"
    _make_synthetic_subtitle_mp4(ep, width=640, height=360, fps=25, duration_sec=0.4)

    window = MainWindow(preset_store=PresetStore(config_dir=tmp_path / "cfg"))
    qtbot.addWidget(window)
    window.show()
    window.load_video(ep)

    # Advance 2 frames forward
    window._step_frame(1)
    window._step_frame(1)
    before_back_ms = window._current_frame_ms
    assert before_back_ms > window._start_pts_ms

    prev_calls: list[int] = []
    seek_calls: list[int] = []
    iter_calls: list[int] = []
    real_prev = VideoReader.read_prev_frame_before_ms
    real_seek = window.seek_to_ms
    real_iter = VideoReader.iter_full_frames_from_ms

    def spy_prev(self, current_pts_ms: int):
        prev_calls.append(current_pts_ms)
        return real_prev(self, current_pts_ms)

    def spy_seek(target_ms: int):
        seek_calls.append(target_ms)
        return real_seek(target_ms)

    def spy_iter(self, start_raw_ms: int):
        iter_calls.append(start_raw_ms)
        return real_iter(self, start_raw_ms)

    monkeypatch.setattr(VideoReader, "read_prev_frame_before_ms", spy_prev)
    monkeypatch.setattr(window, "seek_to_ms", spy_seek)
    monkeypatch.setattr(VideoReader, "iter_full_frames_from_ms", spy_iter)

    # _step_frame(-1) uses read_prev_frame_before_ms without calling seek_to_ms
    window._step_frame(-1)
    assert prev_calls == [before_back_ms]
    assert seek_calls == []
    assert window._current_frame_ms < before_back_ms
    assert window._at_eof is False

    # Stepping forward through EOF must set _at_eof=True and NEVER fall through to seek_to_ms
    for _ in range(30):
        window._step_frame(1)
        if window._at_eof:
            break
    assert window._at_eof is True
    assert seek_calls == []

    # Pressing +1f again at EOF must do zero extra decodes / seeks
    iter_count_at_eof = len(iter_calls)
    window._step_frame(1)
    assert seek_calls == []
    assert len(iter_calls) == iter_count_at_eof

    # Even when container duration exceeds last video frame PTS (longer audio track),
    # start_playback() while _at_eof=True wraps cleanly back to _start_pts_ms
    window._duration_ms = (window._current_frame_ms - window._start_pts_ms) + 5000
    window.start_playback()
    assert window._is_playing is True
    assert window._at_eof is False
    assert iter_calls[-1] == window._start_pts_ms
    window.stop_playback()
    window.close()


def test_roi_canvas_disabled_during_extraction_and_ass_style_preserved_across_radio_toggle(
    qtbot, tmp_path: Path, monkeypatch
):
    from transapp.ui.main_window import MainWindow

    ep = tmp_path / "lock_style_ep.mp4"
    _make_synthetic_subtitle_mp4(ep, width=640, height=360, fps=25, duration_sec=1.0)

    multi_tpl = tmp_path / "multi_toggle.ass"
    multi_tpl.write_text(
        "[Script Info]\nPlayResX: 1920\nPlayResY: 1080\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize\n"
        "Style: OP_Kara,Microsoft YaHei,48\n"
        "Style: Default,Microsoft YaHei,54\n"
        "Style: SignTop,Microsoft YaHei,44\n\n[Events]\n",
        encoding="utf-8",
    )

    ocr_entered = threading.Event()
    release_ocr = threading.Event()

    def slow_ocr(_img: np.ndarray):
        ocr_entered.set()
        release_ocr.wait(timeout=5.0)
        return ([[[[0, 0], [80, 0], [80, 24], [0, 24]], "锁定画布", 0.95]], [0.01])

    window = MainWindow(
        preset_store=PresetStore(config_dir=tmp_path / "cfg"),
        pipeline=ExtractionPipeline(ocr_engine=OcrEngine(ocr_callable=slow_ocr)),
    )
    qtbot.addWidget(window)
    window.show()
    window.load_video(ep)

    assert window.roi_canvas.isEnabled() is True

    # Start extraction -> RoiCanvas must be disabled while worker is running
    try:
        window.btn_start.click()
        assert ocr_entered.wait(timeout=5.0)
        assert window.roi_canvas.isEnabled() is False
        assert window.btn_start.isEnabled() is False
        assert window.btn_test_frame.isEnabled() is False
    finally:
        release_ocr.set()
    qtbot.waitUntil(lambda: window.btn_start.isEnabled(), timeout=8000)
    assert window.roi_canvas.isEnabled() is True

    # Retiring threads gate btn_start / btn_test_frame / roi_canvas until cleaned up,
    # even if QThread.finished fires while isRunning() is still momentarily True (Qt6 race)
    from PySide6.QtCore import QThread

    dummy_retiring = QThread(window)
    running_checks = {"remaining": 2}
    monkeypatch.setattr(
        dummy_retiring,
        "isRunning",
        lambda: (
            running_checks.__setitem__("remaining", running_checks["remaining"] - 1) or True
            if running_checks["remaining"] > 0
            else False
        ),
    )
    monkeypatch.setattr(dummy_retiring, "wait", lambda *_args: False)
    window._retiring_threads.append(dummy_retiring)
    window.status_label.setText("Đang dọn luồng cũ...")
    window._update_controls_state()
    assert window.btn_start.isEnabled() is False
    assert window.btn_test_frame.isEnabled() is False
    assert window.roi_canvas.isEnabled() is False

    dummy_retiring.finished.connect(window._on_retiring_thread_finished)
    dummy_retiring.finished.emit()
    assert dummy_retiring in window._retiring_threads
    qtbot.waitUntil(lambda: dummy_retiring not in window._retiring_threads, timeout=2000)
    assert window.status_label.text() == "Sẵn sàng"
    assert window.btn_start.isEnabled() is True
    assert window.btn_test_frame.isEnabled() is True
    assert window.roi_canvas.isEnabled() is True

    # M4: Switching File mẫu -> Mặc định -> File mẫu restores previously selected style
    window.radio_ass_template.setChecked(True)
    window.ass_template_edit.setText(str(multi_tpl))
    assert window.ass_style_combo.currentText() == "Default"
    window.ass_style_combo.setCurrentText("SignTop")
    assert window.ass_style_combo.currentText() == "SignTop"

    window.radio_ass_default.setChecked(True)
    assert window.ass_style_row_widget.isVisible() is False

    window.radio_ass_template.setChecked(True)
    assert window.ass_style_row_widget.isVisible() is True
    assert window.ass_style_combo.currentText() == "SignTop"
    window.close()


def test_test_current_frame_runs_asynchronously_and_preset_oserror_warns(
    qtbot, tmp_path: Path, monkeypatch
):
    from PySide6.QtWidgets import QMessageBox
    from transapp.ui.main_window import FrameTestWorker, MainWindow

    ep = tmp_path / "async_frame_ep.mp4"
    _make_synthetic_subtitle_mp4(
        ep,
        width=640,
        height=360,
        fps=25,
        duration_sec=1.0,
        cues=((100, 900, "异步测试"),),
    )

    ocr_started = threading.Event()
    release_ocr = threading.Event()
    ocr_thread_ids: list[int] = []

    def blocking_ocr(_img: np.ndarray):
        ocr_thread_ids.append(threading.get_ident())
        ocr_started.set()
        release_ocr.wait(timeout=5.0)
        return ([[[[0, 0], [80, 0], [80, 24], [0, 24]], "异步测试", 0.96]], [0.01])

    store = PresetStore(config_dir=tmp_path / "cfg")
    pipeline = ExtractionPipeline(ocr_engine=OcrEngine(ocr_callable=blocking_ocr))
    window = MainWindow(preset_store=store, pipeline=pipeline)
    qtbot.addWidget(window)
    window.show()
    window.load_video(ep)
    window.seek_to_ms(400)

    main_tid = threading.get_ident()
    try:
        t0 = time.perf_counter()
        started = window.test_current_frame()
        call_elapsed = time.perf_counter() - t0

        # Must return immediately without blocking UI thread while OCR is paused
        assert started is True
        assert call_elapsed < 0.5
        assert ocr_started.wait(timeout=5.0)
        assert len(ocr_thread_ids) == 1
        assert ocr_thread_ids[0] != main_tid
        assert window.btn_test_frame.isEnabled() is False
        assert window.btn_start.isEnabled() is False
        assert window.roi_canvas.isEnabled() is False
    finally:
        release_ocr.set()
    qtbot.waitUntil(lambda: window.btn_test_frame.isEnabled(), timeout=5000)
    assert "异步测试" in window.test_result_label.text()
    assert "异步测试" in window.status_label.text()
    assert window.roi_canvas._mask_pixmap is not None
    assert window.roi_canvas.isEnabled() is True

    # Verify FrameTestWorker class is QObject-based and stale-sender signals are ignored
    stale = FrameTestWorker(
        pipeline=pipeline,
        frame_bgr=np.zeros((360, 640, 3), dtype=np.uint8),
        frame_index=0,
        pts_ms=0,
        roi=window.roi_canvas.roi(),
        config=window.build_config(),
    )
    stale.failed.connect(window._on_frame_test_failed)
    stale.failed.emit("stale frame error")
    assert "stale frame error" not in window.test_result_label.text()

    # Real FrameTestWorker exception routes to _on_frame_test_failed, reports status, and re-enables UI
    real_test_frame_bgr = pipeline.test_frame_bgr
    monkeypatch.setattr(
        pipeline,
        "test_frame_bgr",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("ocr exploded")),
    )
    assert window.test_current_frame() is True
    qtbot.waitUntil(lambda: window.btn_test_frame.isEnabled(), timeout=5000)
    assert "ocr exploded" in window.status_label.text()
    assert "ocr exploded" in window.test_result_label.text()
    assert window.btn_start.isEnabled() is True
    assert window.roi_canvas.isEnabled() is True
    monkeypatch.setattr(pipeline, "test_frame_bgr", real_test_frame_bgr)

    # M8: PresetStore save/delete OSError shows QMessageBox.warning instead of raising
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda _parent, title, text: warnings.append((title, text)),
    )

    real_save = store.save
    monkeypatch.setattr(store, "save", lambda _preset: (_ for _ in ()).throw(PermissionError("disk full")))
    assert window.save_preset("FailPreset") is False
    assert len(warnings) == 1
    assert "disk full" in warnings[0][1]

    monkeypatch.setattr(store, "save", real_save)
    assert window.save_preset("RealPreset") is True
    monkeypatch.setattr(store, "delete", lambda _name: (_ for _ in ()).throw(OSError("read-only")))
    assert window.delete_current_preset() is False
    assert len(warnings) == 2
    assert "read-only" in warnings[1][1]

    window.close()





