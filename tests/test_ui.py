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

    # Spy on VideoReader.read_frame_at_ms and VideoReader.__init__
    read_calls: list[int] = []
    init_calls: list[Path] = []
    real_read = VideoReader.read_frame_at_ms
    real_init = VideoReader.__init__

    def spy_read(self, target_ms: int):
        read_calls.append(target_ms)
        return real_read(self, target_ms)

    def spy_init(self, path):
        init_calls.append(Path(path))
        return real_init(self, path)

    monkeypatch.setattr(VideoReader, "read_frame_at_ms", spy_read)
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
    window.btn_step_fwd.click()
    assert window._current_frame_ms > 5120
    assert len(read_calls) == 1

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
