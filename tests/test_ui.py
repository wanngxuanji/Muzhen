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
            frame.pts = pts_ms
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

    # Smoke-test transapp.app.main() without blocking on event loop
    monkeypatch.setattr(QApplication, "exec", lambda _self: 0)
    assert app_mod.main() == 0
