"""Main desktop window for single-episode hardsub extraction and export."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import threading

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from transapp.core.models import (
    ExtractConfig,
    ExtractionCancelled,
    ExtractResult,
    FrameTestResult,
    RoiRect,
)
from transapp.core.ocr_engine import OcrEngine
from transapp.core.pipeline import ExtractionPipeline
from transapp.core.presets import PresetStore, RoiPreset
from transapp.core.video_reader import VideoMetadata, VideoReader
from transapp.exporters import export_ass, export_srt, format_ass_time
from transapp.ui.review_table import ReviewTable
from transapp.ui.roi_canvas import RoiCanvas


class _ExtractionWorker(QObject):
    """Runs single-episode ExtractionPipeline inside a dedicated QThread."""

    progress = Signal(str, int, int)
    finished = Signal(object)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(
        self,
        pipeline: ExtractionPipeline,
        video_path: Path,
        roi: RoiRect,
        config: ExtractConfig,
        cancel_event: threading.Event,
    ) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._video_path = video_path
        self._roi = roi
        self._config = config
        self._cancel_event = cancel_event

    @Slot()
    def run(self) -> None:
        try:
            result = self._pipeline.run(
                self._video_path,
                self._roi,
                self._config,
                progress_cb=lambda stage, cur, tot: self.progress.emit(stage, cur, tot),
                cancel_check=self._cancel_event.is_set,
            )
        except ExtractionCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.finished.emit(result)


class MainWindow(QMainWindow):
    """Main desktop window coordinating ROI selection, extraction, review, and export."""

    def __init__(
        self,
        preset_store: PresetStore | None = None,
        pipeline: ExtractionPipeline | None = None,
        overwrite_confirm_fn: Callable[[list[Path]], bool] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._preset_store = preset_store if preset_store is not None else PresetStore()
        self._pipeline = pipeline if pipeline is not None else ExtractionPipeline()
        self._overwrite_confirm_fn = overwrite_confirm_fn

        self._video_path: Path | None = None
        self._video_meta: VideoMetadata | None = None
        self._preview_reader: VideoReader | None = None
        self._current_frame_ms: int = 0
        self._extract_result: ExtractResult | None = None

        self._worker_thread: QThread | None = None
        self._worker: _ExtractionWorker | None = None
        self._cancel_event: threading.Event | None = None

        self.setWindowTitle("TransApp - Tách Sub Raw Tiếng Trung")
        self.resize(1240, 840)

        self._build_ui()
        self._connect_signals()
        self._refresh_presets()
        self._update_roi_label(self.roi_canvas.roi())
        self._update_controls_state()

    @property
    def current_result(self) -> ExtractResult | None:
        """Return the active in-memory extraction result, if any."""
        return self._extract_result

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: Frame preview, ROI canvas, scrubber
        preview_group = QGroupBox("1. Xem trước & chọn vùng ROI")
        preview_layout = QVBoxLayout(preview_group)
        self.roi_canvas = RoiCanvas()
        preview_layout.addWidget(self.roi_canvas, 1)

        scrub_row = QHBoxLayout()
        self.btn_seek_start = QPushButton("|<")
        self.btn_step_back = QPushButton("-1f")
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.btn_step_fwd = QPushButton("+1f")
        self.btn_seek_end = QPushButton(">|")
        for btn in (self.btn_seek_start, self.btn_step_back, self.btn_step_fwd, self.btn_seek_end):
            btn.setFixedWidth(44)
        self.time_label = QLabel("0:00:00.00 / 0:00:00.00")
        scrub_row.addWidget(self.btn_seek_start)
        scrub_row.addWidget(self.btn_step_back)
        scrub_row.addWidget(self.frame_slider, 1)
        scrub_row.addWidget(self.btn_step_fwd)
        scrub_row.addWidget(self.btn_seek_end)
        scrub_row.addWidget(self.time_label)
        preview_layout.addLayout(scrub_row)

        self.roi_label = QLabel()
        preview_layout.addWidget(self.roi_label)

        # Right panel: Config & run controls
        config_group = QGroupBox("2. Cấu hình & chạy tập hiện tại")
        config_layout = QVBoxLayout(config_group)
        config_layout.setSpacing(8)

        # Video file picker
        video_row = QHBoxLayout()
        video_row.addWidget(QLabel("File video:"))
        self.video_path_edit = QLineEdit()
        self.video_path_edit.setReadOnly(True)
        self.video_path_edit.setPlaceholderText("Chưa chọn file video...")
        self.btn_browse_video = QPushButton("Chọn video...")
        video_row.addWidget(self.video_path_edit, 1)
        video_row.addWidget(self.btn_browse_video)
        config_layout.addLayout(video_row)

        # ROI Preset controls
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset ROI:"))
        self.preset_combo = QComboBox()
        self.btn_save_preset = QPushButton("Lưu")
        self.btn_delete_preset = QPushButton("Xóa")
        preset_row.addWidget(self.preset_combo, 1)
        preset_row.addWidget(self.btn_save_preset)
        preset_row.addWidget(self.btn_delete_preset)
        config_layout.addLayout(preset_row)

        # Output directory
        out_dir_group = QGroupBox("Thư mục xuất")
        out_dir_layout = QVBoxLayout(out_dir_group)
        self.radio_dir_same = QRadioButton("Cùng thư mục video")
        self.radio_dir_same.setChecked(True)
        self.radio_dir_custom = QRadioButton("Chọn thư mục khác")
        custom_dir_row = QHBoxLayout()
        self.custom_dir_edit = QLineEdit()
        self.custom_dir_edit.setPlaceholderText("Đường dẫn thư mục xuất...")
        self.btn_browse_out_dir = QPushButton("...")
        self.btn_browse_out_dir.setFixedWidth(36)
        custom_dir_row.addWidget(self.radio_dir_custom)
        custom_dir_row.addWidget(self.custom_dir_edit, 1)
        custom_dir_row.addWidget(self.btn_browse_out_dir)
        out_dir_layout.addWidget(self.radio_dir_same)
        out_dir_layout.addLayout(custom_dir_row)
        config_layout.addWidget(out_dir_group)

        # ASS Style
        ass_group = QGroupBox("Style ASS")
        ass_layout = QVBoxLayout(ass_group)
        self.radio_ass_default = QRadioButton("Mặc định (theo độ phân giải video)")
        self.radio_ass_default.setChecked(True)
        self.radio_ass_template = QRadioButton("File mẫu (.ass)")
        ass_tpl_row = QHBoxLayout()
        self.ass_template_edit = QLineEdit()
        self.ass_template_edit.setPlaceholderText("Đường dẫn file .ass mẫu...")
        self.btn_browse_ass_template = QPushButton("Chọn...")
        ass_tpl_row.addWidget(self.radio_ass_template)
        ass_tpl_row.addWidget(self.ass_template_edit, 1)
        ass_tpl_row.addWidget(self.btn_browse_ass_template)
        ass_layout.addWidget(self.radio_ass_default)
        ass_layout.addLayout(ass_tpl_row)
        config_layout.addWidget(ass_group)

        # Export formats + threshold + OCR provider
        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Xuất ra:"))
        self.chk_ass = QCheckBox(".ass")
        self.chk_ass.setChecked(True)
        self.chk_srt = QCheckBox(".srt")
        self.chk_srt.setChecked(True)
        fmt_row.addWidget(self.chk_ass)
        fmt_row.addWidget(self.chk_srt)
        fmt_row.addSpacing(16)
        fmt_row.addWidget(QLabel("Ngưỡng nghi ngờ OCR:"))
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.0, 1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setValue(0.85)
        fmt_row.addWidget(self.spin_conf)
        fmt_row.addStretch(1)
        config_layout.addLayout(fmt_row)

        provider_row = QHBoxLayout()
        provider_row.addWidget(QLabel("OCR Backend:"))
        self.provider_combo = QComboBox()
        self.provider_combo.addItem("CPU (ONNX Runtime)", "cpu")
        if "directml" in OcrEngine.available_providers():
            self.provider_combo.addItem("DirectML (GPU)", "directml")
        provider_row.addWidget(self.provider_combo)
        provider_row.addStretch(1)
        config_layout.addLayout(provider_row)

        # Test 1 frame
        test_row = QHBoxLayout()
        self.btn_test_frame = QPushButton("Test thử 1 frame hiện tại")
        self.test_result_label = QLabel("")
        self.test_result_label.setWordWrap(True)
        test_row.addWidget(self.btn_test_frame)
        test_row.addWidget(self.test_result_label, 1)
        config_layout.addLayout(test_row)

        # Run / Stop
        run_row = QHBoxLayout()
        self.btn_start = QPushButton("▶ Bắt đầu tách sub")
        self.btn_stop = QPushButton("■ Dừng")
        run_row.addWidget(self.btn_start, 2)
        run_row.addWidget(self.btn_stop, 1)
        config_layout.addLayout(run_row)

        progress_row = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label = QLabel("Sẵn sàng")
        progress_row.addWidget(self.progress_bar, 1)
        progress_row.addWidget(self.status_label)
        config_layout.addLayout(progress_row)
        config_layout.addStretch(1)

        top_splitter.addWidget(preview_group)
        top_splitter.addWidget(config_group)
        top_splitter.setStretchFactor(0, 3)
        top_splitter.setStretchFactor(1, 2)

        # Bottom panel: Review table + export button
        bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(6)
        self.review_table = ReviewTable()
        bottom_layout.addWidget(self.review_table, 1)

        export_row = QHBoxLayout()
        export_row.addStretch(1)
        self.btn_export = QPushButton("Xuất .ass + .srt")
        self.btn_export.setMinimumHeight(34)
        export_row.addWidget(self.btn_export)
        bottom_layout.addLayout(export_row)

        vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        vertical_splitter.addWidget(top_splitter)
        vertical_splitter.addWidget(bottom_panel)
        vertical_splitter.setStretchFactor(0, 3)
        vertical_splitter.setStretchFactor(1, 2)
        main_layout.addWidget(vertical_splitter, 1)

    def _connect_signals(self) -> None:
        self.roi_canvas.roi_changed.connect(self._update_roi_label)
        self.btn_browse_video.clicked.connect(self._browse_video)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_selected)
        self.btn_save_preset.clicked.connect(lambda: self.save_preset())
        self.btn_delete_preset.clicked.connect(self.delete_current_preset)

        self.btn_seek_start.clicked.connect(lambda: self.seek_to_ms(0))
        self.btn_seek_end.clicked.connect(self._seek_to_end)
        self.btn_step_back.clicked.connect(lambda: self._step_frame(-1))
        self.btn_step_fwd.clicked.connect(lambda: self._step_frame(1))
        self.frame_slider.valueChanged.connect(self._on_slider_moved)

        self.radio_dir_same.toggled.connect(lambda _: self._update_controls_state())
        self.radio_ass_default.toggled.connect(lambda _: self._update_controls_state())
        self.btn_browse_out_dir.clicked.connect(self._browse_output_dir)
        self.btn_browse_ass_template.clicked.connect(self._browse_ass_template)

        self.btn_test_frame.clicked.connect(self.test_current_frame)
        self.btn_start.clicked.connect(self.start_extraction)
        self.btn_stop.clicked.connect(self.stop_extraction)
        self.btn_export.clicked.connect(self.export_subtitles)

    def _is_running(self) -> bool:
        return self._worker_thread is not None and self._worker_thread.isRunning()

    def _update_roi_label(self, roi: RoiRect) -> None:
        self.roi_label.setText(
            f"ROI: x={roi.x * 100:.1f}%  y={roi.y * 100:.1f}%  "
            f"w={roi.w * 100:.1f}%  h={roi.h * 100:.1f}%"
        )

    def _update_controls_state(self) -> None:
        running = self._is_running()
        has_video = self._video_path is not None and self._video_meta is not None
        has_result = self._extract_result is not None

        self.btn_browse_video.setEnabled(not running)
        self.preset_combo.setEnabled(not running)
        self.btn_save_preset.setEnabled(not running)
        self.btn_delete_preset.setEnabled(not running and self.preset_combo.currentIndex() > 0)

        self.btn_seek_start.setEnabled(has_video and not running)
        self.btn_step_back.setEnabled(has_video and not running)
        self.frame_slider.setEnabled(has_video and not running)
        self.btn_step_fwd.setEnabled(has_video and not running)
        self.btn_seek_end.setEnabled(has_video and not running)

        custom_dir = self.radio_dir_custom.isChecked()
        self.custom_dir_edit.setEnabled(custom_dir and not running)
        self.btn_browse_out_dir.setEnabled(custom_dir and not running)

        custom_tpl = self.radio_ass_template.isChecked()
        self.ass_template_edit.setEnabled(custom_tpl and not running)
        self.btn_browse_ass_template.setEnabled(custom_tpl and not running)

        self.btn_test_frame.setEnabled(has_video and not running)
        self.btn_start.setEnabled(has_video and not running)
        self.btn_stop.setEnabled(running)
        self.btn_export.setEnabled(has_video and has_result and not running)

    def _refresh_presets(self, select_name: str | None = None) -> None:
        names = self._preset_store.list_names()
        self.preset_combo.blockSignals(True)
        try:
            self.preset_combo.clear()
            self.preset_combo.addItem("(Chưa chọn preset)")
            for name in names:
                self.preset_combo.addItem(name)
            if select_name and select_name in names:
                self.preset_combo.setCurrentText(select_name)
            else:
                self.preset_combo.setCurrentIndex(0)
        finally:
            self.preset_combo.blockSignals(False)
        self._update_controls_state()

    def _on_preset_selected(self, index: int) -> None:
        if index <= 0:
            self._update_controls_state()
            return
        name = self.preset_combo.itemText(index)
        preset = self._preset_store.load(name)
        if preset is None:
            return
        self.roi_canvas.set_roi(preset.roi)
        self.spin_conf.setValue(preset.conf_threshold)
        if preset.ass_template_path:
            self.radio_ass_template.setChecked(True)
            self.ass_template_edit.setText(preset.ass_template_path)
        else:
            self.radio_ass_default.setChecked(True)
            self.ass_template_edit.clear()
        self._update_controls_state()

    def save_preset(self, name: str | None = None) -> bool:
        """Save current ROI, threshold, and optional ASS template path as a named preset."""
        preset_name = name
        if preset_name is None:
            default_name = (
                self.preset_combo.currentText()
                if self.preset_combo.currentIndex() > 0
                else ""
            )
            entered, ok = QInputDialog.getText(
                self,
                "Lưu Preset ROI",
                "Tên preset:",
                text=default_name,
            )
            if not ok:
                return False
            preset_name = entered

        clean_name = preset_name.strip()
        if not clean_name:
            return False

        tpl_path = (
            self.ass_template_edit.text().strip()
            if self.radio_ass_template.isChecked() and self.ass_template_edit.text().strip()
            else None
        )
        preset = RoiPreset(
            name=clean_name,
            roi=self.roi_canvas.roi(),
            ass_template_path=tpl_path,
            conf_threshold=float(self.spin_conf.value()),
        )
        self._preset_store.save(preset)
        self._refresh_presets(select_name=clean_name)
        return True

    def delete_current_preset(self) -> bool:
        """Delete the currently selected ROI preset."""
        if self.preset_combo.currentIndex() <= 0:
            return False
        name = self.preset_combo.currentText()
        deleted = self._preset_store.delete(name)
        self._refresh_presets()
        return deleted

    def _close_preview_reader(self) -> None:
        if self._preview_reader is not None:
            try:
                self._preview_reader.close()
            finally:
                self._preview_reader = None

    def _clear_session_results(self) -> None:
        """Release previous episode's extracted events and thumbnails from memory."""
        self._extract_result = None
        self.review_table.clear()
        self.test_result_label.setText("")

    def load_video(self, video_path: Path | str) -> None:
        """Load a single video episode, clearing any previous episode's memory state."""
        if self._is_running():
            self.stop_extraction()
            if self._worker_thread is not None:
                self._worker_thread.wait(5000)
                self._cleanup_worker()

        self._clear_session_results()
        self.roi_canvas.set_frame(None)
        self._close_preview_reader()

        path = Path(video_path)
        try:
            reader = VideoReader(path)
        except Exception as exc:
            self._video_path = None
            self._video_meta = None
            self.video_path_edit.clear()
            self.frame_slider.setRange(0, 0)
            self.time_label.setText("0:00:00.00 / 0:00:00.00")
            self.status_label.setText("Lỗi mở video")
            self._update_controls_state()
            QMessageBox.warning(self, "Không thể mở video", str(exc))
            return

        self._video_path = path
        self._video_meta = reader.metadata
        self._preview_reader = reader
        self._current_frame_ms = 0

        self.video_path_edit.setText(str(path))
        self.frame_slider.blockSignals(True)
        try:
            self.frame_slider.setRange(0, max(0, reader.metadata.duration_ms))
            self.frame_slider.setValue(0)
        finally:
            self.frame_slider.blockSignals(False)

        self.progress_bar.setValue(0)
        self.status_label.setText("Sẵn sàng")
        self.seek_to_ms(0)
        self._update_controls_state()

    def _browse_video(self) -> None:
        start_dir = str(self._video_path.parent) if self._video_path is not None else ""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn file video",
            start_dir,
            "Video Files (*.mp4 *.mkv *.avi *.mov *.ts *.webm);;All Files (*)",
        )
        if file_path:
            self.load_video(file_path)

    def _browse_output_dir(self) -> None:
        start_dir = self.custom_dir_edit.text().strip() or (
            str(self._video_path.parent) if self._video_path is not None else ""
        )
        chosen = QFileDialog.getExistingDirectory(self, "Chọn thư mục xuất", start_dir)
        if chosen:
            self.radio_dir_custom.setChecked(True)
            self.custom_dir_edit.setText(chosen)

    def _browse_ass_template(self) -> None:
        start_dir = str(self._video_path.parent) if self._video_path is not None else ""
        chosen, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn file .ass mẫu",
            start_dir,
            "ASS Subtitles (*.ass);;All Files (*)",
        )
        if chosen:
            self.radio_ass_template.setChecked(True)
            self.ass_template_edit.setText(chosen)

    def _ensure_preview_reader(self) -> VideoReader | None:
        if self._video_path is None or self._is_running():
            return None
        if self._preview_reader is None:
            self._preview_reader = VideoReader(self._video_path)
        return self._preview_reader

    def seek_to_ms(self, target_ms: int) -> None:
        """Seek preview canvas to the frame nearest ``target_ms``."""
        if self._video_meta is None:
            return
        clamped = max(0, min(int(target_ms), max(0, self._video_meta.duration_ms)))
        reader = self._ensure_preview_reader()
        if reader is None:
            return
        try:
            _idx, pts_ms, full_bgr = reader.read_frame_at_ms(clamped)
        except Exception as exc:
            self.status_label.setText(f"Lỗi seek frame: {exc}")
            return

        self._current_frame_ms = pts_ms
        self.roi_canvas.set_frame(full_bgr)
        self.frame_slider.blockSignals(True)
        try:
            self.frame_slider.setValue(min(pts_ms, self.frame_slider.maximum()))
        finally:
            self.frame_slider.blockSignals(False)
        self.time_label.setText(
            f"{format_ass_time(pts_ms)} / {format_ass_time(self._video_meta.duration_ms)}"
        )

    def _on_slider_moved(self, value_ms: int) -> None:
        self.seek_to_ms(value_ms)

    def _step_frame(self, delta: int) -> None:
        if self._video_meta is None:
            return
        step_ms = max(1, round(1000.0 / max(1.0, self._video_meta.fps)))
        self.seek_to_ms(self._current_frame_ms + delta * step_ms)

    def _seek_to_end(self) -> None:
        if self._video_meta is None:
            return
        step_ms = max(1, round(1000.0 / max(1.0, self._video_meta.fps)))
        self.seek_to_ms(max(0, self._video_meta.duration_ms - step_ms))

    def build_config(self) -> ExtractConfig:
        """Build an ExtractConfig from the current UI controls."""
        provider = self.provider_combo.currentData() or "cpu"
        return ExtractConfig(
            conf_threshold=float(self.spin_conf.value()),
            provider=str(provider),
        )

    def test_current_frame(self) -> FrameTestResult | None:
        """Run stroke-mask + OCR on the currently previewed frame and display overlay."""
        if self._video_path is None or self._is_running():
            return None

        # Close any idle preview reader so only one VideoReader touches the file at a time.
        self._close_preview_reader()
        config = self.build_config()
        roi = self.roi_canvas.roi()

        try:
            result = self._pipeline.test_single_frame(
                self._video_path,
                self._current_frame_ms,
                roi,
                config,
            )
            with VideoReader(self._video_path) as reader:
                _idx, pts_ms, full_bgr = reader.read_frame_at_ms(result.pts_ms)
        except Exception as exc:
            self.test_result_label.setText(f"Lỗi test frame: {exc}")
            return None

        self._current_frame_ms = pts_ms
        self.roi_canvas.set_frame(full_bgr, mask=result.mask)
        if result.text:
            self.test_result_label.setText(
                f'Frame test: "{result.text}" (conf: {result.confidence:.2f})'
            )
        else:
            self.test_result_label.setText("Frame test: (không phát hiện chữ)")
        return result

    def start_extraction(self) -> None:
        """Launch single-episode extraction on a background QThread."""
        if self._video_path is None or self._is_running():
            return

        self._clear_session_results()
        self._close_preview_reader()

        self.progress_bar.setValue(0)
        self.status_label.setText("Đang khởi chạy...")

        cancel_event = threading.Event()
        thread = QThread(self)
        worker = _ExtractionWorker(
            pipeline=self._pipeline,
            video_path=self._video_path,
            roi=self.roi_canvas.roi(),
            config=self.build_config(),
            cancel_event=cancel_event,
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_worker_progress)
        worker.finished.connect(self._on_worker_finished)
        worker.cancelled.connect(self._on_worker_cancelled)
        worker.failed.connect(self._on_worker_failed)

        self._cancel_event = cancel_event
        self._worker_thread = thread
        self._worker = worker
        self._update_controls_state()
        thread.start()

    def stop_extraction(self) -> None:
        """Request cooperative cancellation of the active worker."""
        if self._cancel_event is not None:
            self._cancel_event.set()
            self.btn_stop.setEnabled(False)
            self.status_label.setText("Đang dừng...")

    @Slot(str, int, int)
    def _on_worker_progress(self, stage: str, current: int, total: int) -> None:
        if stage == "scan":
            pct = int(round(70.0 * current / max(1, total)))
            self.progress_bar.setValue(max(0, min(70, pct)))
            self.status_label.setText(
                f"Đang quét frame: {format_ass_time(current)} / {format_ass_time(total)}"
            )
        elif stage == "ocr":
            pct = 70 + int(round(29.0 * current / max(1, total)))
            self.progress_bar.setValue(max(70, min(99, pct)))
            self.status_label.setText(f"Đang OCR: {current}/{total}")
        elif stage == "done":
            self.progress_bar.setValue(100)
            self.status_label.setText(f"Hoàn tất: {total} dòng")

    @Slot(object)
    def _on_worker_finished(self, result: ExtractResult) -> None:
        self._cleanup_worker()
        self._extract_result = result
        self.review_table.set_events(result.events)
        self.progress_bar.setValue(100)
        self.status_label.setText(
            f"Hoàn tất: {len(result.events)} dòng ({result.elapsed_sec:.1f}s)"
        )
        self._update_controls_state()

    @Slot()
    def _on_worker_cancelled(self) -> None:
        self._cleanup_worker()
        self._clear_session_results()
        self.progress_bar.setValue(0)
        self.status_label.setText("Đã hủy")
        self._update_controls_state()

    @Slot(str)
    def _on_worker_failed(self, message: str) -> None:
        self._cleanup_worker()
        self._clear_session_results()
        self.progress_bar.setValue(0)
        self.status_label.setText("Lỗi tách sub")
        self._update_controls_state()
        QMessageBox.critical(self, "Lỗi tách phụ đề", message)

    def _cleanup_worker(self) -> None:
        thread = self._worker_thread
        worker = self._worker
        if worker is not None:
            for sig in (worker.progress, worker.finished, worker.cancelled, worker.failed):
                try:
                    sig.disconnect()
                except (RuntimeError, TypeError):
                    pass
        if thread is not None:
            thread.quit()
            thread.wait(5000)
            thread.deleteLater()
        if worker is not None:
            worker.deleteLater()
        self._worker_thread = None
        self._worker = None
        self._cancel_event = None

    def _confirm_overwrite(self, existing_paths: list[Path]) -> bool:
        if self._overwrite_confirm_fn is not None:
            return bool(self._overwrite_confirm_fn(existing_paths))

        names = "\n".join(f"- {p.name}" for p in existing_paths)
        reply = QMessageBox.question(
            self,
            "Xác nhận ghi đè file phụ đề",
            f"Các file sau đã tồn tại:\n{names}\n\nBạn có muốn ghi đè không?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def export_subtitles(self) -> bool:
        """Export current review events to selected .ass/.srt files with overwrite guard."""
        if self._video_path is None or self._extract_result is None:
            return False

        want_ass = self.chk_ass.isChecked()
        want_srt = self.chk_srt.isChecked()
        if not want_ass and not want_srt:
            QMessageBox.information(
                self,
                "Chưa chọn định dạng",
                "Vui lòng chọn ít nhất một định dạng xuất (.ass hoặc .srt).",
            )
            return False

        if self.radio_dir_same.isChecked():
            out_dir = self._video_path.parent
        else:
            custom = self.custom_dir_edit.text().strip()
            if not custom:
                QMessageBox.warning(self, "Thiếu thư mục xuất", "Vui lòng chọn thư mục xuất.")
                return False
            out_dir = Path(custom)

        template_path: Path | None = None
        if want_ass and self.radio_ass_template.isChecked():
            tpl_text = self.ass_template_edit.text().strip()
            if not tpl_text or not Path(tpl_text).is_file():
                QMessageBox.warning(
                    self,
                    "File mẫu .ass không hợp lệ",
                    "Không tìm thấy file .ass mẫu đã chọn.",
                )
                return False
            template_path = Path(tpl_text)

        stem = self._video_path.stem
        ass_path = out_dir / f"{stem}.ass"
        srt_path = out_dir / f"{stem}.srt"

        targets: list[Path] = []
        if want_ass:
            targets.append(ass_path)
        if want_srt:
            targets.append(srt_path)

        existing = [p for p in targets if p.exists()]
        if existing and not self._confirm_overwrite(existing):
            return False

        events = self.review_table.events()
        if want_ass:
            export_ass(
                events,
                ass_path,
                video_width=self._extract_result.width,
                video_height=self._extract_result.height,
                template_path=template_path,
            )
        if want_srt:
            export_srt(events, srt_path)

        self.status_label.setText(
            "Đã xuất: " + ", ".join(p.name for p in targets)
        )
        return True

    def closeEvent(self, event: QCloseEvent) -> None:
        """Stop background extraction, close open readers, and clear review memory."""
        if self._cancel_event is not None:
            self._cancel_event.set()
        if self._worker_thread is not None:
            self._worker_thread.quit()
            self._worker_thread.wait(5000)
        self._cleanup_worker()
        self._close_preview_reader()
        self._clear_session_results()
        self.roi_canvas.set_frame(None)
        super().closeEvent(event)
