"""Main desktop window for single-episode hardsub extraction and export."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import threading

import numpy as np
from PySide6.QtCore import QObject, QRectF, QSize, Qt, QThread, Signal, Slot
from PySide6.QtGui import QCloseEvent, QColor, QKeyEvent, QLinearGradient, QPainter, QPaintEvent, QPen, QRegion
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStyle,
    QStyleOptionSlider,
    QTextEdit,
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
from transapp.ui.icons import make_icon
from transapp.ui.review_table import ReviewTable
from transapp.ui.roi_canvas import RoiCanvas


def _make_divider() -> QFrame:
    line = QFrame()
    line.setProperty("divider", True)
    return line


def _make_section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("role", "sectionTitle")
    return lbl


class TrimRangeSlider(QSlider):
    """QSlider subclass that highlights the active [In, Out] trim range on the groove."""

    def __init__(
        self,
        orientation: Qt.Orientation = Qt.Orientation.Horizontal,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(orientation, parent)
        self._trim_in_ms: int | None = None
        self._trim_out_ms: int | None = None

    def set_trim_range(self, in_ms: int | None, out_ms: int | None) -> None:
        """Update highlighted [in_ms, out_ms] bounds (0-based ms) and repaint."""
        self._trim_in_ms = int(in_ms) if in_ms is not None else None
        self._trim_out_ms = int(out_ms) if out_ms is not None else None
        self.update()

    def trim_range(self) -> tuple[int | None, int | None]:
        """Return the current (in_ms, out_ms) trim bounds."""
        return (self._trim_in_ms, self._trim_out_ms)

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        if self._trim_in_ms is None and self._trim_out_ms is None:
            return
        min_v = self.minimum()
        max_v = self.maximum()
        if max_v <= min_v:
            return

        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        groove = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            opt,
            QStyle.SubControl.SC_SliderGroove,
            self,
        )
        handle = self.style().subControlRect(
            QStyle.ComplexControl.CC_Slider,
            opt,
            QStyle.SubControl.SC_SliderHandle,
            self,
        )
        if groove.width() <= 0:
            return

        in_val = max(min_v, min(max_v, self._trim_in_ms if self._trim_in_ms is not None else min_v))
        out_val = max(min_v, min(max_v, self._trim_out_ms if self._trim_out_ms is not None else max_v))
        if out_val < in_val:
            return

        span_w = max(1, groove.width() - handle.width())
        origin_x = groove.left() + handle.width() // 2
        x_in = origin_x + QStyle.sliderPositionFromValue(min_v, max_v, in_val, span_w)
        x_out = origin_x + QStyle.sliderPositionFromValue(min_v, max_v, out_val, span_w)
        cy = groove.center().y()

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        clip = QRegion(self.rect()).subtracted(QRegion(handle.adjusted(-1, -1, 1, 1)))
        painter.setClipRegion(clip)

        bar_x = float(min(x_in, x_out))
        bar_w = max(2.0, float(abs(x_out - x_in)))
        grad = QLinearGradient(bar_x, float(cy), bar_x + bar_w, float(cy))
        grad.setColorAt(0.0, QColor("#10B981"))
        grad.setColorAt(1.0, QColor("#2563EB"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(grad)
        painter.drawRoundedRect(QRectF(bar_x, float(cy) - 2.0, bar_w, 4.0), 2.0, 2.0)

        if self._trim_in_ms is not None:
            painter.setPen(QPen(QColor("#10B981"), 2.0))
            painter.drawLine(x_in, cy - 5, x_in, cy + 5)
        if self._trim_out_ms is not None:
            painter.setPen(QPen(QColor("#3B82F6"), 2.0))
            painter.drawLine(x_out, cy - 5, x_out, cy + 5)
        painter.end()


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
        self._start_pts_ms: int = 0
        self._duration_ms: int = 0
        self._current_frame_ms: int = 0
        self._current_frame_index: int = 0
        self._current_frame_bgr: np.ndarray | None = None
        self._extract_result: ExtractResult | None = None
        self._trim_start_ms: int | None = None
        self._trim_end_ms: int | None = None

        self._worker_thread: QThread | None = None
        self._worker: _ExtractionWorker | None = None
        self._cancel_event: threading.Event | None = None
        self._retiring_threads: list[QThread] = []
        self._retiring_workers: dict[QThread, _ExtractionWorker | None] = {}

        self.setWindowTitle("TransApp - Tách Sub Raw Tiếng Trung")
        target_w, target_h = 1160, 680
        screen = self.screen()
        if screen is not None:
            avail = screen.availableGeometry()
            if avail.width() > 0 and avail.height() > 0:
                target_w = min(target_w, avail.width())
                target_h = min(target_h, avail.height())
        self.resize(target_w, target_h)

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

        # === Top Studio Header Bar ===
        top_bar = QFrame()
        top_bar.setObjectName("topBar")
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(12, 7, 12, 7)
        top_bar_layout.setSpacing(8)

        brand_icon = QLabel()
        brand_icon.setPixmap(make_icon("film", color="#3B82F6", size=18).pixmap(18, 18))
        brand_title = QLabel("TransApp")
        brand_title.setObjectName("brandTitle")
        version_badge = QLabel("v1")
        version_badge.setObjectName("versionBadge")
        top_bar_layout.addWidget(brand_icon)
        top_bar_layout.addWidget(brand_title)
        top_bar_layout.addWidget(version_badge)
        top_bar_layout.addSpacing(8)

        self.video_path_edit = QLineEdit()
        self.video_path_edit.setReadOnly(True)
        self.video_path_edit.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.video_path_edit.setPlaceholderText("Chưa chọn file video (.mp4, .mkv, .ts...)")
        self.video_meta_badge = QLabel("")
        self.video_meta_badge.setObjectName("metaBadge")
        self.video_meta_badge.setVisible(False)
        self.btn_browse_video = QPushButton("Chọn video")
        self.btn_browse_video.setIcon(make_icon("folder", size=14))
        self.btn_browse_video.setIconSize(QSize(14, 14))
        self.btn_browse_video.setCursor(Qt.CursorShape.PointingHandCursor)
        top_bar_layout.addWidget(self.video_path_edit, 1)
        top_bar_layout.addWidget(self.video_meta_badge)
        top_bar_layout.addWidget(self.btn_browse_video)

        v_sep = QFrame()
        v_sep.setFixedWidth(1)
        v_sep.setStyleSheet("background-color: #252A34; margin: 3px 2px;")
        top_bar_layout.addWidget(v_sep)

        preset_lbl = QLabel("Preset")
        preset_lbl.setProperty("role", "fieldLabel")
        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(145)
        self.btn_save_preset = QPushButton()
        self.btn_save_preset.setProperty("variant", "tool")
        self.btn_save_preset.setIcon(make_icon("save", size=14))
        self.btn_save_preset.setIconSize(QSize(14, 14))
        self.btn_save_preset.setFixedSize(30, 28)
        self.btn_save_preset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save_preset.setToolTip("Lưu Preset ROI hiện tại")

        self.btn_delete_preset = QPushButton()
        self.btn_delete_preset.setProperty("variant", "danger")
        self.btn_delete_preset.setIcon(make_icon("trash", color="#FCA5A5", size=14))
        self.btn_delete_preset.setIconSize(QSize(14, 14))
        self.btn_delete_preset.setFixedSize(30, 28)
        self.btn_delete_preset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_delete_preset.setToolTip("Xóa Preset đang chọn")

        top_bar_layout.addWidget(preset_lbl)
        top_bar_layout.addWidget(self.preset_combo)
        top_bar_layout.addWidget(self.btn_save_preset)
        top_bar_layout.addWidget(self.btn_delete_preset)
        main_layout.addWidget(top_bar)

        # === Main Workspace: Preview (Left) + Inspector (Right) ===
        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        preview_card = QFrame()
        preview_card.setProperty("card", True)
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(10, 10, 10, 10)
        preview_layout.setSpacing(6)

        self.roi_canvas = RoiCanvas()
        preview_layout.addWidget(self.roi_canvas, 1)

        scrub_row = QHBoxLayout()
        scrub_row.setSpacing(6)
        self.btn_seek_start = QPushButton()
        self.btn_seek_start.setIcon(make_icon("skip_back", size=14))
        self.btn_seek_start.setToolTip("Về đầu video")
        self.btn_step_back = QPushButton()
        self.btn_step_back.setIcon(make_icon("step_back", size=14))
        self.btn_step_back.setToolTip("Lùi 1 frame")
        self.frame_slider = TrimRangeSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setTracking(False)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_step_fwd = QPushButton()
        self.btn_step_fwd.setIcon(make_icon("step_fwd", size=14))
        self.btn_step_fwd.setToolTip("Tiến 1 frame")
        self.btn_seek_end = QPushButton()
        self.btn_seek_end.setIcon(make_icon("skip_fwd", size=14))
        self.btn_seek_end.setToolTip("Tới cuối video")
        for btn in (self.btn_seek_start, self.btn_step_back, self.btn_step_fwd, self.btn_seek_end):
            btn.setProperty("variant", "tool")
            btn.setIconSize(QSize(14, 14))
            btn.setFixedSize(30, 28)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.time_label = QLabel("0:00:00.00 / 0:00:00.00")
        self.time_label.setObjectName("timecodeBadge")
        scrub_row.addWidget(self.btn_seek_start)
        scrub_row.addWidget(self.btn_step_back)
        scrub_row.addWidget(self.frame_slider, 1)
        scrub_row.addWidget(self.btn_step_fwd)
        scrub_row.addWidget(self.btn_seek_end)
        scrub_row.addWidget(self.time_label)
        preview_layout.addLayout(scrub_row)

        trim_row = QHBoxLayout()
        trim_row.setContentsMargins(0, 0, 0, 0)
        trim_row.setSpacing(6)
        self.btn_set_in = QPushButton("In")
        self.btn_set_in.setProperty("variant", "trim")
        self.btn_set_in.setIcon(make_icon("trim_in", color="#10B981", size=12))
        self.btn_set_in.setIconSize(QSize(12, 12))
        self.btn_set_in.setFixedHeight(24)
        self.btn_set_in.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_set_in.setToolTip("Đặt điểm bắt đầu In tại frame hiện tại (I)")

        self.lbl_trim_range = QLabel("Toàn tập")
        self.lbl_trim_range.setObjectName("trimRangeBadge")
        self.lbl_trim_range.setProperty("active", False)
        self.lbl_trim_range.setFixedHeight(24)
        self.lbl_trim_range.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

        self.btn_set_out = QPushButton("Out")
        self.btn_set_out.setProperty("variant", "trim")
        self.btn_set_out.setIcon(make_icon("trim_out", color="#60A5FA", size=12))
        self.btn_set_out.setIconSize(QSize(12, 12))
        self.btn_set_out.setFixedHeight(24)
        self.btn_set_out.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_set_out.setToolTip("Đặt điểm kết thúc Out tại frame hiện tại (O)")

        self.btn_clear_trim = QPushButton("Xóa trim")
        self.btn_clear_trim.setProperty("variant", "trim")
        self.btn_clear_trim.setIcon(make_icon("x_mark", color="#9BA4B5", size=11))
        self.btn_clear_trim.setIconSize(QSize(11, 11))
        self.btn_clear_trim.setFixedHeight(24)
        self.btn_clear_trim.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear_trim.setToolTip("Xóa giới hạn In/Out để quét toàn tập")

        trim_row.addWidget(self.btn_set_in)
        trim_row.addWidget(self.lbl_trim_range)
        trim_row.addWidget(self.btn_set_out)
        trim_row.addWidget(self.btn_clear_trim)
        trim_row.addStretch(1)
        preview_layout.addLayout(trim_row)

        roi_bar = QHBoxLayout()
        roi_bar.setSpacing(8)
        self.roi_label = QLabel()
        self.roi_label.setObjectName("roiPill")
        self.test_result_label = QLabel("")
        self.test_result_label.setObjectName("testResultLabel")
        self.test_result_label.setWordWrap(False)
        self.btn_test_frame = QPushButton("Test 1 frame")
        self.btn_test_frame.setIcon(make_icon("scan", color="#10B981", size=14))
        self.btn_test_frame.setIconSize(QSize(14, 14))
        self.btn_test_frame.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_test_frame.setToolTip("Chạy thử bộ lọc chữ & OCR trên khung hình hiện tại")
        roi_bar.addWidget(self.roi_label)
        roi_bar.addWidget(self.test_result_label, 1)
        roi_bar.addWidget(self.btn_test_frame)
        preview_layout.addLayout(roi_bar)

        # Right panel: Flat Studio Inspector Card
        inspector_card = QFrame()
        inspector_card.setProperty("card", True)
        inspector_card.setMinimumWidth(330)
        inspector_outer = QVBoxLayout(inspector_card)
        inspector_outer.setContentsMargins(12, 10, 12, 10)
        inspector_outer.setSpacing(8)

        scroll = QScrollArea()
        scroll.setObjectName("inspectorScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inspector_body = QWidget()
        body_layout = QVBoxLayout(inspector_body)
        body_layout.setContentsMargins(0, 0, 2, 0)
        body_layout.setSpacing(6)

        # Section 1: OCR
        body_layout.addWidget(_make_section_label("NHẬN DIỆN OCR"))
        ocr_grid = QGridLayout()
        ocr_grid.setContentsMargins(0, 0, 0, 0)
        ocr_grid.setHorizontalSpacing(10)
        ocr_grid.setVerticalSpacing(6)

        lbl_backend = QLabel("Backend")
        lbl_backend.setProperty("role", "fieldLabel")
        self.provider_combo = QComboBox()
        self.provider_combo.addItem("CPU (ONNX Runtime)", "cpu")
        if "directml" in OcrEngine.available_providers():
            self.provider_combo.addItem("DirectML (GPU)", "directml")
            self.provider_combo.setCurrentIndex(1)

        lbl_model = QLabel("Model OCR")
        lbl_model.setProperty("role", "fieldLabel")
        self.ocr_model_combo = QComboBox()
        self.ocr_model_combo.addItem("PP-OCRv5 Chuẩn xác (Server)", "accurate")
        self.ocr_model_combo.addItem("PP-OCRv5 Nhanh (Mobile)", "fast")
        self.ocr_model_combo.setToolTip(
            "Server (mặc định): chuyên tiếng Trung, chính xác tối đa + bầu chọn đa frame. "
            "Mobile: nhanh hơn, nhẹ hơn."
        )

        lbl_conf = QLabel("Ngưỡng nghi ngờ")
        lbl_conf.setProperty("role", "fieldLabel")
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.0, 1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setValue(0.85)

        ocr_grid.addWidget(lbl_backend, 0, 0)
        ocr_grid.addWidget(self.provider_combo, 0, 1)
        ocr_grid.addWidget(lbl_model, 1, 0)
        ocr_grid.addWidget(self.ocr_model_combo, 1, 1)
        ocr_grid.addWidget(lbl_conf, 2, 0)
        ocr_grid.addWidget(self.spin_conf, 2, 1)
        ocr_grid.setColumnStretch(1, 1)
        body_layout.addLayout(ocr_grid)

        body_layout.addWidget(_make_divider())

        # Section 2: Export settings
        body_layout.addWidget(_make_section_label("XUẤT PHỤ ĐỀ"))
        fmt_row = QHBoxLayout()
        fmt_row.setSpacing(12)
        lbl_fmt = QLabel("Định dạng")
        lbl_fmt.setProperty("role", "fieldLabel")
        self.chk_ass = QCheckBox(".ass (Aegisub)")
        self.chk_ass.setChecked(True)
        self.chk_ass.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chk_srt = QCheckBox(".srt (Premiere)")
        self.chk_srt.setChecked(True)
        self.chk_srt.setCursor(Qt.CursorShape.PointingHandCursor)
        fmt_row.addWidget(lbl_fmt)
        fmt_row.addWidget(self.chk_ass)
        fmt_row.addWidget(self.chk_srt)
        fmt_row.addStretch(1)
        body_layout.addLayout(fmt_row)

        dir_head = QHBoxLayout()
        dir_head.setSpacing(10)
        lbl_dir = QLabel("Thư mục")
        lbl_dir.setProperty("role", "fieldLabel")
        self.radio_dir_same = QRadioButton("Cùng video")
        self.radio_dir_same.setChecked(True)
        self.radio_dir_same.setCursor(Qt.CursorShape.PointingHandCursor)
        self.radio_dir_custom = QRadioButton("Khác")
        self.radio_dir_custom.setCursor(Qt.CursorShape.PointingHandCursor)
        self._dir_radio_group = QButtonGroup(self)
        self._dir_radio_group.addButton(self.radio_dir_same)
        self._dir_radio_group.addButton(self.radio_dir_custom)
        dir_head.addWidget(lbl_dir)
        dir_head.addWidget(self.radio_dir_same)
        dir_head.addWidget(self.radio_dir_custom)
        dir_head.addStretch(1)
        body_layout.addLayout(dir_head)

        custom_dir_row = QHBoxLayout()
        custom_dir_row.setSpacing(6)
        self.custom_dir_edit = QLineEdit()
        self.custom_dir_edit.setPlaceholderText("Đường dẫn thư mục xuất...")
        self.btn_browse_out_dir = QPushButton()
        self.btn_browse_out_dir.setProperty("variant", "tool")
        self.btn_browse_out_dir.setIcon(make_icon("folder", size=14))
        self.btn_browse_out_dir.setIconSize(QSize(14, 14))
        self.btn_browse_out_dir.setFixedSize(28, 26)
        self.btn_browse_out_dir.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_browse_out_dir.setToolTip("Chọn thư mục xuất")
        custom_dir_row.addWidget(self.custom_dir_edit, 1)
        custom_dir_row.addWidget(self.btn_browse_out_dir)
        body_layout.addLayout(custom_dir_row)

        ass_head = QHBoxLayout()
        ass_head.setSpacing(10)
        lbl_ass = QLabel("Style ASS")
        lbl_ass.setProperty("role", "fieldLabel")
        self.radio_ass_default = QRadioButton("Mặc định")
        self.radio_ass_default.setChecked(True)
        self.radio_ass_default.setCursor(Qt.CursorShape.PointingHandCursor)
        self.radio_ass_template = QRadioButton("File mẫu")
        self.radio_ass_template.setCursor(Qt.CursorShape.PointingHandCursor)
        self._ass_radio_group = QButtonGroup(self)
        self._ass_radio_group.addButton(self.radio_ass_default)
        self._ass_radio_group.addButton(self.radio_ass_template)
        ass_head.addWidget(lbl_ass)
        ass_head.addWidget(self.radio_ass_default)
        ass_head.addWidget(self.radio_ass_template)
        ass_head.addStretch(1)
        body_layout.addLayout(ass_head)

        ass_tpl_row = QHBoxLayout()
        ass_tpl_row.setSpacing(6)
        self.ass_template_edit = QLineEdit()
        self.ass_template_edit.setPlaceholderText("Đường dẫn file .ass mẫu...")
        self.btn_browse_ass_template = QPushButton()
        self.btn_browse_ass_template.setProperty("variant", "tool")
        self.btn_browse_ass_template.setIcon(make_icon("folder", size=14))
        self.btn_browse_ass_template.setIconSize(QSize(14, 14))
        self.btn_browse_ass_template.setFixedSize(28, 26)
        self.btn_browse_ass_template.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_browse_ass_template.setToolTip("Chọn file .ass mẫu")
        ass_tpl_row.addWidget(self.ass_template_edit, 1)
        ass_tpl_row.addWidget(self.btn_browse_ass_template)
        body_layout.addLayout(ass_tpl_row)

        body_layout.addStretch(1)
        scroll.setWidget(inspector_body)
        inspector_outer.addWidget(scroll, 1)

        # Anchored Run/Stop footer inside Inspector
        inspector_outer.addWidget(_make_divider())

        run_row = QHBoxLayout()
        run_row.setSpacing(8)
        self.btn_start = QPushButton("Bắt đầu tách sub")
        self.btn_start.setProperty("variant", "primary")
        self.btn_start.setIcon(make_icon("play", color="#FFFFFF", size=14))
        self.btn_start.setIconSize(QSize(14, 14))
        self.btn_start.setMinimumHeight(34)
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)

        self.btn_stop = QPushButton("Dừng")
        self.btn_stop.setProperty("variant", "danger")
        self.btn_stop.setIcon(make_icon("stop", color="#FCA5A5", size=13))
        self.btn_stop.setIconSize(QSize(13, 13))
        self.btn_stop.setMinimumHeight(34)
        self.btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)

        run_row.addWidget(self.btn_start, 3)
        run_row.addWidget(self.btn_stop, 1)
        inspector_outer.addLayout(run_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label = QLabel("Sẵn sàng")
        self.status_label.setObjectName("statusLabel")
        inspector_outer.addWidget(self.progress_bar)
        inspector_outer.addWidget(self.status_label)

        top_splitter.addWidget(preview_card)
        top_splitter.addWidget(inspector_card)
        top_splitter.setStretchFactor(0, 18)
        top_splitter.setStretchFactor(1, 10)

        # === Bottom Dock: Review Table + Primary Export Button ===
        review_card = QFrame()
        review_card.setProperty("card", True)
        review_layout = QVBoxLayout(review_card)
        review_layout.setContentsMargins(10, 8, 10, 8)
        review_layout.setSpacing(0)

        self.review_table = ReviewTable()
        self.btn_export = QPushButton("Xuất .ass + .srt")
        self.btn_export.setProperty("variant", "primary")
        self.btn_export.setIcon(make_icon("export", color="#FFFFFF", size=14))
        self.btn_export.setIconSize(QSize(14, 14))
        self.btn_export.setCursor(Qt.CursorShape.PointingHandCursor)
        self.review_table.add_header_widget(self.btn_export)
        review_layout.addWidget(self.review_table, 1)

        vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        vertical_splitter.addWidget(top_splitter)
        vertical_splitter.addWidget(review_card)
        vertical_splitter.setStretchFactor(0, 13)
        vertical_splitter.setStretchFactor(1, 10)
        main_layout.addWidget(vertical_splitter, 1)

    def _connect_signals(self) -> None:
        self.roi_canvas.roi_changed.connect(self._update_roi_label)
        self.btn_browse_video.clicked.connect(self._browse_video)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_selected)
        self.btn_save_preset.clicked.connect(lambda: self.save_preset())
        self.btn_delete_preset.clicked.connect(self.delete_current_preset)

        self.btn_seek_start.clicked.connect(self._seek_to_start)
        self.btn_seek_end.clicked.connect(self._seek_to_end)
        self.btn_step_back.clicked.connect(lambda: self._step_frame(-1))
        self.btn_step_fwd.clicked.connect(lambda: self._step_frame(1))
        self.frame_slider.sliderMoved.connect(self._on_slider_moved)
        self.frame_slider.valueChanged.connect(self._on_slider_value_changed)
        self.btn_set_in.clicked.connect(lambda: self.set_trim_in())
        self.btn_set_out.clicked.connect(lambda: self.set_trim_out())
        self.btn_clear_trim.clicked.connect(self.clear_trim)

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
        if self._video_meta is not None:
            x1, y1, x2, y2 = roi.to_pixels(self._video_meta.width, self._video_meta.height)
            self.roi_label.setText(
                f"ROI: {x2 - x1}×{y2 - y1} px ({roi.x * 100:.1f}%, {roi.y * 100:.1f}%)"
            )
            return
        self.roi_label.setText(
            f"ROI {roi.x * 100:.1f}%, {roi.y * 100:.1f}% · "
            f"{roi.w * 100:.1f}×{roi.h * 100:.1f}%"
        )

    def _current_rel_ms(self) -> int:
        return max(0, int(self._current_frame_ms - self._start_pts_ms))

    def _update_trim_display(self) -> None:
        self.frame_slider.set_trim_range(self._trim_start_ms, self._trim_end_ms)
        has_trim = self._trim_start_ms is not None or self._trim_end_ms is not None
        if not has_trim:
            self.lbl_trim_range.setText("Toàn tập")
        else:
            in_disp = self._trim_start_ms if self._trim_start_ms is not None else 0
            out_disp = self._trim_end_ms if self._trim_end_ms is not None else self._duration_ms
            self.lbl_trim_range.setText(
                f"{format_ass_time(in_disp)} → {format_ass_time(out_disp)}"
            )
        if bool(self.lbl_trim_range.property("active")) != has_trim:
            self.lbl_trim_range.setProperty("active", has_trim)
            style = self.lbl_trim_range.style()
            if style is not None:
                style.unpolish(self.lbl_trim_range)
                style.polish(self.lbl_trim_range)

    def set_trim_in(self, rel_ms: int | None = None) -> None:
        """Set 0-based In trim mark (defaults to current preview position)."""
        if self._video_meta is None or self._is_running():
            return
        val = (
            self._current_rel_ms()
            if rel_ms is None
            else max(0, min(int(rel_ms), self._duration_ms))
        )
        self._trim_start_ms = val
        if self._trim_end_ms is not None and self._trim_start_ms >= self._trim_end_ms:
            self._trim_end_ms = None
            self.status_label.setText("Đã đặt In (tự xóa Out vì In ≥ Out)")
        self._update_trim_display()
        self._update_controls_state()

    def set_trim_out(self, rel_ms: int | None = None) -> None:
        """Set 0-based Out trim mark (defaults to current preview position)."""
        if self._video_meta is None or self._is_running():
            return
        val = (
            self._current_rel_ms()
            if rel_ms is None
            else max(0, min(int(rel_ms), self._duration_ms))
        )
        if val <= 0:
            self.status_label.setText("Điểm Out phải lớn hơn 0:00:00.00")
            return
        self._trim_end_ms = val
        if self._trim_start_ms is not None and self._trim_start_ms >= self._trim_end_ms:
            self._trim_start_ms = None
            self.status_label.setText("Đã đặt Out (tự xóa In vì In ≥ Out)")
        self._update_trim_display()
        self._update_controls_state()

    def clear_trim(self) -> None:
        """Reset OP/ED trim marks back to full video range."""
        self._trim_start_ms = None
        self._trim_end_ms = None
        self._update_trim_display()
        self._update_controls_state()

    @staticmethod
    def _is_text_input_focused() -> bool:
        focus = QApplication.focusWidget()
        if focus is None:
            return False
        if isinstance(focus, (QLineEdit, QPlainTextEdit, QTextEdit, QAbstractSpinBox, QComboBox)):
            return True
        if (
            isinstance(focus, QAbstractItemView)
            and focus.state() == QAbstractItemView.State.EditingState
        ):
            return True
        return False

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if (
            event.modifiers() == Qt.KeyboardModifier.NoModifier
            and not self._is_text_input_focused()
        ):
            if event.key() == Qt.Key.Key_I:
                self.set_trim_in()
                event.accept()
                return
            if event.key() == Qt.Key.Key_O:
                self.set_trim_out()
                event.accept()
                return
        super().keyPressEvent(event)

    def _update_controls_state(self) -> None:
        running = self._is_running()
        has_video = self._video_path is not None and self._video_meta is not None
        has_result = self._extract_result is not None
        has_trim = self._trim_start_ms is not None or self._trim_end_ms is not None

        self.btn_browse_video.setEnabled(not running)
        self.preset_combo.setEnabled(not running)
        self.btn_save_preset.setEnabled(not running)
        self.btn_delete_preset.setEnabled(not running and self.preset_combo.currentIndex() > 0)

        self.btn_seek_start.setEnabled(has_video and not running)
        self.btn_step_back.setEnabled(has_video and not running)
        self.frame_slider.setEnabled(has_video and not running)
        self.btn_step_fwd.setEnabled(has_video and not running)
        self.btn_seek_end.setEnabled(has_video and not running)
        self.btn_set_in.setEnabled(has_video and not running)
        self.btn_set_out.setEnabled(has_video and not running)
        self.btn_clear_trim.setEnabled(has_video and has_trim and not running)

        custom_dir = self.radio_dir_custom.isChecked()
        self.custom_dir_edit.setEnabled(custom_dir and not running)
        self.btn_browse_out_dir.setEnabled(not running)

        custom_tpl = self.radio_ass_template.isChecked()
        self.ass_template_edit.setEnabled(custom_tpl and not running)
        self.btn_browse_ass_template.setEnabled(not running)

        self.btn_test_frame.setEnabled(has_video and not running)
        self.provider_combo.setEnabled(not running)
        self.ocr_model_combo.setEnabled(not running)
        self.spin_conf.setEnabled(not running)
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
        self._stop_and_wait_worker()

        self._clear_session_results()
        self._trim_start_ms = None
        self._trim_end_ms = None
        self._update_trim_display()
        self._current_frame_bgr = None
        self._current_frame_index = 0
        self.roi_canvas.set_frame(None)
        self._close_preview_reader()

        path = Path(video_path)
        try:
            reader = VideoReader(path)
        except Exception as exc:
            self._video_path = None
            self._video_meta = None
            self._start_pts_ms = 0
            self._duration_ms = 0
            self._current_frame_ms = 0
            self.video_path_edit.clear()
            self.video_meta_badge.clear()
            self.video_meta_badge.setVisible(False)
            self.frame_slider.setRange(0, 0)
            self.time_label.setText("0:00:00.00 / 0:00:00.00")
            self.status_label.setText("Lỗi mở video")
            self._update_roi_label(self.roi_canvas.roi())
            self._update_controls_state()
            QMessageBox.warning(self, "Không thể mở video", str(exc))
            return

        self._video_path = path
        self._video_meta = reader.metadata
        self._preview_reader = reader
        self._start_pts_ms = int(reader.metadata.start_pts_ms)
        self._duration_ms = max(0, reader.metadata.duration_ms)
        self._current_frame_ms = self._start_pts_ms
        self._current_frame_index = 0

        self.video_path_edit.setText(str(path))
        fps_val = reader.metadata.fps
        fps_str = f"{fps_val:.0f}" if abs(fps_val - round(fps_val)) < 0.05 else f"{fps_val:.2f}"
        self.video_meta_badge.setText(
            f"{reader.metadata.width}×{reader.metadata.height} · {fps_str} fps"
        )
        self.video_meta_badge.setVisible(True)
        self.frame_slider.blockSignals(True)
        try:
            self.frame_slider.setRange(0, self._duration_ms)
            self.frame_slider.setValue(0)
        finally:
            self.frame_slider.blockSignals(False)

        self.progress_bar.setValue(0)
        self.status_label.setText("Sẵn sàng")
        self._update_roi_label(self.roi_canvas.roi())
        self.seek_to_ms(self._start_pts_ms)
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
        min_ms = self._start_pts_ms
        max_ms = self._start_pts_ms + self._duration_ms
        clamped = max(min_ms, min(int(target_ms), max_ms))
        try:
            reader = self._ensure_preview_reader()
            if reader is None:
                return
            frame_idx, pts_ms, full_bgr = reader.read_frame_at_ms(clamped)
        except Exception as exc:
            self.status_label.setText(f"Lỗi seek frame: {exc}")
            return

        self._current_frame_index = frame_idx
        self._current_frame_ms = pts_ms
        self._current_frame_bgr = full_bgr
        self.roi_canvas.set_frame(full_bgr)
        self.frame_slider.blockSignals(True)
        try:
            offset_ms = max(0, min(pts_ms - self._start_pts_ms, self.frame_slider.maximum()))
            self.frame_slider.setValue(offset_ms)
        finally:
            self.frame_slider.blockSignals(False)
        rel_ms = max(0, pts_ms - self._start_pts_ms)
        self.time_label.setText(
            f"{format_ass_time(rel_ms)} / {format_ass_time(self._duration_ms)}"
        )

    def _on_slider_moved(self, offset_ms: int) -> None:
        if self._video_meta is None:
            return
        clamped = max(0, min(int(offset_ms), self._duration_ms))
        self.time_label.setText(
            f"{format_ass_time(clamped)} / {format_ass_time(self._duration_ms)}"
        )

    def _on_slider_value_changed(self, offset_ms: int) -> None:
        if self._video_meta is None:
            return
        self.seek_to_ms(self._start_pts_ms + int(offset_ms))

    def _step_frame(self, delta: int) -> None:
        if self._video_meta is None:
            return
        step_ms = max(1, round(1000.0 / max(1.0, self._video_meta.fps)))
        self.seek_to_ms(self._current_frame_ms + delta * step_ms)

    def _seek_to_start(self) -> None:
        if self._video_meta is None:
            return
        self.seek_to_ms(self._start_pts_ms)

    def _seek_to_end(self) -> None:
        if self._video_meta is None:
            return
        self.seek_to_ms(self._start_pts_ms + self._duration_ms)

    def build_config(self) -> ExtractConfig:
        """Build an ExtractConfig from the current UI controls."""
        provider = self.provider_combo.currentData() or "cpu"
        ocr_model = self.ocr_model_combo.currentData() or "accurate"
        return ExtractConfig(
            conf_threshold=float(self.spin_conf.value()),
            provider=str(provider),
            ocr_model=str(ocr_model),
            trim_start_ms=self._trim_start_ms,
            trim_end_ms=self._trim_end_ms,
        )

    def test_current_frame(self) -> FrameTestResult | None:
        """Run stroke-mask + OCR on the currently previewed frame and display overlay."""
        if self._video_path is None or self._is_running():
            return None

        config = self.build_config()
        roi = self.roi_canvas.roi()

        try:
            full_bgr = self._current_frame_bgr
            if full_bgr is None:
                reader = self._ensure_preview_reader()
                if reader is None:
                    return None
                frame_idx, pts_ms, full_bgr = reader.read_frame_at_ms(self._current_frame_ms)
                self._current_frame_index = frame_idx
                self._current_frame_ms = pts_ms
                self._current_frame_bgr = full_bgr
            result = self._pipeline.test_frame_bgr(
                full_bgr,
                self._current_frame_index,
                self._current_frame_ms,
                roi,
                config,
            )
        except Exception as exc:
            self.test_result_label.setText(f"Lỗi test frame: {exc}")
            return None

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
        worker.finished.connect(thread.quit)
        worker.cancelled.connect(self._on_worker_cancelled)
        worker.cancelled.connect(thread.quit)
        worker.failed.connect(self._on_worker_failed)
        worker.failed.connect(thread.quit)

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
        if self.sender() is not self._worker:
            return
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
        if self.sender() is not self._worker:
            return
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
        if self.sender() is not self._worker:
            return
        self._cleanup_worker()
        self._clear_session_results()
        self.progress_bar.setValue(0)
        self.status_label.setText("Đã hủy")
        self._update_controls_state()

    @Slot(str)
    def _on_worker_failed(self, message: str) -> None:
        if self.sender() is not self._worker:
            return
        self._cleanup_worker()
        self._clear_session_results()
        self.progress_bar.setValue(0)
        self.status_label.setText("Lỗi tách sub")
        self._update_controls_state()
        QMessageBox.critical(self, "Lỗi tách phụ đề", message)

    def _stop_and_wait_worker(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
        self._cleanup_worker()

    def _disconnect_worker_slots(self, worker: _ExtractionWorker) -> None:
        for sig, slot in (
            (worker.progress, self._on_worker_progress),
            (worker.finished, self._on_worker_finished),
            (worker.cancelled, self._on_worker_cancelled),
            (worker.failed, self._on_worker_failed),
        ):
            try:
                sig.disconnect(slot)
            except (RuntimeError, TypeError):
                pass

    def _retire_thread(self, thread: QThread) -> None:
        if thread not in self._retiring_threads or thread.isRunning():
            return
        self._retiring_threads.remove(thread)
        worker = self._retiring_workers.pop(thread, None)
        if worker is not None:
            worker.deleteLater()
        thread.deleteLater()

    @Slot()
    def _on_retiring_thread_finished(self) -> None:
        sender = self.sender()
        if isinstance(sender, QThread):
            self._retire_thread(sender)

    def _cleanup_worker(self) -> None:
        thread = self._worker_thread
        worker = self._worker
        self._worker_thread = None
        self._worker = None
        self._cancel_event = None

        if worker is not None:
            self._disconnect_worker_slots(worker)

        if thread is not None:
            thread.quit()
            finished = thread.wait(5000)
            if finished and not thread.isRunning():
                if worker is not None:
                    worker.deleteLater()
                thread.deleteLater()
            else:
                self._retiring_threads.append(thread)
                self._retiring_workers[thread] = worker
                thread.finished.connect(self._on_retiring_thread_finished)
                if thread.isFinished() and not thread.isRunning():
                    self._retire_thread(thread)
        elif worker is not None:
            worker.deleteLater()

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
        try:
            if want_ass:
                export_ass(
                    events,
                    ass_path,
                    video_width=self._extract_result.width,
                    video_height=self._extract_result.height,
                    template_path=template_path,
                    frame_pts_ms=self._extract_result.frame_pts_ms,
                )
            if want_srt:
                export_srt(events, srt_path)
        except Exception as exc:
            self.status_label.setText(f"Lỗi xuất file: {exc}")
            QMessageBox.warning(self, "Lỗi xuất phụ đề", str(exc))
            return False

        self.status_label.setText(
            "Đã xuất: " + ", ".join(p.name for p in targets)
        )
        return True

    def closeEvent(self, event: QCloseEvent) -> None:
        """Stop background extraction, close open readers, and clear review memory."""
        self._stop_and_wait_worker()
        for retiring in list(self._retiring_threads):
            retiring.quit()
            retiring.wait()
            self._retire_thread(retiring)
        self._close_preview_reader()
        self._clear_session_results()
        self._current_frame_bgr = None
        self.roi_canvas.set_frame(None)
        super().closeEvent(event)
