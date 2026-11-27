"""Main desktop window for single-episode hardsub extraction and export."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import os
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
from PySide6.QtCore import QEvent, QObject, QRectF, QSize, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QKeyEvent,
    QLinearGradient,
    QPainter,
    QPaintEvent,
    QPen,
    QRegion,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStyle,
    QStyleOptionSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from muzhen import __version__
from muzhen.core.audio_player import AudioPreviewPlayer
from muzhen.core.discord_rpc import DiscordRpcClient, PresenceState
from muzhen.core.models import (
    ExtractConfig,
    ExtractionCancelled,
    ExtractResult,
    FrameTestResult,
    RoiRect,
)
from muzhen.core.ocr_engine import OcrEngine
from muzhen.core.pipeline import ExtractionPipeline
from muzhen.core.presets import PresetStore, RoiPreset
from muzhen.core.updater import GITHUB_REPO, UpdateInfo, UpdatePrefsStore, check_for_updates
from muzhen.core.video_reader import VideoMetadata, VideoReader
from muzhen.exporters import (
    export_ass,
    export_srt,
    export_txt,
    export_vtt,
    format_ass_time,
    prepare_export_cues,
)
from muzhen.ui.i18n import LANGUAGE_CHOICES, resolve_language, tr
from muzhen.ui.icons import make_icon
from muzhen.ui.review_table import ReviewTable, _CompactButton, _ShrinkableLabel
from muzhen.ui.roi_canvas import RoiCanvas, extract_dropped_video_path

GITHUB_REPO_URL = f"https://github.com/{GITHUB_REPO}"


def _make_divider() -> QFrame:
    line = QFrame()
    line.setProperty("divider", True)
    return line


def _make_section_label(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setProperty("role", "sectionTitle")
    return lbl


def _make_panel_header(title_text: str) -> tuple[QFrame, QHBoxLayout, QLabel]:
    header = QFrame()
    header.setObjectName("panelHeader")
    layout = QHBoxLayout(header)
    layout.setContentsMargins(2, 0, 6, 0)
    layout.setSpacing(6)
    title = QLabel(title_text)
    title.setObjectName("panelTitle")
    layout.addWidget(title)
    layout.addStretch(1)
    return header, layout, title


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
        grad.setColorAt(1.0, QColor("#2D8CEB"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(grad)
        painter.drawRoundedRect(QRectF(bar_x, float(cy) - 2.0, bar_w, 4.0), 2.0, 2.0)

        if self._trim_in_ms is not None:
            painter.setPen(QPen(QColor("#10B981"), 2.0))
            painter.drawLine(x_in, cy - 5, x_in, cy + 5)
        if self._trim_out_ms is not None:
            painter.setPen(QPen(QColor("#3EA6FF"), 2.0))
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


class FrameTestWorker(QObject):
    """Runs single-frame stroke-mask + OCR preview inside a dedicated QThread."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        pipeline: ExtractionPipeline,
        frame_bgr: np.ndarray,
        frame_index: int,
        pts_ms: int,
        roi: RoiRect,
        config: ExtractConfig,
    ) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._frame_bgr = frame_bgr
        self._frame_index = frame_index
        self._pts_ms = pts_ms
        self._roi = roi
        self._config = config

    @Slot()
    def run(self) -> None:
        try:
            result = self._pipeline.test_frame_bgr(
                self._frame_bgr,
                self._frame_index,
                self._pts_ms,
                self._roi,
                self._config,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.finished.emit(result)


class UpdateDialog(QDialog):
    """Premiere-styled dialog presenting an available GitHub release and changelog."""

    def __init__(
        self,
        info: UpdateInfo,
        lang: str = "en",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.info = info
        self.lang = lang
        self.action: str = "later"

        self.setWindowTitle(tr(lang, "update_dialog_title"))
        self.setModal(True)
        self.setMinimumWidth(460)
        self.resize(500, 380)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 14)
        root.setSpacing(10)

        header = QLabel(tr(lang, "update_dialog_header"))
        header.setObjectName("updateHeaderTitle")
        root.addWidget(header)

        meta_row = QHBoxLayout()
        meta_row.setContentsMargins(0, 0, 0, 0)
        meta_row.setSpacing(8)
        self.version_pill = QLabel(f"v{info.current_version}  →  v{info.latest_version}")
        self.version_pill.setObjectName("updateVersionPill")
        meta_row.addWidget(self.version_pill)

        self.published_label: QLabel | None = None
        if info.published_at:
            published_date = info.published_at[:10] if len(info.published_at) >= 10 else info.published_at
            self.published_label = QLabel(tr(lang, "update_dialog_published", date=published_date))
            self.published_label.setObjectName("updatePublishedLabel")
            meta_row.addWidget(self.published_label)
        meta_row.addStretch(1)
        root.addLayout(meta_row)

        root.addWidget(_make_section_label(tr(lang, "update_dialog_release_notes")))

        self.changelog_edit = QPlainTextEdit()
        self.changelog_edit.setObjectName("updateChangelog")
        self.changelog_edit.setReadOnly(True)
        self.changelog_edit.setMinimumHeight(180)
        self.changelog_edit.setPlainText(
            info.changelog or tr(lang, "update_dialog_fallback_notes")
        )
        root.addWidget(self.changelog_edit, 1)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 4, 0, 0)
        btn_row.setSpacing(8)

        self.btn_skip_version = QPushButton(tr(lang, "btn_skip_version"))
        self.btn_skip_version.setProperty("variant", "tool")
        self.btn_skip_version.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_skip_version.setCursor(Qt.CursorShape.PointingHandCursor)

        self.btn_later = QPushButton(tr(lang, "btn_later"))
        self.btn_later.setProperty("variant", "tool")
        self.btn_later.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_later.setCursor(Qt.CursorShape.PointingHandCursor)

        self.btn_update_now = QPushButton(tr(lang, "btn_update_now"))
        self.btn_update_now.setProperty("variant", "primary")
        self.btn_update_now.setDefault(True)
        self.btn_update_now.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_update_now.setCursor(Qt.CursorShape.PointingHandCursor)

        btn_row.addWidget(self.btn_skip_version)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_later)
        btn_row.addWidget(self.btn_update_now)
        root.addLayout(btn_row)

        self.btn_update_now.clicked.connect(self._on_update_now)
        self.btn_skip_version.clicked.connect(self._on_skip_version)
        self.btn_later.clicked.connect(self._on_later)

    def _on_update_now(self) -> None:
        self.action = "update"
        self.accept()

    def _on_skip_version(self) -> None:
        self.action = "skip"
        self.reject()

    def _on_later(self) -> None:
        self.action = "later"
        self.reject()


class _UpdateCheckWorker(QObject):
    """Runs GitHub Releases update lookup off the Qt UI thread."""

    finished = Signal(object, bool)
    failed = Signal(str, bool)

    def __init__(
        self,
        check_fn: Callable[[str], UpdateInfo | None],
        current_version: str,
        manual: bool,
    ) -> None:
        super().__init__()
        self._check_fn = check_fn
        self._current_version = current_version
        self._manual = manual

    @Slot()
    def run(self) -> None:
        try:
            info = self._check_fn(self._current_version)
        except Exception as exc:
            self.failed.emit(str(exc), self._manual)
        else:
            self.finished.emit(info, self._manual)


class MainWindow(QMainWindow):
    """Main desktop window coordinating ROI selection, extraction, review, and export."""

    def __init__(
        self,
        preset_store: PresetStore | None = None,
        pipeline: ExtractionPipeline | None = None,
        overwrite_confirm_fn: Callable[[list[Path]], bool] | None = None,
        unsaved_confirm_fn: Callable[[], bool] | None = None,
        parent: QWidget | None = None,
        auto_check_updates: bool = True,
        update_prefs_store: UpdatePrefsStore | None = None,
        update_check_fn: Callable[[str], UpdateInfo | None] | None = None,
        open_url_fn: Callable[[str], bool] | None = None,
        language: str | None = None,
        discord_rpc: DiscordRpcClient | None = None,
    ) -> None:
        super().__init__(parent)
        self._preset_store = preset_store if preset_store is not None else PresetStore()
        self._pipeline = pipeline if pipeline is not None else ExtractionPipeline()
        self._overwrite_confirm_fn = overwrite_confirm_fn
        self._unsaved_confirm_fn = unsaved_confirm_fn
        self._update_prefs = (
            update_prefs_store if update_prefs_store is not None else UpdatePrefsStore()
        )
        self._update_check_fn: Callable[[str], UpdateInfo | None] = (
            update_check_fn if update_check_fn is not None else check_for_updates
        )
        self._open_url_fn: Callable[[str], bool] = (
            open_url_fn
            if open_url_fn is not None
            else (lambda url: QDesktopServices.openUrl(QUrl(url)))
        )

        raw_lang_pref = language if language is not None else self._update_prefs.get_language()
        normalized_pref = raw_lang_pref.strip().lower() if isinstance(raw_lang_pref, str) else "auto"
        if normalized_pref not in ("auto", "en", "vi", "zh"):
            normalized_pref = "auto"
        self._language_pref: str = normalized_pref
        self._lang: str = resolve_language(self._language_pref)
        self._status_key: str | None = "status_ready"
        self._status_kwargs: dict[str, Any] = {}
        self._gpu_available: bool = "directml" in OcrEngine.available_providers()

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
        self._has_unexported_edits: bool = False
        self._last_export_dir: Path | None = None
        self._is_playing: bool = False
        self._at_eof: bool = False
        self._last_advance_error: bool = False
        self._play_iter: Iterator[tuple[int, int, np.ndarray]] | None = None
        self._pending_play_frame: tuple[int, int, np.ndarray] | None = None
        self._audio_player = AudioPreviewPlayer(self)
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._on_playback_tick)

        self._worker_thread: QThread | None = None
        self._worker: _ExtractionWorker | None = None
        self._cancel_event: threading.Event | None = None
        self._frame_test_thread: QThread | None = None
        self._frame_test_worker: FrameTestWorker | None = None
        self._frame_test_bgr: np.ndarray | None = None
        self._last_frame_test_result: FrameTestResult | None = None
        self._update_thread: QThread | None = None
        self._update_worker: _UpdateCheckWorker | None = None
        self._available_update: UpdateInfo | None = None
        self._update_startup_timer = QTimer(self)
        self._update_startup_timer.setSingleShot(True)
        self._update_startup_timer.timeout.connect(
            lambda: self.check_for_updates_async(manual=False)
        )
        self._retiring_threads: list[QThread] = []
        self._retiring_workers: dict[QThread, QObject | None] = {}
        self._saved_settings_width: int = 268
        self._workbench_panels: tuple[QFrame, ...] = ()
        self._panel_titles: dict[QFrame, QLabel] = {}
        self._active_panel: QFrame | None = None
        self._focus_tracking_active: bool = False

        self._rpc_enabled: bool = self._update_prefs.get_discord_rpc_enabled()
        self._rpc_hide_filename: bool = self._update_prefs.get_discord_rpc_hide_filename()
        self._rpc_show_github: bool = self._update_prefs.get_discord_rpc_show_github_button()
        self._rpc_session_started_at: int = int(time.time())
        self._rpc_extract_started_at: int = 0
        self._rpc_extract_provider: str = "cpu"
        self._rpc_extract_model: str = "accurate"
        self._rpc_last_stage: str = ""
        if discord_rpc is not None:
            self._discord_rpc = discord_rpc
            self._discord_rpc.configure(
                enabled=self._rpc_enabled,
                hide_filename=self._rpc_hide_filename,
                show_github_button=self._rpc_show_github,
            )
        else:
            self._discord_rpc = DiscordRpcClient(
                app_version=__version__,
                enabled=self._rpc_enabled,
                hide_filename=self._rpc_hide_filename,
                show_github_button=self._rpc_show_github,
            )

        self.setAcceptDrops(True)
        self.setWindowTitle(f"Muzhen - V{__version__}")
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
        self._install_focus_tracking()
        self._sync_language_combo()
        self._refresh_presets()
        self._retranslate_ui()
        self._update_controls_state()
        self._publish_discord_presence(mode="idle", urgent=True)

        in_pytest = bool(os.environ.get("PYTEST_CURRENT_TEST"))
        if discord_rpc is not None or not in_pytest:
            self._discord_rpc.start()
        if auto_check_updates and (update_check_fn is not None or not in_pytest):
            self._update_startup_timer.start(1200)

    @property
    def current_result(self) -> ExtractResult | None:
        """Return the active in-memory extraction result, if any."""
        return self._extract_result

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        workbench_gutter = QWidget()
        workbench_gutter.setObjectName("workbenchGutter")
        workbench_gutter.setStyleSheet("QWidget#workbenchGutter { background-color: #111111; }")
        gutter_layout = QVBoxLayout(workbench_gutter)
        gutter_layout.setContentsMargins(4, 4, 4, 4)
        gutter_layout.setSpacing(0)

        self.workbench_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.workbench_splitter.setHandleWidth(3)
        self.workbench_splitter.setChildrenCollapsible(False)

        self.settings_panel = QFrame()
        self.settings_panel.setProperty("panel", True)
        self.settings_panel.setProperty("activePanel", False)
        self.settings_panel.setMinimumWidth(244)
        self.settings_panel.setMaximumWidth(380)
        settings_root = QVBoxLayout(self.settings_panel)
        settings_root.setContentsMargins(0, 0, 0, 0)
        settings_root.setSpacing(0)

        settings_header, _settings_hdr_layout, settings_title = _make_panel_header(
            tr(self._lang, "panel_settings")
        )
        settings_root.addWidget(settings_header)

        settings_content = QWidget()
        inspector_outer = QVBoxLayout(settings_content)
        inspector_outer.setContentsMargins(8, 6, 8, 6)
        inspector_outer.setSpacing(6)

        scroll = QScrollArea()
        scroll.setObjectName("inspectorScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(0)
        scroll.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)

        inspector_body = QWidget()
        body_layout = QVBoxLayout(inspector_body)
        body_layout.setContentsMargins(0, 0, 2, 0)
        body_layout.setSpacing(6)

        self.lbl_render_method = _make_section_label(tr(self._lang, "render_method"))
        body_layout.addWidget(self.lbl_render_method)

        render_seg = QFrame()
        render_seg.setObjectName("segmentedControl")
        render_seg_layout = QHBoxLayout(render_seg)
        render_seg_layout.setContentsMargins(2, 2, 2, 2)
        render_seg_layout.setSpacing(2)
        self.btn_render_gpu = QPushButton(tr(self._lang, "render_gpu"))
        self.btn_render_cpu = QPushButton(tr(self._lang, "render_cpu"))
        self._render_button_group = QButtonGroup(self)
        self._render_button_group.setExclusive(True)
        for seg_btn in (self.btn_render_gpu, self.btn_render_cpu):
            seg_btn.setProperty("variant", "segment")
            seg_btn.setCheckable(True)
            seg_btn.setFixedHeight(22)
            seg_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            seg_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._render_button_group.addButton(seg_btn)
            render_seg_layout.addWidget(seg_btn, 1)
        if self._gpu_available:
            self.btn_render_gpu.setChecked(True)
        else:
            self.btn_render_cpu.setChecked(True)
            self.btn_render_gpu.setEnabled(False)
        body_layout.addWidget(render_seg)

        model_row = QHBoxLayout()
        model_row.setContentsMargins(0, 0, 0, 0)
        model_row.setSpacing(6)
        self.lbl_model = QLabel(tr(self._lang, "model_label"))
        self.lbl_model.setProperty("role", "fieldLabel")
        self.ocr_model_combo = QComboBox()
        self.ocr_model_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.ocr_model_combo.setMinimumContentsLength(6)
        self.ocr_model_combo.addItem(tr(self._lang, "model_precision"), "accurate")
        self.ocr_model_combo.addItem(tr(self._lang, "model_turbo"), "fast")
        model_row.addWidget(self.lbl_model)
        model_row.addWidget(self.ocr_model_combo, 1)
        body_layout.addLayout(model_row)

        conf_row = QHBoxLayout()
        conf_row.setContentsMargins(0, 0, 0, 0)
        conf_row.setSpacing(6)
        self.lbl_conf = QLabel(tr(self._lang, "flag_threshold"))
        self.lbl_conf.setProperty("role", "fieldLabel")
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.0, 1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setValue(0.85)
        conf_row.addWidget(self.lbl_conf)
        conf_row.addWidget(self.spin_conf, 1)
        body_layout.addLayout(conf_row)

        body_layout.addWidget(_make_divider())

        self.lbl_roi_preset = _make_section_label(tr(self._lang, "roi_preset"))
        body_layout.addWidget(self.lbl_roi_preset)

        preset_row = QHBoxLayout()
        preset_row.setContentsMargins(0, 0, 0, 0)
        preset_row.setSpacing(4)
        self.preset_combo = QComboBox()
        self.preset_combo.setFixedHeight(24)
        self.preset_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.preset_combo.setMinimumContentsLength(6)

        self.btn_save_preset = QPushButton()
        self.btn_save_preset.setProperty("variant", "tool")
        self.btn_save_preset.setIcon(make_icon("save", size=13))
        self.btn_save_preset.setIconSize(QSize(13, 13))
        self.btn_save_preset.setFixedSize(24, 24)
        self.btn_save_preset.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_save_preset.setCursor(Qt.CursorShape.PointingHandCursor)

        self.btn_delete_preset = QPushButton()
        self.btn_delete_preset.setProperty("variant", "danger")
        self.btn_delete_preset.setIcon(make_icon("trash", color="#F08080", size=13))
        self.btn_delete_preset.setIconSize(QSize(13, 13))
        self.btn_delete_preset.setFixedSize(24, 24)
        self.btn_delete_preset.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_delete_preset.setCursor(Qt.CursorShape.PointingHandCursor)

        preset_row.addWidget(self.preset_combo, 1)
        preset_row.addWidget(self.btn_save_preset)
        preset_row.addWidget(self.btn_delete_preset)
        body_layout.addLayout(preset_row)

        body_layout.addWidget(_make_divider())

        self.lbl_export_format = _make_section_label(tr(self._lang, "export_format"))
        body_layout.addWidget(self.lbl_export_format)

        fmt_row = QHBoxLayout()
        fmt_row.setContentsMargins(0, 0, 0, 0)
        fmt_row.setSpacing(4)
        self.chk_ass = QPushButton(".ass")
        self.chk_srt = QPushButton(".srt")
        self.chk_vtt = QPushButton(".vtt")
        self.chk_txt = QPushButton(".txt")
        self._format_chips = (self.chk_ass, self.chk_srt, self.chk_vtt, self.chk_txt)
        for chip in self._format_chips:
            chip.setProperty("variant", "formatChip")
            chip.setCheckable(True)
            chip.setFixedHeight(24)
            chip.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            fmt_row.addWidget(chip, 1)
        self.chk_ass.setChecked(True)
        self.chk_srt.setChecked(True)
        body_layout.addLayout(fmt_row)

        self.lbl_output_folder = _make_section_label(tr(self._lang, "output_folder"))
        body_layout.addWidget(self.lbl_output_folder)

        out_seg = QFrame()
        out_seg.setObjectName("segmentedControl")
        out_seg_layout = QHBoxLayout(out_seg)
        out_seg_layout.setContentsMargins(2, 2, 2, 2)
        out_seg_layout.setSpacing(2)
        self.btn_out_same = QPushButton(tr(self._lang, "output_same"))
        self.btn_out_custom = QPushButton(tr(self._lang, "output_custom"))
        self._out_dir_group = QButtonGroup(self)
        self._out_dir_group.setExclusive(True)
        for out_btn in (self.btn_out_same, self.btn_out_custom):
            out_btn.setProperty("variant", "segment")
            out_btn.setCheckable(True)
            out_btn.setFixedHeight(22)
            out_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            out_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._out_dir_group.addButton(out_btn)
            out_seg_layout.addWidget(out_btn, 1)
        self.btn_out_same.setChecked(True)
        body_layout.addWidget(out_seg)

        self.custom_dir_row_widget = QWidget()
        custom_dir_row = QHBoxLayout(self.custom_dir_row_widget)
        custom_dir_row.setContentsMargins(0, 0, 0, 0)
        custom_dir_row.setSpacing(4)
        self.custom_dir_edit = QLineEdit()
        self.custom_dir_edit.setFixedHeight(24)
        self.btn_browse_out_dir = QPushButton()
        self.btn_browse_out_dir.setProperty("variant", "tool")
        self.btn_browse_out_dir.setIcon(make_icon("folder", size=13))
        self.btn_browse_out_dir.setIconSize(QSize(13, 13))
        self.btn_browse_out_dir.setFixedSize(24, 24)
        self.btn_browse_out_dir.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_browse_out_dir.setCursor(Qt.CursorShape.PointingHandCursor)
        custom_dir_row.addWidget(self.custom_dir_edit, 1)
        custom_dir_row.addWidget(self.btn_browse_out_dir)
        self.custom_dir_row_widget.setVisible(False)
        body_layout.addWidget(self.custom_dir_row_widget)

        body_layout.addWidget(_make_divider())

        lang_row = QHBoxLayout()
        lang_row.setContentsMargins(0, 0, 0, 0)
        lang_row.setSpacing(6)
        self.lbl_language = QLabel(tr(self._lang, "language_label"))
        self.lbl_language.setProperty("role", "fieldLabel")
        self.language_combo = QComboBox()
        self.language_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.language_combo.setMinimumContentsLength(6)
        for code, label in LANGUAGE_CHOICES:
            self.language_combo.addItem(label, code)
        lang_row.addWidget(self.lbl_language)
        lang_row.addWidget(self.language_combo, 1)
        body_layout.addLayout(lang_row)

        body_layout.addWidget(_make_divider())

        self.lbl_discord_rpc = _make_section_label(tr(self._lang, "discord_rpc"))
        body_layout.addWidget(self.lbl_discord_rpc)

        rpc_seg = QFrame()
        rpc_seg.setObjectName("segmentedControl")
        rpc_seg_layout = QHBoxLayout(rpc_seg)
        rpc_seg_layout.setContentsMargins(2, 2, 2, 2)
        rpc_seg_layout.setSpacing(2)
        self.btn_rpc_on = QPushButton(tr(self._lang, "rpc_on"))
        self.btn_rpc_off = QPushButton(tr(self._lang, "rpc_off"))
        self._rpc_toggle_group = QButtonGroup(self)
        self._rpc_toggle_group.setExclusive(True)
        for rpc_btn in (self.btn_rpc_on, self.btn_rpc_off):
            rpc_btn.setProperty("variant", "segment")
            rpc_btn.setCheckable(True)
            rpc_btn.setFixedHeight(22)
            rpc_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            rpc_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self._rpc_toggle_group.addButton(rpc_btn)
            rpc_seg_layout.addWidget(rpc_btn, 1)
        if self._rpc_enabled:
            self.btn_rpc_on.setChecked(True)
        else:
            self.btn_rpc_off.setChecked(True)
        body_layout.addWidget(rpc_seg)

        self.rpc_privacy_widget = QWidget()
        rpc_privacy_layout = QVBoxLayout(self.rpc_privacy_widget)
        rpc_privacy_layout.setContentsMargins(2, 2, 0, 0)
        rpc_privacy_layout.setSpacing(4)

        self.chk_rpc_hide_filename = QCheckBox(tr(self._lang, "rpc_hide_filename"))
        self.chk_rpc_hide_filename.setChecked(self._rpc_hide_filename)
        self.chk_rpc_hide_filename.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.chk_rpc_hide_filename.setCursor(Qt.CursorShape.PointingHandCursor)

        self.chk_rpc_show_github = QCheckBox(tr(self._lang, "rpc_show_github"))
        self.chk_rpc_show_github.setChecked(self._rpc_show_github)
        self.chk_rpc_show_github.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.chk_rpc_show_github.setCursor(Qt.CursorShape.PointingHandCursor)

        rpc_privacy_layout.addWidget(self.chk_rpc_hide_filename)
        rpc_privacy_layout.addWidget(self.chk_rpc_show_github)
        self.rpc_privacy_widget.setVisible(self._rpc_enabled)
        body_layout.addWidget(self.rpc_privacy_widget)

        body_layout.addStretch(1)
        scroll.setWidget(inspector_body)
        inspector_outer.addWidget(scroll, 1)

        inspector_outer.addWidget(_make_divider())

        run_row = QHBoxLayout()
        run_row.setSpacing(6)
        self.btn_start = QPushButton(tr(self._lang, "btn_extract"))
        self.btn_start.setProperty("variant", "primary")
        self.btn_start.setIcon(make_icon("play", color="#FFFFFF", size=13))
        self.btn_start.setIconSize(QSize(13, 13))
        self.btn_start.setMinimumHeight(30)
        self.btn_start.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        self.btn_start.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_start.setCursor(Qt.CursorShape.PointingHandCursor)

        self.btn_stop = QPushButton(tr(self._lang, "btn_stop"))
        self.btn_stop.setProperty("variant", "danger")
        self.btn_stop.setIcon(make_icon("stop", color="#F08080", size=12))
        self.btn_stop.setIconSize(QSize(12, 12))
        self.btn_stop.setMinimumHeight(30)
        self.btn_stop.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        self.btn_stop.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)

        run_row.addWidget(self.btn_start, 3)
        run_row.addWidget(self.btn_stop, 1)
        inspector_outer.addLayout(run_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label = QLabel(tr(self._lang, "status_ready"))
        self.status_label.setObjectName("statusLabel")
        self.status_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        inspector_outer.addWidget(self.progress_bar)
        inspector_outer.addWidget(self.status_label)

        inspector_outer.addWidget(_make_divider())

        self.btn_update_available = QPushButton("")
        self.btn_update_available.setProperty("variant", "update")
        self.btn_update_available.setFixedHeight(24)
        self.btn_update_available.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        self.btn_update_available.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_update_available.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_update_available.setVisible(False)
        inspector_outer.addWidget(self.btn_update_available)

        footer_row = QHBoxLayout()
        footer_row.setContentsMargins(0, 0, 0, 0)
        footer_row.setSpacing(4)

        self.btn_check_update = QPushButton(tr(self._lang, "btn_check_updates"))
        self.btn_check_update.setProperty("variant", "tool")
        self.btn_check_update.setIcon(make_icon("refresh", size=12))
        self.btn_check_update.setIconSize(QSize(12, 12))
        self.btn_check_update.setFixedHeight(24)
        self.btn_check_update.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        self.btn_check_update.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_check_update.setCursor(Qt.CursorShape.PointingHandCursor)

        self.btn_author_star = QPushButton("Muzhen")
        self.btn_author_star.setProperty("variant", "ghost")
        self.btn_author_star.setIcon(make_icon("star", color="#E5B54D", size=12))
        self.btn_author_star.setIconSize(QSize(12, 12))
        self.btn_author_star.setFixedHeight(24)
        self.btn_author_star.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        self.btn_author_star.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_author_star.setCursor(Qt.CursorShape.PointingHandCursor)

        footer_row.addWidget(self.btn_check_update, 4)
        footer_row.addWidget(self.btn_author_star, 5)
        inspector_outer.addLayout(footer_row)
        settings_root.addWidget(settings_content, 1)

        self.monitor_panel = QFrame()
        self.monitor_panel.setProperty("panel", True)
        self.monitor_panel.setProperty("activePanel", True)
        monitor_root = QVBoxLayout(self.monitor_panel)
        monitor_root.setContentsMargins(0, 0, 0, 0)
        monitor_root.setSpacing(0)

        monitor_header, monitor_hdr_layout, monitor_title = _make_panel_header(
            tr(self._lang, "panel_monitor")
        )
        self.video_name_label = _ShrinkableLabel("", max_min_width=64)
        self.video_name_label.setObjectName("videoNameLabel")
        self.video_name_label.setMaximumWidth(150)
        self.video_name_label.setVisible(False)

        self.video_meta_badge = _ShrinkableLabel("", max_min_width=72)
        self.video_meta_badge.setObjectName("metaBadge")
        self.video_meta_badge.setVisible(False)

        self.btn_browse_video = QPushButton()
        self.btn_browse_video.setProperty("variant", "tool")
        self.btn_browse_video.setIcon(make_icon("folder", size=12))
        self.btn_browse_video.setIconSize(QSize(12, 12))
        self.btn_browse_video.setFixedSize(24, 22)
        self.btn_browse_video.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_browse_video.setCursor(Qt.CursorShape.PointingHandCursor)

        self.roi_label = QLabel()
        self.roi_label.setObjectName("roiPill")

        monitor_hdr_layout.addWidget(self.video_name_label)
        monitor_hdr_layout.addWidget(self.video_meta_badge)
        monitor_hdr_layout.addWidget(self.btn_browse_video)
        monitor_hdr_layout.addWidget(self.roi_label)
        monitor_root.addWidget(monitor_header)

        monitor_body = QWidget()
        preview_layout = QVBoxLayout(monitor_body)
        preview_layout.setContentsMargins(6, 6, 6, 6)
        preview_layout.setSpacing(5)

        self.roi_canvas = RoiCanvas()
        preview_layout.addWidget(self.roi_canvas, 1)

        scrub_row = QHBoxLayout()
        scrub_row.setSpacing(4)
        self.btn_seek_start = QPushButton()
        self.btn_seek_start.setIcon(make_icon("skip_back", size=13))
        self.btn_play_pause = QPushButton()
        self.btn_play_pause.setIcon(make_icon("play", size=13))
        self.btn_step_back = QPushButton()
        self.btn_step_back.setIcon(make_icon("step_back", size=13))
        self.frame_slider = TrimRangeSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setTracking(False)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.frame_slider.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_step_fwd = QPushButton()
        self.btn_step_fwd.setIcon(make_icon("step_fwd", size=13))
        self.btn_seek_end = QPushButton()
        self.btn_seek_end.setIcon(make_icon("skip_fwd", size=13))
        self.btn_mute = QPushButton()
        self.btn_mute.setCheckable(True)
        self.btn_mute.setChecked(False)
        self.btn_mute.setIcon(make_icon("volume", size=13))
        for btn in (
            self.btn_seek_start,
            self.btn_play_pause,
            self.btn_step_back,
            self.btn_step_fwd,
            self.btn_seek_end,
            self.btn_mute,
        ):
            btn.setProperty("variant", "tool")
            btn.setIconSize(QSize(13, 13))
            btn.setFixedSize(26, 26)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
        for step_btn in (self.btn_step_back, self.btn_step_fwd):
            step_btn.setAutoRepeat(True)
            step_btn.setAutoRepeatDelay(250)
            step_btn.setAutoRepeatInterval(55)
        self.time_label = QLabel("0:00:00.00 / 0:00:00.00")
        self.time_label.setObjectName("timecodeBadge")
        scrub_row.addWidget(self.btn_seek_start)
        scrub_row.addWidget(self.btn_play_pause)
        scrub_row.addWidget(self.btn_step_back)
        scrub_row.addWidget(self.frame_slider, 1)
        scrub_row.addWidget(self.btn_step_fwd)
        scrub_row.addWidget(self.btn_seek_end)
        scrub_row.addWidget(self.btn_mute)
        scrub_row.addWidget(self.time_label)
        preview_layout.addLayout(scrub_row)

        trim_row = QHBoxLayout()
        trim_row.setContentsMargins(0, 0, 0, 0)
        trim_row.setSpacing(3)
        self.btn_set_in = _CompactButton(tr(self._lang, "trim_in"))
        self.btn_set_in.setProperty("variant", "trim")
        self.btn_set_in.setIcon(make_icon("trim_in", color="#10B981", size=12))
        self.btn_set_in.setIconSize(QSize(12, 12))
        self.btn_set_in.setFixedHeight(24)
        self.btn_set_in.setCursor(Qt.CursorShape.PointingHandCursor)

        self.lbl_trim_range = QLabel(tr(self._lang, "trim_full"))
        self.lbl_trim_range.setObjectName("trimRangeBadge")
        self.lbl_trim_range.setProperty("active", False)
        self.lbl_trim_range.setFixedHeight(24)
        self.lbl_trim_range.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

        self.btn_set_out = _CompactButton(tr(self._lang, "trim_out"))
        self.btn_set_out.setProperty("variant", "trim")
        self.btn_set_out.setIcon(make_icon("trim_out", color="#3EA6FF", size=12))
        self.btn_set_out.setIconSize(QSize(12, 12))
        self.btn_set_out.setFixedHeight(24)
        self.btn_set_out.setCursor(Qt.CursorShape.PointingHandCursor)

        self.btn_clear_trim = QPushButton()
        self.btn_clear_trim.setProperty("variant", "tool")
        self.btn_clear_trim.setIcon(make_icon("x_mark", color="#A0A0A0", size=11))
        self.btn_clear_trim.setIconSize(QSize(11, 11))
        self.btn_clear_trim.setFixedSize(24, 24)
        self.btn_clear_trim.setCursor(Qt.CursorShape.PointingHandCursor)

        for trim_btn in (self.btn_set_in, self.btn_set_out, self.btn_clear_trim):
            trim_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        trim_sep = QFrame()
        trim_sep.setFixedSize(1, 16)
        trim_sep.setStyleSheet("background-color: #2E2E2E;")

        self.test_result_label = QLabel("")
        self.test_result_label.setObjectName("testResultLabel")
        self.test_result_label.setWordWrap(False)
        self.test_result_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )

        self.btn_toggle_settings = _CompactButton(tr(self._lang, "btn_settings"))
        self.btn_toggle_settings.setProperty("variant", "toggle")
        self.btn_toggle_settings.setIcon(make_icon("sidebar", size=12))
        self.btn_toggle_settings.setIconSize(QSize(12, 12))
        self.btn_toggle_settings.setCheckable(True)
        self.btn_toggle_settings.setChecked(True)
        self.btn_toggle_settings.setFixedHeight(24)
        self.btn_toggle_settings.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_toggle_settings.setCursor(Qt.CursorShape.PointingHandCursor)

        self.btn_test_frame = _CompactButton(tr(self._lang, "btn_test_frame"))
        self.btn_test_frame.setProperty("variant", "trim")
        self.btn_test_frame.setIcon(make_icon("scan", color="#10B981", size=12))
        self.btn_test_frame.setIconSize(QSize(12, 12))
        self.btn_test_frame.setFixedHeight(24)
        self.btn_test_frame.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_test_frame.setCursor(Qt.CursorShape.PointingHandCursor)

        trim_row.addWidget(self.btn_set_in)
        trim_row.addWidget(self.lbl_trim_range)
        trim_row.addWidget(self.btn_set_out)
        trim_row.addWidget(self.btn_clear_trim)
        trim_row.addWidget(trim_sep)
        trim_row.addWidget(self.test_result_label, 1)
        trim_row.addWidget(self.btn_toggle_settings)
        trim_row.addWidget(self.btn_test_frame)
        preview_layout.addLayout(trim_row)
        monitor_root.addWidget(monitor_body, 1)

        self.subtitles_panel = QFrame()
        self.subtitles_panel.setProperty("panel", True)
        self.subtitles_panel.setProperty("activePanel", False)
        subtitles_root = QVBoxLayout(self.subtitles_panel)
        subtitles_root.setContentsMargins(0, 0, 0, 0)
        subtitles_root.setSpacing(0)

        subtitles_header, subtitles_hdr_layout, subtitles_title = _make_panel_header(
            tr(self._lang, "panel_subtitles")
        )
        self.btn_export = QPushButton()
        self.btn_export.setProperty("variant", "primary")
        self.btn_export.setIcon(make_icon("export", color="#FFFFFF", size=12))
        self.btn_export.setIconSize(QSize(12, 12))
        self.btn_export.setFixedHeight(22)
        self.btn_export.setStyleSheet("padding: 1px 8px; font-size: 11px; min-height: 16px;")
        self.btn_export.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_export.setCursor(Qt.CursorShape.PointingHandCursor)

        self.btn_open_out_dir = QPushButton()
        self.btn_open_out_dir.setProperty("variant", "tool")
        self.btn_open_out_dir.setIcon(make_icon("folder_open", size=12))
        self.btn_open_out_dir.setIconSize(QSize(12, 12))
        self.btn_open_out_dir.setFixedSize(24, 22)
        self.btn_open_out_dir.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_open_out_dir.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_open_out_dir.setEnabled(False)

        subtitles_hdr_layout.addWidget(self.btn_export)
        subtitles_hdr_layout.addWidget(self.btn_open_out_dir)
        subtitles_root.addWidget(subtitles_header)

        subtitles_body = QWidget()
        review_layout = QVBoxLayout(subtitles_body)
        review_layout.setContentsMargins(6, 6, 6, 6)
        review_layout.setSpacing(0)

        self.review_table = ReviewTable()
        review_layout.addWidget(self.review_table, 1)
        subtitles_root.addWidget(subtitles_body, 1)

        self.workbench_splitter.addWidget(self.settings_panel)
        self.workbench_splitter.addWidget(self.monitor_panel)
        self.workbench_splitter.addWidget(self.subtitles_panel)
        self.workbench_splitter.setStretchFactor(0, 0)
        self.workbench_splitter.setStretchFactor(1, 12)
        self.workbench_splitter.setStretchFactor(2, 11)
        self.workbench_splitter.setSizes([268, 456, 436])

        gutter_layout.addWidget(self.workbench_splitter, 1)
        main_layout.addWidget(workbench_gutter, 1)

        self._workbench_panels = (
            self.settings_panel,
            self.monitor_panel,
            self.subtitles_panel,
        )
        self._panel_titles = {
            self.settings_panel: settings_title,
            self.monitor_panel: monitor_title,
            self.subtitles_panel: subtitles_title,
        }
        self._active_panel = self.monitor_panel

    def _connect_signals(self) -> None:
        self.roi_canvas.roi_changed.connect(self._update_roi_label)
        self.roi_canvas.video_dropped.connect(self._on_video_dropped)
        self.roi_canvas.open_requested.connect(self._browse_video)
        self.btn_browse_video.clicked.connect(self._browse_video)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_selected)
        self.btn_save_preset.clicked.connect(lambda: self.save_preset())
        self.btn_delete_preset.clicked.connect(self.delete_current_preset)
        self.btn_toggle_settings.toggled.connect(self._on_toggle_settings)
        self.btn_update_available.clicked.connect(self._on_update_badge_clicked)
        self.btn_check_update.clicked.connect(lambda: self.check_for_updates_async(manual=True))
        self.btn_author_star.clicked.connect(self._on_author_star_clicked)
        self.language_combo.currentIndexChanged.connect(self._on_language_combo_changed)
        self.btn_rpc_on.toggled.connect(self._on_discord_rpc_settings_changed)
        self.chk_rpc_hide_filename.toggled.connect(self._on_discord_rpc_settings_changed)
        self.chk_rpc_show_github.toggled.connect(self._on_discord_rpc_settings_changed)

        self.btn_seek_start.clicked.connect(self._seek_to_start)
        self.btn_play_pause.clicked.connect(self.toggle_playback)
        self.btn_seek_end.clicked.connect(self._seek_to_end)
        self.btn_mute.toggled.connect(self._on_mute_toggled)
        self.btn_step_back.clicked.connect(lambda: self._step_frame(-1))
        self.btn_step_fwd.clicked.connect(lambda: self._step_frame(1))
        self.frame_slider.sliderPressed.connect(self.stop_playback)
        self.frame_slider.sliderMoved.connect(self._on_slider_moved)
        self.frame_slider.valueChanged.connect(self._on_slider_value_changed)
        self.btn_set_in.clicked.connect(lambda: self.set_trim_in())
        self.btn_set_out.clicked.connect(lambda: self.set_trim_out())
        self.btn_clear_trim.clicked.connect(self.clear_trim)

        for chip in self._format_chips:
            chip.toggled.connect(lambda _: self._update_controls_state())
        self.btn_out_same.toggled.connect(lambda _: self._update_controls_state())
        self.btn_out_custom.toggled.connect(lambda _: self._update_controls_state())
        self.btn_browse_out_dir.clicked.connect(self._browse_output_dir)

        self.btn_test_frame.clicked.connect(self.test_current_frame)
        self.btn_start.clicked.connect(self.start_extraction)
        self.btn_stop.clicked.connect(self.stop_extraction)
        self.btn_export.clicked.connect(self.export_subtitles)
        self.btn_open_out_dir.clicked.connect(self.open_output_folder)

        self.review_table.jump_requested.connect(self._on_review_jump_requested)
        self.review_table.row_seek_requested.connect(self._on_review_jump_requested)
        self.review_table.snap_start_requested.connect(self._on_review_snap_start)
        self.review_table.snap_end_requested.connect(self._on_review_snap_end)
        self.review_table.split_requested.connect(self._on_review_split_requested)
        self.review_table.dirty_changed.connect(self._on_review_dirty_changed)
        self.review_table.events_changed.connect(self._on_review_events_changed)
        self.review_table.transport_key_forwarded.connect(self._on_review_transport_key)

    def selected_provider(self) -> str:
        """Return the active OCR execution provider ('directml' or 'cpu')."""
        if self._gpu_available and self.btn_render_gpu.isChecked():
            return "directml"
        return "cpu"

    def active_language(self) -> str:
        """Return the active resolved UI language code ('en', 'vi', or 'zh')."""
        return self._lang

    def _sync_language_combo(self) -> None:
        idx = self.language_combo.findData(self._language_pref)
        if idx < 0:
            idx = 0
        self.language_combo.blockSignals(True)
        try:
            self.language_combo.setCurrentIndex(idx)
        finally:
            self.language_combo.blockSignals(False)

    def _on_language_combo_changed(self, index: int) -> None:
        code = self.language_combo.itemData(index)
        if isinstance(code, str):
            self.set_language(code, persist=True)

    def set_language(self, pref: str, *, persist: bool = True) -> None:
        """Switch UI language live ('auto', 'en', 'vi', 'zh') with optional persistence."""
        normalized = pref.strip().lower() if isinstance(pref, str) else "auto"
        if normalized not in ("auto", "en", "vi", "zh"):
            normalized = "auto"
        self._language_pref = normalized
        self._lang = resolve_language(normalized)
        if persist:
            try:
                self._update_prefs.set_language(normalized)
            except OSError:
                pass
        self._sync_language_combo()
        self._retranslate_ui()

    def _set_status(self, key: str, **kwargs: Any) -> None:
        self._status_key = key
        self._status_kwargs = dict(kwargs)
        self.status_label.setText(tr(self._lang, key, **kwargs))

    def _format_elided_filename(self, name: str) -> str:
        return self.video_name_label.fontMetrics().elidedText(
            name, Qt.TextElideMode.ElideMiddle, 146
        )

    def _on_author_star_clicked(self) -> None:
        self._open_url_fn(GITHUB_REPO_URL)

    def _on_discord_rpc_settings_changed(self, _checked: bool = False) -> None:
        enabled = self.btn_rpc_on.isChecked()
        hide_filename = self.chk_rpc_hide_filename.isChecked()
        show_github = self.chk_rpc_show_github.isChecked()

        self._rpc_enabled = enabled
        self._rpc_hide_filename = hide_filename
        self._rpc_show_github = show_github
        self.rpc_privacy_widget.setVisible(enabled)

        try:
            self._update_prefs.set_discord_rpc_enabled(enabled)
            self._update_prefs.set_discord_rpc_hide_filename(hide_filename)
            self._update_prefs.set_discord_rpc_show_github_button(show_github)
        except OSError:
            pass

        self._discord_rpc.configure(
            enabled=enabled,
            hide_filename=hide_filename,
            show_github_button=show_github,
        )

    def _publish_discord_presence(
        self,
        *,
        mode: str,
        stage: str = "",
        progress_current: int = 0,
        progress_total: int = 0,
        urgent: bool | None = None,
    ) -> None:
        fname = self._video_path.name if self._video_path is not None else ""
        width = self._video_meta.width if self._video_meta is not None else 0
        height = self._video_meta.height if self._video_meta is not None else 0
        fps = float(self._video_meta.fps) if self._video_meta is not None else 0.0

        total_cues = 0
        flagged_cues = 0
        if mode == "reviewing":
            events = self.review_table.events()
            total_cues = len(events)
            flagged_cues = sum(1 for ev in events if ev.flagged)

        started_at = (
            self._rpc_extract_started_at
            if mode == "extracting" and self._rpc_extract_started_at > 0
            else self._rpc_session_started_at
        )
        self._discord_rpc.update(
            PresenceState(
                mode=mode,
                filename=fname,
                width=width,
                height=height,
                fps=fps,
                provider=self._rpc_extract_provider,
                ocr_model=self._rpc_extract_model,
                stage=stage,
                progress_current=progress_current,
                progress_total=progress_total,
                total_cues=total_cues,
                flagged_cues=flagged_cues,
                started_at=started_at,
            ),
            urgent=urgent,
        )

    def _on_review_events_changed(self) -> None:
        if self._extract_result is not None and not self._is_extracting():
            self._publish_discord_presence(mode="reviewing", urgent=True)

    def _retranslate_ui(self) -> None:
        lang = self._lang
        self._panel_titles[self.settings_panel].setText(tr(lang, "panel_settings"))
        self._panel_titles[self.monitor_panel].setText(tr(lang, "panel_monitor"))
        self._panel_titles[self.subtitles_panel].setText(tr(lang, "panel_subtitles"))

        self.lbl_render_method.setText(tr(lang, "render_method"))
        self.btn_render_gpu.setText(tr(lang, "render_gpu"))
        self.btn_render_cpu.setText(tr(lang, "render_cpu"))
        self.btn_render_gpu.setToolTip(
            "" if self._gpu_available else tr(lang, "gpu_unavailable_tip")
        )

        self.lbl_model.setText(tr(lang, "model_label"))
        self.ocr_model_combo.blockSignals(True)
        try:
            self.ocr_model_combo.setItemText(0, tr(lang, "model_precision"))
            self.ocr_model_combo.setItemText(1, tr(lang, "model_turbo"))
        finally:
            self.ocr_model_combo.blockSignals(False)
        self.ocr_model_combo.setToolTip(tr(lang, "model_tooltip"))

        self.lbl_conf.setText(tr(lang, "flag_threshold"))

        self.lbl_roi_preset.setText(tr(lang, "roi_preset"))
        if self.preset_combo.count() > 0:
            self.preset_combo.setItemText(0, tr(lang, "preset_none"))
        self.btn_save_preset.setToolTip(tr(lang, "preset_save_tip"))
        self.btn_delete_preset.setToolTip(tr(lang, "preset_delete_tip"))

        self.lbl_export_format.setText(tr(lang, "export_format"))
        self.lbl_output_folder.setText(tr(lang, "output_folder"))
        self.btn_out_same.setText(tr(lang, "output_same"))
        self.btn_out_custom.setText(tr(lang, "output_custom"))
        self.custom_dir_edit.setPlaceholderText(tr(lang, "output_custom_placeholder"))
        self.btn_browse_out_dir.setToolTip(tr(lang, "output_browse_tip"))

        self.lbl_language.setText(tr(lang, "language_label"))
        auto_idx = self.language_combo.findData("auto")
        if auto_idx >= 0:
            self.language_combo.setItemText(auto_idx, tr(lang, "lang_auto"))

        self.lbl_discord_rpc.setText(tr(lang, "discord_rpc"))
        self.btn_rpc_on.setText(tr(lang, "rpc_on"))
        self.btn_rpc_off.setText(tr(lang, "rpc_off"))
        self.chk_rpc_hide_filename.setText(tr(lang, "rpc_hide_filename"))
        self.chk_rpc_hide_filename.setToolTip(tr(lang, "rpc_hide_filename_tip"))
        self.chk_rpc_show_github.setText(tr(lang, "rpc_show_github"))
        self.chk_rpc_show_github.setToolTip(tr(lang, "rpc_show_github_tip"))

        self.btn_start.setText(tr(lang, "btn_extract"))
        self.btn_stop.setText(tr(lang, "btn_stop"))

        self.btn_check_update.setText(tr(lang, "btn_check_updates"))
        self.btn_check_update.setToolTip(tr(lang, "tip_check_updates"))
        self.btn_update_available.setToolTip(tr(lang, "tip_update_available"))
        if self._available_update is not None:
            self.btn_update_available.setText(
                tr(lang, "btn_update_available", version=self._available_update.latest_version)
            )
        self.btn_author_star.setToolTip(f"{tr(lang, 'star_github_tip')} — {GITHUB_REPO}")

        self.btn_browse_video.setToolTip(tr(lang, "open_video_tip"))
        self.btn_seek_start.setToolTip(tr(lang, "tip_seek_start"))
        self._update_play_button_icon()
        self.btn_step_back.setToolTip(tr(lang, "tip_step_back"))
        self.btn_step_fwd.setToolTip(tr(lang, "tip_step_fwd"))
        self.btn_seek_end.setToolTip(tr(lang, "tip_seek_end"))
        self._update_mute_button_icon()

        self.btn_set_in.setText(tr(lang, "trim_in"))
        self.btn_set_in.setToolTip(tr(lang, "tip_trim_in"))
        self.btn_set_out.setText(tr(lang, "trim_out"))
        self.btn_set_out.setToolTip(tr(lang, "tip_trim_out"))
        self.btn_clear_trim.setToolTip(tr(lang, "tip_trim_clear"))
        self._update_trim_display()

        self.btn_toggle_settings.setText(tr(lang, "btn_settings"))
        self.btn_toggle_settings.setToolTip(tr(lang, "tip_toggle_settings"))
        self.btn_test_frame.setText(tr(lang, "btn_test_frame"))
        self.btn_test_frame.setToolTip(tr(lang, "test_frame_tip"))

        self.btn_open_out_dir.setToolTip(tr(lang, "open_out_dir_tip"))
        self._update_roi_label(self.roi_canvas.roi())
        self._update_export_button()

        if self._status_key is not None:
            self.status_label.setText(tr(lang, self._status_key, **self._status_kwargs))

        self.roi_canvas.set_language(lang)
        self.review_table.set_language(lang)

    def _install_focus_tracking(self) -> None:
        if self._focus_tracking_active:
            return
        app = QApplication.instance()
        if app is not None:
            app.focusChanged.connect(self._on_app_focus_changed)
            app.installEventFilter(self)
            self._focus_tracking_active = True

    def _disconnect_focus_tracking(self) -> None:
        if not self._focus_tracking_active:
            return
        self._focus_tracking_active = False
        app = QApplication.instance()
        if app is not None:
            try:
                app.focusChanged.disconnect(self._on_app_focus_changed)
            except (RuntimeError, TypeError):
                pass
            try:
                app.removeEventFilter(self)
            except RuntimeError:
                pass

    def _panel_for_widget(self, widget: QWidget | None) -> QFrame | None:
        if widget is None:
            return None
        for panel in self._workbench_panels:
            if panel is widget or panel.isAncestorOf(widget):
                return panel
        return None

    def _set_active_panel(self, panel: QFrame) -> None:
        if panel not in self._workbench_panels:
            return
        self._active_panel = panel
        for pane in self._workbench_panels:
            is_active = pane is panel
            if bool(pane.property("activePanel")) != is_active:
                pane.setProperty("activePanel", is_active)
                style = pane.style()
                if style is not None:
                    style.unpolish(pane)
                    style.polish(pane)
                title = self._panel_titles.get(pane)
                if title is not None:
                    title_style = title.style()
                    if title_style is not None:
                        title_style.unpolish(title)
                        title_style.polish(title)
                pane.update()

    @Slot(QWidget, QWidget)
    def _on_app_focus_changed(
        self, _old: QWidget | None, new: QWidget | None
    ) -> None:
        if not self._focus_tracking_active:
            return
        panel = self._panel_for_widget(new)
        if panel is not None and panel.isVisible():
            self._set_active_panel(panel)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if (
            self._focus_tracking_active
            and event.type() == QEvent.Type.MouseButtonPress
            and isinstance(watched, QWidget)
        ):
            panel = self._panel_for_widget(watched)
            if panel is not None and panel.isVisible():
                self._set_active_panel(panel)
        return super().eventFilter(watched, event)

    def _on_toggle_settings(self, checked: bool) -> None:
        if not checked:
            sizes = self.workbench_splitter.sizes()
            if sizes and sizes[0] > 0:
                self._saved_settings_width = max(244, min(380, sizes[0]))
            self.settings_panel.setVisible(False)
            if self._active_panel is self.settings_panel:
                self._set_active_panel(self.monitor_panel)
            return

        self.settings_panel.setVisible(True)
        sizes = self.workbench_splitter.sizes()
        total = sum(sizes)
        if total > 0:
            left = max(244, min(380, self._saved_settings_width))
            rem = max(0, total - left)
            mid = int(rem * 12 / 23)
            right = max(0, rem - mid)
            self.workbench_splitter.setSizes([left, mid, right])

    def _is_extracting(self) -> bool:
        return self._worker_thread is not None

    def _is_frame_testing(self) -> bool:
        return self._frame_test_thread is not None

    def _has_retiring_threads(self) -> bool:
        return bool(self._retiring_threads)

    def _is_running(self) -> bool:
        return (
            self._is_extracting()
            or self._is_frame_testing()
            or self._has_retiring_threads()
        )

    def _update_roi_label(self, roi: RoiRect) -> None:
        if self._video_meta is not None:
            x1, y1, x2, y2 = roi.to_pixels(self._video_meta.width, self._video_meta.height)
            dims = f"{x2 - x1}×{y2 - y1} px"
            self.roi_label.setText(tr(self._lang, "roi_pill", dims=dims))
            self.roi_label.setToolTip(
                f"{dims} ({roi.x * 100:.1f}%, {roi.y * 100:.1f}%)"
            )
            return
        dims = f"{roi.w * 100:.0f}×{roi.h * 100:.0f}%"
        self.roi_label.setText(tr(self._lang, "roi_pill", dims=dims))
        self.roi_label.setToolTip(
            f"{roi.x * 100:.1f}%, {roi.y * 100:.1f}% · {roi.w * 100:.1f}×{roi.h * 100:.1f}%"
        )

    def _current_rel_ms(self) -> int:
        return max(0, int(self._current_frame_ms - self._start_pts_ms))

    def _update_trim_display(self) -> None:
        self.frame_slider.set_trim_range(self._trim_start_ms, self._trim_end_ms)
        has_trim = self._trim_start_ms is not None or self._trim_end_ms is not None
        if not has_trim:
            self.lbl_trim_range.setText(tr(self._lang, "trim_full"))
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
            self._set_status("status_trim_in_cleared_out")
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
            self._set_status("status_trim_out_zero")
            return
        self._trim_end_ms = val
        if self._trim_start_ms is not None and self._trim_start_ms >= self._trim_end_ms:
            self._trim_start_ms = None
            self._set_status("status_trim_out_cleared_in")
        self._update_trim_display()
        self._update_controls_state()

    def clear_trim(self) -> None:
        """Reset OP/ED trim marks back to full video range."""
        self._trim_start_ms = None
        self._trim_end_ms = None
        self._update_trim_display()
        self._update_controls_state()

    def _is_text_input_focused(self) -> bool:
        if QApplication.activePopupWidget() is not None:
            return True
        if self.review_table.table.state() == QAbstractItemView.State.EditingState:
            return True
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

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if not self._is_running() and extract_dropped_video_path(event.mimeData()) is not None:
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        if self._is_running():
            event.ignore()
            return
        video_path = extract_dropped_video_path(event.mimeData())
        if video_path is not None:
            event.acceptProposedAction()
            self.load_video(video_path)
            return
        event.ignore()

    def _on_video_dropped(self, path_str: str) -> None:
        if self._is_running():
            return
        self.load_video(path_str)

    def _on_review_jump_requested(self, rel_ms: int) -> None:
        if self._video_meta is None or self._is_running():
            return
        self.seek_to_ms(self._start_pts_ms + int(rel_ms))

    def _on_review_snap_start(self) -> None:
        if self._video_meta is None or self._is_running():
            return
        if not self.review_table.apply_playhead_to_selected_start(self._current_rel_ms()):
            self._set_status("status_snap_start_invalid")

    def _on_review_snap_end(self) -> None:
        if self._video_meta is None or self._is_running():
            return
        if not self.review_table.apply_playhead_to_selected_end(self._current_rel_ms()):
            self._set_status("status_snap_end_invalid")

    def _on_review_split_requested(self) -> None:
        if self._is_running():
            return
        playhead = self._current_rel_ms() if self._video_meta is not None else None
        self.review_table.split_selected_row(playhead)

    def _on_review_dirty_changed(self, dirty: bool) -> None:
        self._has_unexported_edits = bool(dirty)

    def _on_review_transport_key(self, event: object) -> None:
        if (
            isinstance(event, QKeyEvent)
            and not self._is_text_input_focused()
            and not self._is_running()
            and self._handle_transport_key(event)
        ):
            event.accept()

    def _handle_transport_key(self, event: QKeyEvent) -> bool:
        if self._video_meta is None:
            return False
        key = event.key()
        mods = event.modifiers()
        no_mod = mods == Qt.KeyboardModifier.NoModifier
        shift_only = mods == Qt.KeyboardModifier.ShiftModifier
        ctrl_only = mods == Qt.KeyboardModifier.ControlModifier

        if key == Qt.Key.Key_Space and no_mod:
            if not event.isAutoRepeat():
                self.toggle_playback()
            return True
        if key == Qt.Key.Key_I and no_mod:
            if not event.isAutoRepeat():
                self.set_trim_in()
            return True
        if key == Qt.Key.Key_O and no_mod:
            if not event.isAutoRepeat():
                self.set_trim_out()
            return True
        if key == Qt.Key.Key_M and no_mod:
            if not event.isAutoRepeat():
                self.toggle_mute()
            return True
        if (
            (key in (Qt.Key.Key_Left, Qt.Key.Key_Comma) and no_mod)
            or (key == Qt.Key.Key_Less and (no_mod or shift_only))
        ):
            self._step_frame(-1)
            return True
        if (
            (key in (Qt.Key.Key_Right, Qt.Key.Key_Period) and no_mod)
            or (key == Qt.Key.Key_Greater and (no_mod or shift_only))
        ):
            self._step_frame(1)
            return True
        if key == Qt.Key.Key_Left and shift_only:
            self._jump_ms(-1000)
            return True
        if key == Qt.Key.Key_Right and shift_only:
            self._jump_ms(1000)
            return True
        if key == Qt.Key.Key_Left and ctrl_only:
            self._jump_ms(-5000)
            return True
        if key == Qt.Key.Key_Right and ctrl_only:
            self._jump_ms(5000)
            return True
        if key == Qt.Key.Key_Home and no_mod:
            self._seek_to_start()
            return True
        if key == Qt.Key.Key_End and no_mod:
            self._seek_to_end()
            return True
        return False

    def open_output_folder(self) -> bool:
        """Open the folder containing the most recently exported subtitle files."""
        if self._last_export_dir is None or not self._last_export_dir.exists():
            return False
        return QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_export_dir)))

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if not self._is_text_input_focused() and not self._is_running():
            key = event.key()
            mods = event.modifiers()
            ctrl_only = mods == Qt.KeyboardModifier.ControlModifier
            ctrl_shift = mods == (
                Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
            )
            alt_only = mods == Qt.KeyboardModifier.AltModifier

            if key == Qt.Key.Key_S and alt_only:
                self.review_table.jump_to_selected_start()
                event.accept()
                return
            if key == Qt.Key.Key_E and alt_only:
                self.review_table.jump_to_selected_end()
                event.accept()
                return
            if key == Qt.Key.Key_BracketLeft and ctrl_only:
                self._on_review_snap_start()
                event.accept()
                return
            if key == Qt.Key.Key_BracketRight and ctrl_only:
                self._on_review_snap_end()
                event.accept()
                return
            if key == Qt.Key.Key_D and ctrl_only:
                self.review_table.mark_selected_ok()
                event.accept()
                return
            if key == Qt.Key.Key_M and ctrl_only:
                self.review_table.merge_selected_rows()
                event.accept()
                return
            if key == Qt.Key.Key_K and ctrl_only:
                self._on_review_split_requested()
                event.accept()
                return
            if key == Qt.Key.Key_Z and ctrl_only:
                self.review_table.undo()
                event.accept()
                return
            if (key == Qt.Key.Key_Y and ctrl_only) or (key == Qt.Key.Key_Z and ctrl_shift):
                self.review_table.redo()
                event.accept()
                return
            if key == Qt.Key.Key_F and ctrl_only:
                self.review_table.open_find_bar(focus_replace=False)
                event.accept()
                return
            if key == Qt.Key.Key_H and ctrl_only:
                self.review_table.open_find_bar(focus_replace=True)
                event.accept()
                return

            if self._handle_transport_key(event):
                event.accept()
                return
        super().keyPressEvent(event)

    def _selected_export_formats(self) -> list[str]:
        selected: list[str] = []
        if self.chk_ass.isChecked():
            selected.append(".ass")
        if self.chk_srt.isChecked():
            selected.append(".srt")
        if self.chk_vtt.isChecked():
            selected.append(".vtt")
        if self.chk_txt.isChecked():
            selected.append(".txt")
        return selected

    def _update_export_button(self) -> None:
        selected = self._selected_export_formats()
        if not selected:
            self.btn_export.setText(tr(self._lang, "btn_no_format"))
        elif len(selected) <= 2:
            self.btn_export.setText(
                tr(self._lang, "btn_export_formats", formats=" + ".join(selected))
            )
        else:
            self.btn_export.setText(
                tr(self._lang, "btn_export_formats", formats=f"({len(selected)})")
            )

        running = self._is_running()
        has_video = self._video_path is not None and self._video_meta is not None
        has_result = self._extract_result is not None
        self.btn_export.setEnabled(
            has_video and has_result and not running and bool(selected)
        )
        self.btn_open_out_dir.setEnabled(self._last_export_dir is not None and not running)

    def _update_controls_state(self) -> None:
        running = self._is_running()
        extracting = self._is_extracting()
        has_video = self._video_path is not None and self._video_meta is not None
        has_trim = self._trim_start_ms is not None or self._trim_end_ms is not None

        self.roi_canvas.setEnabled(not running)
        self.btn_browse_video.setEnabled(not running)
        self.preset_combo.setEnabled(not running)
        self.btn_save_preset.setEnabled(not running)
        self.btn_delete_preset.setEnabled(not running and self.preset_combo.currentIndex() > 0)

        self.btn_seek_start.setEnabled(has_video and not running)
        self.btn_play_pause.setEnabled(has_video and not running)
        self.btn_step_back.setEnabled(has_video and not running)
        self.frame_slider.setEnabled(has_video and not running)
        self.btn_step_fwd.setEnabled(has_video and not running)
        self.btn_seek_end.setEnabled(has_video and not running)
        self.btn_mute.setEnabled(has_video and not running)
        self.btn_set_in.setEnabled(has_video and not running)
        self.btn_set_out.setEnabled(has_video and not running)
        self.btn_clear_trim.setEnabled(has_video and has_trim and not running)

        custom_dir = self.btn_out_custom.isChecked()
        self.custom_dir_row_widget.setVisible(custom_dir)
        self.btn_out_same.setEnabled(not running)
        self.btn_out_custom.setEnabled(not running)
        self.custom_dir_edit.setEnabled(custom_dir and not running)
        self.btn_browse_out_dir.setEnabled(not running)
        for chip in self._format_chips:
            chip.setEnabled(not running)

        self.btn_test_frame.setEnabled(has_video and not running)
        self.btn_render_gpu.setEnabled(self._gpu_available and not running)
        self.btn_render_cpu.setEnabled(not running)
        self.ocr_model_combo.setEnabled(not running)
        self.spin_conf.setEnabled(not running)
        self.language_combo.setEnabled(not running)
        self.btn_start.setEnabled(has_video and not running)
        self.btn_stop.setEnabled(extracting)
        self._update_export_button()

    def _refresh_presets(self, select_name: str | None = None) -> None:
        names = self._preset_store.list_names()
        self.preset_combo.blockSignals(True)
        try:
            self.preset_combo.clear()
            self.preset_combo.addItem(tr(self._lang, "preset_none"))
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
        self._update_controls_state()

    def save_preset(self, name: str | None = None) -> bool:
        """Save current ROI and confidence threshold as a named preset."""
        preset_name = name
        if preset_name is None:
            default_name = (
                self.preset_combo.currentText()
                if self.preset_combo.currentIndex() > 0
                else ""
            )
            entered, ok = QInputDialog.getText(
                self,
                tr(self._lang, "dlg_save_preset_title"),
                tr(self._lang, "dlg_save_preset_prompt"),
                text=default_name,
            )
            if not ok:
                return False
            preset_name = entered

        clean_name = preset_name.strip()
        if not clean_name:
            return False

        preset = RoiPreset(
            name=clean_name,
            roi=self.roi_canvas.roi(),
            ass_template_path=None,
            conf_threshold=float(self.spin_conf.value()),
        )
        try:
            self._preset_store.save(preset)
        except OSError as exc:
            self._set_status("status_save_preset_error", error=exc)
            QMessageBox.warning(self, tr(self._lang, "dlg_preset_error_title"), str(exc))
            return False
        self._refresh_presets(select_name=clean_name)
        return True

    def delete_current_preset(self) -> bool:
        """Delete the currently selected ROI preset."""
        if self.preset_combo.currentIndex() <= 0:
            return False
        name = self.preset_combo.currentText()
        try:
            deleted = self._preset_store.delete(name)
        except OSError as exc:
            self._set_status("status_delete_preset_error", error=exc)
            QMessageBox.warning(self, tr(self._lang, "dlg_preset_error_title"), str(exc))
            return False
        self._refresh_presets()
        return deleted

    def _close_preview_reader(self) -> None:
        self.stop_playback()
        self._play_iter = None
        if self._preview_reader is not None:
            try:
                self._preview_reader.close()
            finally:
                self._preview_reader = None

    def _clear_session_results(self) -> None:
        """Release previous episode's extracted events and thumbnails from memory."""
        self._extract_result = None
        self._has_unexported_edits = False
        self._last_export_dir = None
        self.review_table.set_frame_pts_ms(None)
        self.review_table.clear()
        self.test_result_label.setText("")

    def load_video(self, video_path: Path | str) -> None:
        """Load a single video episode, clearing any previous episode's memory state."""
        self.review_table.commit_or_close_active_editor()
        if not self._confirm_discard_unsaved_edits():
            return
        self._stop_and_wait_worker()

        self._clear_session_results()
        self.review_table.set_frame_pts_ms(None)
        self._trim_start_ms = None
        self._trim_end_ms = None
        self._update_trim_display()
        self._current_frame_bgr = None
        self._current_frame_index = 0
        self._at_eof = False
        self._last_advance_error = False
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
            self.video_name_label.clear()
            self.video_name_label.setToolTip("")
            self.video_name_label.setVisible(False)
            self.video_meta_badge.clear()
            self.video_meta_badge.setVisible(False)
            self.frame_slider.setRange(0, 0)
            self._update_time_label()
            self._set_status("status_open_video_failed")
            self._update_roi_label(self.roi_canvas.roi())
            self._update_controls_state()
            self._publish_discord_presence(mode="idle", urgent=True)
            QMessageBox.warning(self, tr(self._lang, "dlg_open_video_error_title"), str(exc))
            return

        self._video_path = path
        self._video_meta = reader.metadata
        self._preview_reader = reader
        self._start_pts_ms = int(reader.metadata.start_pts_ms)
        self._duration_ms = max(0, reader.metadata.duration_ms)
        self._current_frame_ms = self._start_pts_ms
        self._current_frame_index = 0
        self.review_table.set_frame_step_ms(
            max(1, round(1000.0 / max(1.0, float(reader.metadata.fps))))
        )

        self.video_name_label.setText(self._format_elided_filename(path.name))
        self.video_name_label.setToolTip(str(path))
        self.video_name_label.setVisible(True)
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
        self._set_status(
            "status_cleaning_worker" if self._has_retiring_threads() else "status_ready"
        )
        self._update_roi_label(self.roi_canvas.roi())
        self.seek_to_ms(self._start_pts_ms)
        self._update_controls_state()
        self._publish_discord_presence(mode="preview", urgent=True)

    def _browse_video(self) -> None:
        start_dir = str(self._video_path.parent) if self._video_path is not None else ""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            tr(self._lang, "dlg_open_video_title"),
            start_dir,
            tr(self._lang, "dlg_open_video_filter"),
        )
        if file_path:
            self.load_video(file_path)

    def _browse_output_dir(self) -> None:
        start_dir = self.custom_dir_edit.text().strip() or (
            str(self._video_path.parent) if self._video_path is not None else ""
        )
        chosen = QFileDialog.getExistingDirectory(
            self, tr(self._lang, "dlg_select_out_dir"), start_dir
        )
        if chosen:
            self.btn_out_custom.setChecked(True)
            self.custom_dir_edit.setText(chosen)

    def _ensure_preview_reader(self) -> VideoReader | None:
        if self._video_path is None or self._is_extracting():
            return None
        if self._preview_reader is None:
            self._preview_reader = VideoReader(self._video_path)
        return self._preview_reader

    def _update_time_label(
        self,
        rel_ms: int | None = None,
        frame_idx: int | None = None,
    ) -> None:
        if self._video_meta is None:
            self.time_label.setText("0:00:00.00 / 0:00:00.00")
            return
        curr_rel = self._current_rel_ms() if rel_ms is None else max(0, int(rel_ms))
        curr_idx = self._current_frame_index if frame_idx is None else max(0, int(frame_idx))
        self.time_label.setText(
            f"{format_ass_time(curr_rel)} / {format_ass_time(self._duration_ms)} · #{curr_idx}"
        )

    def _display_preview_frame(
        self, frame_idx: int, pts_ms: int, full_bgr: np.ndarray
    ) -> None:
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
        self._update_time_label()

    def _update_play_button_icon(self) -> None:
        if self._is_playing:
            self.btn_play_pause.setIcon(make_icon("pause", color="#3EA6FF", size=13))
            self.btn_play_pause.setToolTip(tr(self._lang, "tip_pause"))
        else:
            self.btn_play_pause.setIcon(make_icon("play", size=13))
            self.btn_play_pause.setToolTip(tr(self._lang, "tip_play"))

    def _update_mute_button_icon(self) -> None:
        if self._audio_player.is_muted():
            self.btn_mute.setIcon(make_icon("volume_mute", color="#F08080", size=13))
            self.btn_mute.setToolTip(tr(self._lang, "tip_unmute"))
        else:
            self.btn_mute.setIcon(make_icon("volume", size=13))
            self.btn_mute.setToolTip(tr(self._lang, "tip_mute"))

    def _on_mute_toggled(self, checked: bool) -> None:
        self._audio_player.set_muted(checked)
        self._update_mute_button_icon()

    def toggle_mute(self) -> None:
        """Toggle preview audio mute on/off."""
        self.btn_mute.setChecked(not self.btn_mute.isChecked())

    def start_playback(self) -> None:
        """Start sequential preview playback on the video canvas with synchronized audio."""
        if self._video_meta is None or self._is_running():
            return
        if self._is_playing:
            return

        fps = max(1.0, float(self._video_meta.fps))
        frame_ms = max(1, round(1000.0 / fps))
        max_raw_ms = self._start_pts_ms + self._duration_ms
        wrapped = False
        if self._at_eof or (
            self._duration_ms > 0 and self._current_frame_ms >= max_raw_ms - frame_ms
        ):
            start_raw = self._start_pts_ms
            wrapped = True
        else:
            start_raw = self._current_frame_ms + 1

        try:
            reader = self._ensure_preview_reader()
            if reader is None:
                return
            self._play_iter = reader.iter_full_frames_from_ms(start_raw)
        except Exception as exc:
            self.stop_playback()
            self._play_iter = None
            self._set_status("status_playback_error", error=exc)
            return

        if wrapped:
            self._at_eof = False
        self._pending_play_frame = None
        audio_started = False
        if self._video_path is not None:
            audio_started = self._audio_player.start(self._video_path, start_raw)
        self._is_playing = True
        self._update_play_button_icon()
        interval_ms = 15 if audio_started else max(16, int(round(1000.0 / fps)))
        self._play_timer.start(interval_ms)

    def stop_playback(self) -> None:
        """Pause sequential preview playback and stop audio output."""
        if self._play_timer.isActive():
            self._play_timer.stop()
        self._audio_player.stop()
        if self._pending_play_frame is not None:
            self._pending_play_frame = None
            self._play_iter = None
        if self._is_playing:
            self._is_playing = False
            self._update_play_button_icon()

    def toggle_playback(self) -> None:
        """Toggle video preview playback on/off."""
        if self._is_playing:
            self.stop_playback()
        else:
            self.start_playback()

    def _pull_next_sequential_frame(self) -> tuple[int, int, np.ndarray] | None:
        if self._video_meta is None or self._is_running():
            return None
        self._last_advance_error = False
        try:
            reader = self._ensure_preview_reader()
            if reader is None:
                return None
            if self._play_iter is None:
                self._play_iter = reader.iter_full_frames_from_ms(self._current_frame_ms + 1)
        except Exception as exc:
            self._play_iter = None
            self._last_advance_error = True
            self.stop_playback()
            self._set_status("status_frame_read_error", error=exc)
            return None
        try:
            return next(self._play_iter)
        except StopIteration:
            self._play_iter = None
            self._at_eof = True
            return None
        except Exception as exc:
            self._play_iter = None
            self._last_advance_error = True
            self.stop_playback()
            self._set_status("status_frame_read_error", error=exc)
            return None

    def _advance_one_sequential_frame(self) -> bool:
        """Decode and display the next sequential frame without re-seeking; returns False at EOF."""
        nxt = self._pull_next_sequential_frame()
        if nxt is None:
            return False
        frame_idx, pts_ms, full_bgr = nxt
        self._at_eof = False
        self._display_preview_frame(frame_idx, pts_ms, full_bgr)
        return True

    @Slot()
    def _on_playback_tick(self) -> None:
        if not self._is_playing or self._video_meta is None or self._is_running():
            self.stop_playback()
            return

        audio_pts = self._audio_player.playback_pts_ms()
        if audio_pts is None:
            if not self._advance_one_sequential_frame():
                self.stop_playback()
            return

        fps = max(1.0, float(self._video_meta.fps))
        frame_ms = max(1, round(1000.0 / fps))
        half_frame_ms = max(1, frame_ms // 2)

        latest_due: tuple[int, int, np.ndarray] | None = None
        for _ in range(6):
            if self._pending_play_frame is None:
                self._pending_play_frame = self._pull_next_sequential_frame()
                if self._pending_play_frame is None:
                    if latest_due is not None:
                        self._at_eof = False
                        self._display_preview_frame(*latest_due)
                    self.stop_playback()
                    return
            _, pts_ms, _ = self._pending_play_frame
            if pts_ms > audio_pts + half_frame_ms:
                break
            latest_due = self._pending_play_frame
            self._pending_play_frame = None
            if pts_ms + frame_ms >= audio_pts:
                break

        if latest_due is not None:
            self._at_eof = False
            self._display_preview_frame(*latest_due)

    def seek_to_ms(self, target_ms: int) -> None:
        """Seek preview canvas to the frame nearest ``target_ms``."""
        if self._video_meta is None:
            return
        self.stop_playback()
        self._play_iter = None
        self._at_eof = False
        min_ms = self._start_pts_ms
        max_ms = self._start_pts_ms + self._duration_ms
        clamped = max(min_ms, min(int(target_ms), max_ms))
        try:
            reader = self._ensure_preview_reader()
            if reader is None:
                return
            frame_idx, pts_ms, full_bgr = reader.read_frame_at_ms(clamped)
        except Exception as exc:
            self._set_status("status_frame_seek_error", error=exc)
            return

        self._display_preview_frame(frame_idx, pts_ms, full_bgr)

    def _on_slider_moved(self, offset_ms: int) -> None:
        if self._video_meta is None:
            return
        clamped = max(0, min(int(offset_ms), self._duration_ms))
        fps = max(1.0, float(self._video_meta.fps))
        est_idx = int(round(clamped * fps / 1000.0))
        self._update_time_label(rel_ms=clamped, frame_idx=est_idx)

    def _on_slider_value_changed(self, offset_ms: int) -> None:
        if self._video_meta is None:
            return
        self.seek_to_ms(self._start_pts_ms + int(offset_ms))

    def _step_frame(self, delta: int) -> None:
        if self._video_meta is None or self._is_running():
            return
        self.stop_playback()
        if delta == 1:
            if self._at_eof:
                return
            if self._advance_one_sequential_frame():
                return
            if self._at_eof or self._last_advance_error:
                return
        elif delta == -1:
            self._play_iter = None
            try:
                reader = self._ensure_preview_reader()
                if reader is None:
                    return
                if hasattr(reader, "read_prev_frame_before_ms"):
                    full_bgr, pts_ms = reader.read_prev_frame_before_ms(self._current_frame_ms)
                    self._at_eof = False
                    fps = max(1.0, float(self._video_meta.fps))
                    rel_ms = max(0, pts_ms - self._start_pts_ms)
                    frame_idx = int(round(rel_ms * fps / 1000.0))
                    self._display_preview_frame(frame_idx, pts_ms, full_bgr)
                    return
            except Exception as exc:
                self._set_status("status_step_back_error", error=exc)
                return
        step_ms = max(1, round(1000.0 / max(1.0, self._video_meta.fps)))
        self.seek_to_ms(self._current_frame_ms + delta * step_ms)

    def _jump_ms(self, delta_ms: int) -> None:
        if self._video_meta is None or self._is_running():
            return
        self.seek_to_ms(self._current_frame_ms + int(delta_ms))

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
        provider = self.selected_provider()
        ocr_model = self.ocr_model_combo.currentData() or "accurate"
        return ExtractConfig(
            conf_threshold=float(self.spin_conf.value()),
            provider=str(provider),
            ocr_model=str(ocr_model),
            trim_start_ms=self._trim_start_ms,
            trim_end_ms=self._trim_end_ms,
        )

    def test_current_frame(self) -> bool:
        """Run stroke-mask + OCR on the currently previewed frame asynchronously."""
        if self._video_path is None or self._is_running():
            return False
        self.stop_playback()

        config = self.build_config()
        roi = self.roi_canvas.roi()

        try:
            full_bgr = self._current_frame_bgr
            if full_bgr is None:
                reader = self._ensure_preview_reader()
                if reader is None:
                    return False
                frame_idx, pts_ms, full_bgr = reader.read_frame_at_ms(self._current_frame_ms)
                self._current_frame_index = frame_idx
                self._current_frame_ms = pts_ms
                self._current_frame_bgr = full_bgr
            frame_copy = full_bgr.copy()
        except Exception as exc:
            err_text = tr(self._lang, "status_frame_test_error", error=exc)
            self.test_result_label.setText(err_text)
            self._set_status("status_frame_test_error", error=exc)
            return False

        self.test_result_label.setText(tr(self._lang, "status_testing_frame"))
        self._set_status("status_testing_frame")

        thread = QThread(self)
        worker = FrameTestWorker(
            pipeline=self._pipeline,
            frame_bgr=frame_copy,
            frame_index=self._current_frame_index,
            pts_ms=self._current_frame_ms,
            roi=roi,
            config=config,
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._on_frame_test_finished)
        worker.finished.connect(thread.quit)
        worker.failed.connect(self._on_frame_test_failed)
        worker.failed.connect(thread.quit)

        self._frame_test_thread = thread
        self._frame_test_worker = worker
        self._frame_test_bgr = frame_copy
        self._last_frame_test_result = None
        thread.start()
        self._update_controls_state()
        return True

    @Slot(object)
    def _on_frame_test_finished(self, result: FrameTestResult) -> None:
        if self.sender() is not self._frame_test_worker:
            return
        frame_bgr = self._frame_test_bgr
        self._cleanup_frame_test_worker()
        self._last_frame_test_result = result
        if frame_bgr is not None and self._current_frame_ms == result.pts_ms:
            self.roi_canvas.set_frame(frame_bgr, mask=result.mask)
        if result.text:
            self.test_result_label.setText(
                tr(
                    self._lang,
                    "status_frame_test_ocr",
                    text=result.text,
                    conf=result.confidence,
                )
            )
            self._set_status(
                "status_frame_test_done",
                text=result.text,
                conf=result.confidence,
            )
        else:
            self.test_result_label.setText(tr(self._lang, "status_frame_test_no_text"))
            self._set_status("status_frame_test_no_text_status")
        self._update_controls_state()

    @Slot(str)
    def _on_frame_test_failed(self, message: str) -> None:
        if self.sender() is not self._frame_test_worker:
            return
        self._cleanup_frame_test_worker()
        self.test_result_label.setText(
            tr(self._lang, "status_frame_test_error", error=message)
        )
        self._set_status("status_frame_test_error", error=message)
        self._update_controls_state()

    def start_extraction(self) -> None:
        """Launch single-episode extraction on a background QThread."""
        if self._video_path is None or self._is_running():
            return
        self.review_table.commit_or_close_active_editor()
        if not self._confirm_discard_unsaved_edits():
            return

        self._clear_session_results()
        self._close_preview_reader()

        self.progress_bar.setValue(0)
        self._set_status("status_extracting")

        cancel_event = threading.Event()
        config = self.build_config()
        thread = QThread(self)
        worker = _ExtractionWorker(
            pipeline=self._pipeline,
            video_path=self._video_path,
            roi=self.roi_canvas.roi(),
            config=config,
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
        self._rpc_extract_started_at = int(time.time())
        self._rpc_extract_provider = config.provider
        self._rpc_extract_model = config.ocr_model
        self._rpc_last_stage = ""
        self._publish_discord_presence(mode="extracting", urgent=True)
        thread.start()
        self._update_controls_state()

    def stop_extraction(self) -> None:
        """Request cooperative cancellation of the active worker."""
        if self._cancel_event is not None:
            self._cancel_event.set()
            self.btn_stop.setEnabled(False)
            self._set_status("status_stopping")

    @Slot(str, int, int)
    def _on_worker_progress(self, stage: str, current: int, total: int) -> None:
        if self.sender() is not self._worker:
            return
        if stage == "scan":
            pct = int(round(70.0 * current / max(1, total)))
            self.progress_bar.setValue(max(0, min(70, pct)))
            self._set_status(
                "status_scanning",
                current=format_ass_time(current),
                total=format_ass_time(total),
            )
            stage_changed = self._rpc_last_stage != "scan"
            self._rpc_last_stage = "scan"
            self._publish_discord_presence(
                mode="extracting",
                stage="scan",
                progress_current=current,
                progress_total=total,
                urgent=stage_changed,
            )
        elif stage == "ocr":
            pct = 70 + int(round(29.0 * current / max(1, total)))
            self.progress_bar.setValue(max(70, min(99, pct)))
            self._set_status("status_ocr_progress", current=current, total=total)
            stage_changed = self._rpc_last_stage != "ocr"
            self._rpc_last_stage = "ocr"
            self._publish_discord_presence(
                mode="extracting",
                stage="ocr",
                progress_current=current,
                progress_total=total,
                urgent=stage_changed,
            )
        elif stage == "done":
            self.progress_bar.setValue(100)
            self._set_status("status_done_simple", total=total)

    @Slot(object)
    def _on_worker_finished(self, result: ExtractResult) -> None:
        if self.sender() is not self._worker:
            return
        self._cleanup_worker()
        self._extract_result = result
        self._has_unexported_edits = False
        self.review_table.set_frame_pts_ms(result.frame_pts_ms)
        self.review_table.set_events(result.events)
        self.progress_bar.setValue(100)
        flagged_count = sum(1 for ev in result.events if ev.flagged)
        self._set_status(
            "status_done",
            total=len(result.events),
            flagged=flagged_count,
        )
        self._update_controls_state()
        self._publish_discord_presence(mode="reviewing", urgent=True)

    @Slot()
    def _on_worker_cancelled(self) -> None:
        if self.sender() is not self._worker:
            return
        self._cleanup_worker()
        self._clear_session_results()
        self.progress_bar.setValue(0)
        self._set_status("status_cancelled")
        self._update_controls_state()
        self._publish_discord_presence(
            mode="preview" if self._video_meta is not None else "idle",
            urgent=True,
        )

    @Slot(str)
    def _on_worker_failed(self, message: str) -> None:
        if self.sender() is not self._worker:
            return
        self._cleanup_worker()
        self._clear_session_results()
        self.progress_bar.setValue(0)
        self._set_status("status_extraction_failed")
        self._update_controls_state()
        self._publish_discord_presence(
            mode="preview" if self._video_meta is not None else "idle",
            urgent=True,
        )
        QMessageBox.critical(self, tr(self._lang, "dlg_extract_error_title"), message)

    def _stop_and_wait_worker(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
        self._cleanup_worker()
        self._cleanup_frame_test_worker()

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

    def _disconnect_frame_test_slots(self, worker: FrameTestWorker) -> None:
        for sig, slot in (
            (worker.finished, self._on_frame_test_finished),
            (worker.failed, self._on_frame_test_failed),
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
        if not self._retiring_threads and not self._is_running():
            if (
                self._status_key == "status_cleaning_worker"
                or self.status_label.text() == tr(self._lang, "status_cleaning_worker")
            ):
                self._set_status("status_ready")
        self._update_controls_state()

    def _finalize_retiring_thread(self, thread: QThread) -> None:
        if thread not in self._retiring_threads:
            return
        if thread.isRunning():
            thread.wait(200)
        if thread.isRunning():
            QTimer.singleShot(50, self, lambda t=thread: self._finalize_retiring_thread(t))
            return
        self._retire_thread(thread)

    @Slot()
    def _on_retiring_thread_finished(self) -> None:
        sender = self.sender()
        if isinstance(sender, QThread):
            self._finalize_retiring_thread(sender)

    def _cleanup_frame_test_worker(self) -> None:
        thread = self._frame_test_thread
        worker = self._frame_test_worker
        self._frame_test_thread = None
        self._frame_test_worker = None
        self._frame_test_bgr = None

        if worker is not None:
            self._disconnect_frame_test_slots(worker)

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
                else:
                    self._set_status("status_cleaning_worker")
                    self._update_controls_state()
        elif worker is not None:
            worker.deleteLater()

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
                else:
                    self._set_status("status_cleaning_worker")
                    self._update_controls_state()
        elif worker is not None:
            worker.deleteLater()

    def _confirm_discard_unsaved_edits(self) -> bool:
        if not self._has_unexported_edits:
            return True
        if self._unsaved_confirm_fn is not None:
            return bool(self._unsaved_confirm_fn())

        reply = QMessageBox.question(
            self,
            tr(self._lang, "dlg_unsaved_title"),
            tr(self._lang, "dlg_unsaved_body"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _confirm_overwrite(self, existing_paths: list[Path]) -> bool:
        if self._overwrite_confirm_fn is not None:
            return bool(self._overwrite_confirm_fn(existing_paths))

        names = "\n".join(f"- {p.name}" for p in existing_paths)
        reply = QMessageBox.question(
            self,
            tr(self._lang, "dlg_overwrite_title"),
            tr(self._lang, "dlg_overwrite_body", names=names),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def export_subtitles(self) -> bool:
        """Export current review events to selected .ass/.srt/.vtt/.txt files."""
        self.review_table.commit_or_close_active_editor()
        if self._video_path is None or self._extract_result is None:
            return False

        want_ass = self.chk_ass.isChecked()
        want_srt = self.chk_srt.isChecked()
        want_vtt = self.chk_vtt.isChecked()
        want_txt = self.chk_txt.isChecked()
        if not (want_ass or want_srt or want_vtt or want_txt):
            QMessageBox.information(
                self,
                tr(self._lang, "dlg_no_format_title"),
                tr(self._lang, "dlg_no_format_body"),
            )
            return False

        events = self.review_table.events()
        if not prepare_export_cues(events):
            QMessageBox.information(
                self,
                tr(self._lang, "dlg_no_cues_title"),
                tr(self._lang, "dlg_no_cues_body"),
            )
            return False

        if self.btn_out_same.isChecked():
            out_dir = self._video_path.parent
        else:
            custom = self.custom_dir_edit.text().strip()
            if not custom:
                QMessageBox.warning(
                    self,
                    tr(self._lang, "dlg_missing_out_dir_title"),
                    tr(self._lang, "dlg_missing_out_dir_body"),
                )
                return False
            out_dir = Path(custom)

        stem = self._video_path.stem
        ass_path = out_dir / f"{stem}.ass"
        srt_path = out_dir / f"{stem}.srt"
        vtt_path = out_dir / f"{stem}.vtt"
        txt_path = out_dir / f"{stem}.txt"

        targets: list[Path] = []
        if want_ass:
            targets.append(ass_path)
        if want_srt:
            targets.append(srt_path)
        if want_vtt:
            targets.append(vtt_path)
        if want_txt:
            targets.append(txt_path)

        existing = [p for p in targets if p.exists()]
        if existing and not self._confirm_overwrite(existing):
            return False

        try:
            if want_ass:
                export_ass(
                    events,
                    ass_path,
                    video_width=self._extract_result.width,
                    video_height=self._extract_result.height,
                    frame_pts_ms=self._extract_result.frame_pts_ms,
                )
            if want_srt:
                export_srt(events, srt_path)
            if want_vtt:
                export_vtt(events, vtt_path)
            if want_txt:
                export_txt(events, txt_path)
        except Exception as exc:
            self._set_status("status_export_error", error=exc)
            QMessageBox.warning(self, tr(self._lang, "dlg_export_error_title"), str(exc))
            return False

        self.review_table.mark_clean()
        self._has_unexported_edits = False
        self._last_export_dir = out_dir
        self._update_controls_state()
        self._set_status("status_export_done", files=", ".join(p.name for p in targets))
        return True

    def check_for_updates_async(self, *, manual: bool = False) -> bool:
        """Check GitHub Releases for a newer version on a background QThread."""
        if self._update_thread is not None:
            return False
        if manual:
            self.btn_check_update.setEnabled(False)
            self._set_status("status_checking_updates")

        thread = QThread(self)
        worker = _UpdateCheckWorker(
            check_fn=self._update_check_fn,
            current_version=__version__,
            manual=manual,
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._on_update_check_finished)
        worker.finished.connect(thread.quit)
        worker.failed.connect(self._on_update_check_failed)
        worker.failed.connect(thread.quit)

        self._update_thread = thread
        self._update_worker = worker
        thread.start()
        return True

    @Slot(object, bool)
    def _on_update_check_finished(self, info: UpdateInfo | None, manual: bool) -> None:
        if self.sender() is not self._update_worker:
            return
        self._cleanup_update_worker()
        self.btn_check_update.setEnabled(True)

        if info is not None:
            if not manual and self._update_prefs.is_version_skipped(info.tag_name):
                self.btn_update_available.setVisible(False)
                return
            self._available_update = info
            self.btn_update_available.setText(
                tr(self._lang, "btn_update_available", version=info.latest_version)
            )
            self.btn_update_available.setVisible(True)
            if manual:
                if (
                    self._status_key == "status_checking_updates"
                    or self.status_label.text() == tr(self._lang, "status_checking_updates")
                ):
                    self._set_status("status_update_available", version=info.latest_version)
                self.show_update_dialog(info)
            return

        self._available_update = None
        self.btn_update_available.setVisible(False)
        if manual:
            self._set_status("status_up_to_date", version=__version__)
            QMessageBox.information(
                self,
                tr(self._lang, "msg_up_to_date_title"),
                tr(self._lang, "msg_up_to_date_body", version=__version__),
            )

    @Slot(str, bool)
    def _on_update_check_failed(self, message: str, manual: bool) -> None:
        if self.sender() is not self._update_worker:
            return
        self._cleanup_update_worker()
        self.btn_check_update.setEnabled(True)
        if not manual:
            return
        self._set_status("status_update_failed")
        QMessageBox.warning(
            self,
            tr(self._lang, "msg_up_to_date_title"),
            tr(self._lang, "msg_update_failed_body", error=message),
        )

    def _on_update_badge_clicked(self) -> None:
        if self._available_update is not None:
            self.show_update_dialog(self._available_update)

    def show_update_dialog(self, info: UpdateInfo) -> str:
        """Show the UpdateDialog and apply Update Now / Skip This Version / Later."""
        dialog = UpdateDialog(info, lang=self._lang, parent=self)
        dialog.exec()
        if dialog.action == "update":
            target_url = info.download_url or info.release_url
            if target_url:
                self._open_url_fn(target_url)
        elif dialog.action == "skip":
            try:
                self._update_prefs.set_skipped_version(info.tag_name)
            except OSError:
                pass
            self.btn_update_available.setVisible(False)
        return dialog.action

    def _disconnect_update_slots(self, worker: _UpdateCheckWorker) -> None:
        for sig, slot in (
            (worker.finished, self._on_update_check_finished),
            (worker.failed, self._on_update_check_failed),
        ):
            try:
                sig.disconnect(slot)
            except (RuntimeError, TypeError):
                pass

    def _cleanup_update_worker(self) -> None:
        thread = self._update_thread
        worker = self._update_worker
        self._update_thread = None
        self._update_worker = None

        if worker is not None:
            self._disconnect_update_slots(worker)

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

    def closeEvent(self, event: QCloseEvent) -> None:
        """Stop background extraction, close open readers, and clear review memory."""
        self.review_table.commit_or_close_active_editor()
        if not self._confirm_discard_unsaved_edits():
            event.ignore()
            return
        self._disconnect_focus_tracking()
        self._discord_rpc.stop()
        if self._update_startup_timer.isActive():
            self._update_startup_timer.stop()
        self._cleanup_update_worker()
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
