"""In-memory subtitle review table with flagged filtering, timing sync, and power-user editing."""

from __future__ import annotations

from dataclasses import replace as dc_replace
import re

from PySide6.QtCore import QEvent, QItemSelectionModel, QMetaMethod, QObject, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QKeyEvent, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from transapp.core.models import SubtitleEvent
from transapp.core.postprocess import text_similarity
from transapp.exporters.srt_exporter import format_srt_time
from transapp.ui.icons import make_icon

_COL_STT = 0
_COL_START = 1
_COL_END = 2
_COL_CONF = 3
_COL_TEXT = 4

_MIN_DURATION_WARN_MS = 120
_DEFAULT_FRAME_STEP_MS = 40
_SHIFT_NUDGE_MS = 100
_FALLBACK_FLAG_REASON = "Cần kiểm tra"
_MAX_UNDO_DEPTH = 50
_MERGE_DEDUP_SIMILARITY = 0.80

_TIME_RE = re.compile(r"^\s*(\d+):([0-5]\d):([0-5]\d)(?:[.,](\d{1,3}))?\s*$")


def parse_subtitle_time(value: str) -> int | None:
    """Parse ASS ``H:MM:SS.cc``, SRT ``HH:MM:SS,mmm``, or ``HH:MM:SS.mmm`` into ms."""
    match = _TIME_RE.match(value)
    if match is None:
        return None
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    frac = match.group(4)
    millis = int(frac.ljust(3, "0")) if frac else 0
    return ((hours * 3600) + (minutes * 60) + seconds) * 1000 + millis


def _format_cell_time(ms: int) -> str:
    """Format milliseconds uniformly as ``HH:MM:SS.mmm``."""
    return format_srt_time(max(0, int(ms))).replace(",", ".")


def _encode_display_text(text: str) -> str:
    """Encode multi-line cue text using ``\\N`` so single-line inline edit preserves breaks."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.replace("\n", r"\N")


def _decode_display_text(text: str) -> str:
    """Decode ``\\N`` markers back into real newlines for ``SubtitleEvent.text``."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.replace(r"\N", "\n")


class _SubtitleTableWidget(QTableWidget):
    """QTableWidget keeping ExtendedSelection while making programmatic selectRow exclusive."""

    def selectRow(self, row: int) -> None:
        sel_model = self.selectionModel()
        if sel_model is None or not (0 <= row < self.rowCount()):
            self.clearSelection()
            return
        idx = self.model().index(row, 0)
        sel_model.setCurrentIndex(
            idx,
            QItemSelectionModel.SelectionFlag.ClearAndSelect
            | QItemSelectionModel.SelectionFlag.Rows,
        )


class ReviewTable(QWidget):
    """Editable subtitle review table with QA mini-inspector, video timing sync, and undo/redo."""

    events_changed = Signal()
    events_modified = Signal()
    jump_requested = Signal(int)
    snap_start_requested = Signal()
    snap_end_requested = Signal()
    split_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._events: list[SubtitleEvent] = []
        self._filter_mode: str = "all"
        self._populating: bool = False
        self._raw_pixmap: QPixmap | None = None
        self._frame_step_ms: int = _DEFAULT_FRAME_STEP_MS
        self._undo_stack: list[tuple[list[SubtitleEvent], int]] = []
        self._redo_stack: list[tuple[list[SubtitleEvent], int]] = []

        self._mono_font = QFont("Cascadia Mono")
        self._mono_font.setStyleHint(QFont.StyleHint.Monospace)
        self._mono_font.setPointSizeF(9.0)

        self._cjk_font = QFont("Microsoft YaHei UI")
        self._cjk_font.setPointSizeF(10.0)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        # === Compact Single-Row Header Toolbar ===
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        title = QLabel("RÀ SOÁT PHỤ ĐỀ")
        title.setProperty("role", "sectionTitle")
        header.addWidget(title)

        filter_pill = QFrame()
        filter_pill.setObjectName("filterPill")
        pill_layout = QHBoxLayout(filter_pill)
        pill_layout.setContentsMargins(2, 2, 2, 2)
        pill_layout.setSpacing(2)

        self.radio_all = QRadioButton("Tất cả (0)")
        self.radio_all.setChecked(True)
        self.radio_all.setCursor(Qt.CursorShape.PointingHandCursor)
        self.radio_flagged = QRadioButton("Chỉ dòng nghi ngờ (0)")
        self.radio_flagged.setCursor(Qt.CursorShape.PointingHandCursor)
        pill_layout.addWidget(self.radio_all)
        pill_layout.addWidget(self.radio_flagged)
        header.addWidget(filter_pill)

        self.btn_find = QPushButton()
        self.btn_find.setProperty("variant", "tool")
        self.btn_find.setIcon(make_icon("search", size=13))
        self.btn_find.setIconSize(QSize(13, 13))
        self.btn_find.setFixedSize(26, 26)
        self.btn_find.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_find.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_find.setToolTip("Tìm kiếm & thay thế (Ctrl+F / Ctrl+H)")

        self.btn_undo = QPushButton()
        self.btn_undo.setProperty("variant", "tool")
        self.btn_undo.setIcon(make_icon("undo", size=13))
        self.btn_undo.setIconSize(QSize(13, 13))
        self.btn_undo.setFixedSize(26, 26)
        self.btn_undo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_undo.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_undo.setToolTip("Hoàn tác (Ctrl+Z)")
        self.btn_undo.setEnabled(False)

        self.btn_redo = QPushButton()
        self.btn_redo.setProperty("variant", "tool")
        self.btn_redo.setIcon(make_icon("redo", size=13))
        self.btn_redo.setIconSize(QSize(13, 13))
        self.btn_redo.setFixedSize(26, 26)
        self.btn_redo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_redo.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_redo.setToolTip("Làm lại (Ctrl+Y / Ctrl+Shift+Z)")
        self.btn_redo.setEnabled(False)

        toolbar_sep = QFrame()
        toolbar_sep.setFixedSize(1, 18)
        toolbar_sep.setStyleSheet("background-color: #252A34;")

        self.btn_merge = QPushButton("Gộp")
        self.btn_merge.setProperty("variant", "timing")
        self.btn_merge.setIcon(make_icon("merge", size=12))
        self.btn_merge.setIconSize(QSize(12, 12))
        self.btn_merge.setFixedHeight(26)
        self.btn_merge.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_merge.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_merge.setToolTip("Gộp các dòng liền kề đang chọn (Ctrl+M)")
        self.btn_merge.setEnabled(False)

        self.btn_split = QPushButton("Tách")
        self.btn_split.setProperty("variant", "timing")
        self.btn_split.setIcon(make_icon("split", size=12))
        self.btn_split.setIconSize(QSize(12, 12))
        self.btn_split.setFixedHeight(26)
        self.btn_split.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_split.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_split.setToolTip("Tách dòng đang chọn tại playhead hoặc điểm giữa (Ctrl+K)")
        self.btn_split.setEnabled(False)

        self.btn_delete_row = QPushButton()
        self.btn_delete_row.setProperty("variant", "danger")
        self.btn_delete_row.setIcon(make_icon("trash", color="#FCA5A5", size=14))
        self.btn_delete_row.setIconSize(QSize(14, 14))
        self.btn_delete_row.setFixedSize(28, 26)
        self.btn_delete_row.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_delete_row.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_delete_row.setToolTip("Xóa các dòng đang chọn khỏi kết quả (Delete)")
        self.btn_delete_row.setEnabled(False)

        header.addWidget(self.btn_find)
        header.addWidget(self.btn_undo)
        header.addWidget(self.btn_redo)
        header.addWidget(toolbar_sep)
        header.addWidget(self.btn_merge)
        header.addWidget(self.btn_split)
        header.addWidget(self.btn_delete_row)
        header.addStretch(1)

        self._header_actions_layout = QHBoxLayout()
        self._header_actions_layout.setContentsMargins(0, 0, 0, 0)
        self._header_actions_layout.setSpacing(6)
        header.addLayout(self._header_actions_layout)
        root.addLayout(header)

        # === Collapsible Find & Replace Bar (hidden by default) ===
        self.find_bar = QFrame()
        self.find_bar.setObjectName("findReplaceBar")
        find_layout = QHBoxLayout(self.find_bar)
        find_layout.setContentsMargins(6, 3, 6, 3)
        find_layout.setSpacing(5)

        self.find_edit = QLineEdit()
        self.find_edit.setPlaceholderText("Tìm kiếm...")
        self.find_edit.setMinimumWidth(130)

        self.replace_edit = QLineEdit()
        self.replace_edit.setPlaceholderText("Thay thế...")
        self.replace_edit.setMinimumWidth(130)

        self.btn_find_prev = QPushButton()
        self.btn_find_prev.setProperty("variant", "tool")
        self.btn_find_prev.setIcon(make_icon("step_back", size=12))
        self.btn_find_prev.setIconSize(QSize(12, 12))
        self.btn_find_prev.setFixedSize(24, 24)
        self.btn_find_prev.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_find_prev.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_find_prev.setToolTip("Kết quả trước (Shift+Enter)")

        self.btn_find_next = QPushButton()
        self.btn_find_next.setProperty("variant", "tool")
        self.btn_find_next.setIcon(make_icon("step_fwd", size=12))
        self.btn_find_next.setIconSize(QSize(12, 12))
        self.btn_find_next.setFixedSize(24, 24)
        self.btn_find_next.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_find_next.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_find_next.setToolTip("Kết quả tiếp (Enter)")

        self.btn_replace_one = QPushButton("Thay")
        self.btn_replace_one.setProperty("variant", "timing")
        self.btn_replace_one.setFixedHeight(24)
        self.btn_replace_one.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_replace_one.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_replace_one.setToolTip("Thay thế dòng đang chọn")

        self.btn_replace_all = QPushButton("Thay tất cả")
        self.btn_replace_all.setProperty("variant", "timing")
        self.btn_replace_all.setFixedHeight(24)
        self.btn_replace_all.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_replace_all.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_replace_all.setToolTip("Thay thế toàn bộ kết quả khớp (1 bước Undo)")

        self.lbl_find_status = QLabel("")
        self.lbl_find_status.setObjectName("findStatusLabel")

        self.btn_close_find = QPushButton()
        self.btn_close_find.setProperty("variant", "tool")
        self.btn_close_find.setIcon(make_icon("x_mark", color="#9BA4B5", size=11))
        self.btn_close_find.setIconSize(QSize(11, 11))
        self.btn_close_find.setFixedSize(24, 24)
        self.btn_close_find.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_close_find.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_close_find.setToolTip("Đóng tìm kiếm (Esc)")

        find_layout.addWidget(self.find_edit, 1)
        find_layout.addWidget(self.btn_find_prev)
        find_layout.addWidget(self.btn_find_next)
        find_layout.addWidget(self.replace_edit, 1)
        find_layout.addWidget(self.btn_replace_one)
        find_layout.addWidget(self.btn_replace_all)
        find_layout.addWidget(self.lbl_find_status)
        find_layout.addWidget(self.btn_close_find)
        self.find_bar.setVisible(False)
        root.addWidget(self.find_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.table = _SubtitleTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["#", "Start", "End", "Conf", "Nội dung phụ đề"])
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(27)
        hdr = self.table.horizontalHeader()
        hdr.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        hdr.setSectionResizeMode(_COL_STT, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(_COL_START, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(_COL_END, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(_COL_CONF, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(_COL_TEXT, QHeaderView.ResizeMode.Stretch)

        # === Right-hand 4-Row Timing & QA Mini-Inspector ===
        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(6, 0, 0, 0)
        preview_layout.setSpacing(5)

        # Row 1: Header + Cue Meta pill
        inspector_hdr = QHBoxLayout()
        inspector_hdr.setContentsMargins(0, 0, 0, 0)
        inspector_hdr.setSpacing(6)
        preview_title = QLabel("ĐỐI CHIẾU & TIMING")
        preview_title.setProperty("role", "sectionTitle")
        self.lbl_cue_meta = QLabel("")
        self.lbl_cue_meta.setObjectName("cueMetaBadge")
        self.lbl_cue_meta.setProperty("warning", False)
        self.lbl_cue_meta.setVisible(False)
        inspector_hdr.addWidget(preview_title)
        inspector_hdr.addStretch(1)
        inspector_hdr.addWidget(self.lbl_cue_meta)
        preview_layout.addLayout(inspector_hdr)

        # Row 2: Flag / QA Banner (hidden when selected cue is clean)
        self.flag_banner = QFrame()
        self.flag_banner.setObjectName("flagBanner")
        flag_layout = QHBoxLayout(self.flag_banner)
        flag_layout.setContentsMargins(7, 3, 5, 3)
        flag_layout.setSpacing(6)
        self.lbl_flag_reason = QLabel("")
        self.lbl_flag_reason.setObjectName("flagReasonLabel")
        self.lbl_flag_reason.setWordWrap(False)
        self.lbl_flag_reason.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.btn_mark_ok = QPushButton("Duyệt")
        self.btn_mark_ok.setProperty("variant", "ok")
        self.btn_mark_ok.setIcon(make_icon("check", color="#6EE7B7", size=12))
        self.btn_mark_ok.setIconSize(QSize(12, 12))
        self.btn_mark_ok.setFixedHeight(22)
        self.btn_mark_ok.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.btn_mark_ok.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_mark_ok.setToolTip("Đánh dấu dòng đã kiểm tra xong (Ctrl+D)")
        flag_layout.addWidget(self.lbl_flag_reason, 1)
        flag_layout.addWidget(self.btn_mark_ok)
        self.flag_banner.setVisible(False)
        preview_layout.addWidget(self.flag_banner)

        # Row 3: ROI Crop Thumbnail
        self.preview_label = QLabel("Chọn một dòng để xem ảnh cắt ROI")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(240, 76)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.preview_label.setStyleSheet(
            "background-color: #0A0C10; color: #6E7789; "
            "border: 1px solid #252A34; border-radius: 6px; padding: 6px;"
        )
        preview_layout.addWidget(self.preview_label, 1)

        # Row 4: Compact Timing Controls
        timing_row = QHBoxLayout()
        timing_row.setContentsMargins(0, 0, 0, 0)
        timing_row.setSpacing(4)

        self.btn_jump_start = QPushButton("Start")
        self.btn_jump_start.setIcon(make_icon("skip_back", size=11))
        self.btn_jump_start.setIconSize(QSize(11, 11))
        self.btn_jump_start.setToolTip("Nhảy video tới Start của dòng đang chọn (Alt+S)")

        self.btn_jump_end = QPushButton("End")
        self.btn_jump_end.setIcon(make_icon("skip_fwd", size=11))
        self.btn_jump_end.setIconSize(QSize(11, 11))
        self.btn_jump_end.setToolTip("Nhảy video tới End của dòng đang chọn (Alt+E)")

        self.btn_snap_in = QPushButton("← In")
        self.btn_snap_in.setToolTip("Đặt Start theo frame video hiện tại (Ctrl+[)")

        self.btn_snap_out = QPushButton("Out →")
        self.btn_snap_out.setToolTip("Đặt End theo frame video hiện tại (Ctrl+])")

        self.btn_nudge_back = QPushButton("-1f")
        self.btn_nudge_back.setToolTip("Dịch thời gian lùi 1 frame — giữ để lặp (Shift: -100ms)")

        self.btn_nudge_fwd = QPushButton("+1f")
        self.btn_nudge_fwd.setToolTip("Dịch thời gian tiến 1 frame — giữ để lặp (Shift: +100ms)")

        for nudge_btn in (self.btn_nudge_back, self.btn_nudge_fwd):
            nudge_btn.setAutoRepeat(True)
            nudge_btn.setAutoRepeatDelay(250)
            nudge_btn.setAutoRepeatInterval(60)

        self._timing_buttons = (
            self.btn_jump_start,
            self.btn_jump_end,
            self.btn_snap_in,
            self.btn_snap_out,
            self.btn_nudge_back,
            self.btn_nudge_fwd,
        )
        for btn in self._timing_buttons:
            btn.setProperty("variant", "timing")
            btn.setFixedHeight(24)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setEnabled(False)
            timing_row.addWidget(btn)
        timing_row.addStretch(1)
        preview_layout.addLayout(timing_row)

        splitter.addWidget(self.table)
        splitter.addWidget(preview_panel)
        splitter.setStretchFactor(0, 16)
        splitter.setStretchFactor(1, 9)
        root.addWidget(splitter, 1)

        self.radio_all.toggled.connect(self._on_filter_radio_toggled)
        self.btn_find.clicked.connect(lambda: self.open_find_bar(focus_replace=False))
        self.btn_undo.clicked.connect(self.undo)
        self.btn_redo.clicked.connect(self.redo)
        self.btn_merge.clicked.connect(self.merge_selected_rows)
        self.btn_split.clicked.connect(self._trigger_split)
        self.btn_delete_row.clicked.connect(self.delete_selected_rows)

        self.find_edit.textChanged.connect(lambda _: self._refresh_find_status())
        self.btn_find_prev.clicked.connect(self.find_prev)
        self.btn_find_next.clicked.connect(self.find_next)
        self.btn_replace_one.clicked.connect(self.replace_current)
        self.btn_replace_all.clicked.connect(lambda: self.replace_all())
        self.btn_close_find.clicked.connect(self.close_find_bar)

        self.btn_mark_ok.clicked.connect(self.mark_selected_ok)
        self.btn_jump_start.clicked.connect(self.jump_to_selected_start)
        self.btn_jump_end.clicked.connect(self.jump_to_selected_end)
        self.btn_snap_in.clicked.connect(self.snap_start_requested.emit)
        self.btn_snap_out.clicked.connect(self.snap_end_requested.emit)
        self.btn_nudge_back.clicked.connect(lambda: self._on_nudge_clicked(-1))
        self.btn_nudge_fwd.clicked.connect(lambda: self._on_nudge_clicked(1))
        self.find_edit.installEventFilter(self)
        self.replace_edit.installEventFilter(self)
        self.preview_label.installEventFilter(self)
        self.table.installEventFilter(self)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.itemChanged.connect(self._on_item_changed)

    def add_header_widget(self, widget: QWidget) -> None:
        """Append an action widget to the right side of the review toolbar."""
        self._header_actions_layout.addWidget(widget)

    def set_frame_step_ms(self, step_ms: int) -> None:
        """Configure the 1-frame nudge step in milliseconds."""
        self._frame_step_ms = max(1, int(step_ms))

    def filter_mode(self) -> str:
        """Return active filter mode (``"all"`` or ``"flagged_only"``)."""
        return self._filter_mode

    def set_events(self, events: list[SubtitleEvent]) -> None:
        """Replace in-memory subtitle events, reset undo/redo stacks, and refresh the table."""
        self._events = list(events)
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._update_filter_counts()
        self._rebuild_rows()
        self._refresh_find_status()

    def events(self) -> list[SubtitleEvent]:
        """Return all in-memory subtitle events including inline edits."""
        return list(self._events)

    def set_filter_mode(self, mode: str) -> None:
        """Set filter mode to ``"all"`` or ``"flagged_only"``."""
        if mode not in ("all", "flagged_only"):
            raise ValueError(f"Unsupported filter mode: {mode!r}")
        if mode == self._filter_mode:
            return
        self._filter_mode = mode
        self.radio_all.blockSignals(True)
        self.radio_flagged.blockSignals(True)
        try:
            self.radio_all.setChecked(mode == "all")
            self.radio_flagged.setChecked(mode == "flagged_only")
        finally:
            self.radio_all.blockSignals(False)
            self.radio_flagged.blockSignals(False)
        self._rebuild_rows()
        self._refresh_find_status()

    def clear(self) -> None:
        """Clear all events, undo/redo stacks, table rows, inspector state, and thumbnail."""
        self._events = []
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._raw_pixmap = None
        self._populating = True
        try:
            self.table.setRowCount(0)
        finally:
            self._populating = False
        self.preview_label.clear()
        self.preview_label.setText("Chưa chọn dòng")
        self._update_filter_counts()
        self._update_inspector_for_row(-1)
        self._update_action_buttons_state()
        self._refresh_find_status()

    # === Undo / Redo Snapshot Helpers ===

    def _snapshot_events(self) -> list[SubtitleEvent]:
        """Create fresh SubtitleEvent copies sharing immutable ``thumbnail_png`` bytes."""
        return [dc_replace(ev) for ev in self._events]

    def _push_undo_snapshot(self) -> None:
        preferred_row = max(0, self._selected_visible_row())
        self._undo_stack.append((self._snapshot_events(), preferred_row))
        if len(self._undo_stack) > _MAX_UNDO_DEPTH:
            del self._undo_stack[:-_MAX_UNDO_DEPTH]
        self._redo_stack.clear()
        self._update_action_buttons_state()

    def can_undo(self) -> bool:
        """Return True if an undo snapshot is available."""
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        """Return True if a redo snapshot is available."""
        return bool(self._redo_stack)

    def undo(self) -> bool:
        """Restore the previous event list snapshot and push current state to redo stack."""
        if not self._undo_stack:
            return False
        current_row = max(0, self._selected_visible_row())
        self._redo_stack.append((self._snapshot_events(), current_row))
        if len(self._redo_stack) > _MAX_UNDO_DEPTH:
            del self._redo_stack[:-_MAX_UNDO_DEPTH]

        prev_events, preferred_row = self._undo_stack.pop()
        self._events = [dc_replace(ev) for ev in prev_events]
        self._update_filter_counts()
        self._rebuild_rows(preferred_row=preferred_row)
        self._refresh_find_status()
        self.events_changed.emit()
        self.events_modified.emit()
        return True

    def redo(self) -> bool:
        """Re-apply the most recently undone snapshot."""
        if not self._redo_stack:
            return False
        current_row = max(0, self._selected_visible_row())
        self._undo_stack.append((self._snapshot_events(), current_row))
        if len(self._undo_stack) > _MAX_UNDO_DEPTH:
            del self._undo_stack[:-_MAX_UNDO_DEPTH]

        next_events, preferred_row = self._redo_stack.pop()
        self._events = [dc_replace(ev) for ev in next_events]
        self._update_filter_counts()
        self._rebuild_rows(preferred_row=preferred_row)
        self._refresh_find_status()
        self.events_changed.emit()
        self.events_modified.emit()
        return True

    def _reindex_events(self) -> None:
        for idx, ev in enumerate(self._events, start=1):
            ev.index = idx

    # === Selection Helpers ===

    def _selected_visible_rows(self) -> list[int]:
        sel_model = self.table.selectionModel()
        if sel_model is not None:
            rows = sorted({idx.row() for idx in sel_model.selectedRows()})
            if not rows:
                rows = sorted({idx.row() for idx in sel_model.selectedIndexes()})
            rows = [r for r in rows if 0 <= r < self.table.rowCount()]
            if rows:
                return rows
        return []

    def _selected_visible_and_master_pairs(self) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for row in self._selected_visible_rows():
            stt_item = self.table.item(row, _COL_STT)
            if stt_item is None:
                continue
            master_idx = stt_item.data(Qt.ItemDataRole.UserRole)
            if master_idx is not None and 0 <= int(master_idx) < len(self._events):
                pairs.append((row, int(master_idx)))
        return pairs

    def _selected_visible_row(self) -> int:
        rows = self._selected_visible_rows()
        if not rows:
            return -1
        current = self.table.currentRow()
        if current in rows:
            return current
        return rows[0]

    def _selected_row_and_master_idx(self) -> tuple[int, int] | None:
        row = self._selected_visible_row()
        if row < 0 or row >= self.table.rowCount():
            return None
        stt_item = self.table.item(row, _COL_STT)
        if stt_item is None:
            return None
        master_idx = stt_item.data(Qt.ItemDataRole.UserRole)
        if master_idx is None or not (0 <= master_idx < len(self._events)):
            return None
        return (row, int(master_idx))

    def _contiguous_selected_master_range(self) -> tuple[int, int, int] | None:
        pairs = self._selected_visible_and_master_pairs()
        if len(pairs) < 2:
            return None
        master_indices = sorted({master_idx for _, master_idx in pairs})
        if len(master_indices) < 2:
            return None
        if master_indices[-1] - master_indices[0] + 1 != len(master_indices):
            return None
        return (pairs[0][0], master_indices[0], master_indices[-1])

    def _can_split_selected(self) -> bool:
        pairs = self._selected_visible_and_master_pairs()
        if len(pairs) != 1:
            return False
        ev = self._events[pairs[0][1]]
        return (ev.end_ms - ev.start_ms) >= 2

    def selected_event(self) -> SubtitleEvent | None:
        """Return the currently selected subtitle event, or None."""
        sel = self._selected_row_and_master_idx()
        if sel is None:
            return None
        return self._events[sel[1]]

    def _set_timing_buttons_enabled(self, enabled: bool) -> None:
        for btn in self._timing_buttons:
            btn.setEnabled(enabled)

    def _set_meta_warning(self, warning: bool) -> None:
        if bool(self.lbl_cue_meta.property("warning")) == warning:
            return
        self.lbl_cue_meta.setProperty("warning", warning)
        style = self.lbl_cue_meta.style()
        if style is not None:
            style.unpolish(self.lbl_cue_meta)
            style.polish(self.lbl_cue_meta)

    def _update_action_buttons_state(self) -> None:
        has_selection = bool(self._selected_visible_and_master_pairs())
        self.btn_delete_row.setEnabled(has_selection)
        self.btn_merge.setEnabled(self._contiguous_selected_master_range() is not None)
        self.btn_split.setEnabled(self._can_split_selected())
        self.btn_undo.setEnabled(self.can_undo())
        self.btn_redo.setEnabled(self.can_redo())

    def _update_delete_button_state(self) -> None:
        self._update_action_buttons_state()

    # === Timing & QA Operations ===

    def jump_to_selected_start(self) -> bool:
        """Emit ``jump_requested`` for the selected cue's ``start_ms``."""
        ev = self.selected_event()
        if ev is None:
            return False
        self.jump_requested.emit(int(ev.start_ms))
        return True

    def jump_to_selected_end(self) -> bool:
        """Emit ``jump_requested`` for the selected cue's ``end_ms``."""
        ev = self.selected_event()
        if ev is None:
            return False
        self.jump_requested.emit(int(ev.end_ms))
        return True

    def mark_selected_ok(self) -> bool:
        """Clear flagged status on the selected cue in-place without evicting it."""
        sel = self._selected_row_and_master_idx()
        if sel is None:
            return False
        row, master_idx = sel
        ev = self._events[master_idx]
        if not ev.flagged and not ev.flag_reason:
            return False
        self._push_undo_snapshot()
        ev.flagged = False
        ev.flag_reason = ""
        self._refresh_row_in_place(row, master_idx)
        self.events_changed.emit()
        self.events_modified.emit()
        return True

    def apply_playhead_to_selected_start(self, rel_ms: int) -> bool:
        """Set selected cue's ``start_ms`` to ``rel_ms`` if strictly before ``end_ms``."""
        sel = self._selected_row_and_master_idx()
        if sel is None:
            return False
        row, master_idx = sel
        ev = self._events[master_idx]
        new_start = max(0, int(rel_ms))
        if new_start >= ev.end_ms:
            return False
        if new_start == ev.start_ms:
            return True
        self._push_undo_snapshot()
        ev.start_ms = new_start
        start_item = self.table.item(row, _COL_START)
        if start_item is not None:
            self._populating = True
            try:
                start_item.setText(_format_cell_time(ev.start_ms))
            finally:
                self._populating = False
        self._update_inspector_for_row(row)
        self._update_action_buttons_state()
        self.events_changed.emit()
        self.events_modified.emit()
        return True

    def apply_playhead_to_selected_end(self, rel_ms: int) -> bool:
        """Set selected cue's ``end_ms`` to ``rel_ms`` if strictly after ``start_ms``."""
        sel = self._selected_row_and_master_idx()
        if sel is None:
            return False
        row, master_idx = sel
        ev = self._events[master_idx]
        new_end = max(0, int(rel_ms))
        if new_end <= ev.start_ms:
            return False
        if new_end == ev.end_ms:
            return True
        self._push_undo_snapshot()
        ev.end_ms = new_end
        end_item = self.table.item(row, _COL_END)
        if end_item is not None:
            self._populating = True
            try:
                end_item.setText(_format_cell_time(ev.end_ms))
            finally:
                self._populating = False
        self._update_inspector_for_row(row)
        self._update_action_buttons_state()
        self.events_changed.emit()
        self.events_modified.emit()
        return True

    def nudge_selected_timing(self, delta_ms: int) -> bool:
        """Shift selected cue's ``(start_ms, end_ms)`` by ``delta_ms``, clamped to ``>= 0``."""
        sel = self._selected_row_and_master_idx()
        if sel is None or delta_ms == 0:
            return False
        row, master_idx = sel
        ev = self._events[master_idx]
        duration = max(1, ev.end_ms - ev.start_ms)
        new_start = max(0, ev.start_ms + int(delta_ms))
        new_end = new_start + duration
        if new_start == ev.start_ms and new_end == ev.end_ms:
            return False
        self._push_undo_snapshot()
        ev.start_ms = new_start
        ev.end_ms = new_end
        self._populating = True
        try:
            start_item = self.table.item(row, _COL_START)
            end_item = self.table.item(row, _COL_END)
            if start_item is not None:
                start_item.setText(_format_cell_time(ev.start_ms))
            if end_item is not None:
                end_item.setText(_format_cell_time(ev.end_ms))
        finally:
            self._populating = False
        self._update_inspector_for_row(row)
        self.events_changed.emit()
        self.events_modified.emit()
        return True

    def _on_nudge_clicked(self, sign: int) -> None:
        mods = QApplication.keyboardModifiers()
        step = _SHIFT_NUDGE_MS if (mods & Qt.KeyboardModifier.ShiftModifier) else self._frame_step_ms
        self.nudge_selected_timing(sign * step)

    # === Multi-Row Delete, Merge & Split ===

    def delete_selected_rows(self) -> bool:
        """Remove all currently selected subtitle events in one undoable step and re-index."""
        pairs = self._selected_visible_and_master_pairs()
        if not pairs:
            return False

        first_visible_row = pairs[0][0]
        master_indices = sorted({master_idx for _, master_idx in pairs}, reverse=True)

        self._push_undo_snapshot()
        for master_idx in master_indices:
            del self._events[master_idx]
        self._reindex_events()

        self._update_filter_counts()
        self._rebuild_rows(preferred_row=first_visible_row)
        self._refresh_find_status()
        self.events_changed.emit()
        self.events_modified.emit()
        return True

    def delete_selected_row(self) -> bool:
        """Backward-compatible alias delegating to ``delete_selected_rows``."""
        return self.delete_selected_rows()

    def merge_selected_rows(self) -> bool:
        """Merge >=2 contiguous selected rows into a single cue in one undoable step."""
        span = self._contiguous_selected_master_range()
        if span is None:
            return False
        first_visible_row, m0, mk = span
        group = self._events[m0 : mk + 1]

        start_ms = min(ev.start_ms for ev in group)
        end_ms = max(ev.end_ms for ev in group)
        best_ev = max(group, key=lambda e: (e.confidence, len(e.text)))
        flagged = any(ev.flagged for ev in group)
        if flagged:
            reasons = list(
                dict.fromkeys(
                    ev.flag_reason.strip()
                    for ev in group
                    if ev.flagged and ev.flag_reason.strip()
                )
            )
            flag_reason = "; ".join(reasons)
        else:
            flag_reason = ""

        # Deduplicate near-identical adjacent texts (>= 0.80), otherwise join with \n
        merged_parts: list[tuple[str, float]] = [(group[0].text, group[0].confidence)]
        for ev in group[1:]:
            prev_text, prev_conf = merged_parts[-1]
            if text_similarity(prev_text, ev.text) >= _MERGE_DEDUP_SIMILARITY:
                prev_len = len(prev_text.strip())
                curr_len = len(ev.text.strip())
                if curr_len > prev_len or (curr_len == prev_len and ev.confidence > prev_conf):
                    merged_parts[-1] = (ev.text, max(prev_conf, ev.confidence))
                else:
                    merged_parts[-1] = (prev_text, max(prev_conf, ev.confidence))
            else:
                merged_parts.append((ev.text, ev.confidence))
        merged_text = "\n".join(part for part, _ in merged_parts if part)

        merged_event = SubtitleEvent(
            index=group[0].index,
            start_ms=start_ms,
            end_ms=end_ms,
            text=merged_text,
            confidence=best_ev.confidence,
            flagged=flagged,
            thumbnail_png=best_ev.thumbnail_png,
            flag_reason=flag_reason,
        )

        self._push_undo_snapshot()
        self._events[m0 : mk + 1] = [merged_event]
        self._reindex_events()

        self._update_filter_counts()
        self._rebuild_rows(preferred_row=first_visible_row)
        self._refresh_find_status()
        self.events_changed.emit()
        self.events_modified.emit()
        return True

    def _trigger_split(self) -> None:
        if self.isSignalConnected(QMetaMethod.fromSignal(self.split_requested)):
            self.split_requested.emit()
        else:
            self.split_selected_row()

    def split_selected_row(self, playhead_rel_ms: int | None = None) -> bool:
        """Split the single selected cue at ``playhead_rel_ms`` (if strictly inside) or midpoint."""
        pairs = self._selected_visible_and_master_pairs()
        if len(pairs) != 1:
            return False
        visible_row, master_idx = pairs[0]
        ev = self._events[master_idx]
        if (ev.end_ms - ev.start_ms) < 2:
            return False

        if playhead_rel_ms is not None and ev.start_ms < int(playhead_rel_ms) < ev.end_ms:
            split_ms = int(playhead_rel_ms)
        else:
            split_ms = (ev.start_ms + ev.end_ms) // 2

        if "\n" in ev.text:
            lines = ev.text.split("\n")
            mid_line = max(1, len(lines) // 2)
            text_first = "\n".join(lines[:mid_line])
            text_second = "\n".join(lines[mid_line:])
        else:
            text_first = ev.text
            text_second = ev.text

        first_ev = SubtitleEvent(
            index=ev.index,
            start_ms=ev.start_ms,
            end_ms=split_ms,
            text=text_first,
            confidence=ev.confidence,
            flagged=ev.flagged,
            thumbnail_png=ev.thumbnail_png,
            flag_reason=ev.flag_reason,
        )
        second_ev = SubtitleEvent(
            index=ev.index + 1,
            start_ms=split_ms,
            end_ms=ev.end_ms,
            text=text_second,
            confidence=ev.confidence,
            flagged=ev.flagged,
            thumbnail_png=ev.thumbnail_png,
            flag_reason=ev.flag_reason,
        )

        self._push_undo_snapshot()
        self._events[master_idx : master_idx + 1] = [first_ev, second_ev]
        self._reindex_events()

        self._update_filter_counts()
        self._rebuild_rows(preferred_row=visible_row)
        self._refresh_find_status()
        self.events_changed.emit()
        self.events_modified.emit()
        return True

    # === Collapsible Find & Replace ===

    def open_find_bar(self, focus_replace: bool = False) -> None:
        """Show the inline Find & Replace bar and focus the Find or Replace input."""
        self.find_bar.setVisible(True)
        target = self.replace_edit if focus_replace else self.find_edit
        target.setFocus()
        target.selectAll()
        self._refresh_find_status()

    def close_find_bar(self) -> None:
        """Hide the inline Find & Replace bar and return focus to the table."""
        self.find_bar.setVisible(False)
        self.table.setFocus()

    def _visible_row_master_pairs(self) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        for row_idx in range(self.table.rowCount()):
            stt_item = self.table.item(row_idx, _COL_STT)
            if stt_item is None:
                continue
            master_idx = stt_item.data(Qt.ItemDataRole.UserRole)
            if master_idx is not None and 0 <= int(master_idx) < len(self._events):
                pairs.append((row_idx, int(master_idx)))
        return pairs

    def _matching_visible_rows(self, needle: str) -> list[tuple[int, int]]:
        if not needle:
            return []
        query = _decode_display_text(needle).lower()
        return [
            (row_idx, master_idx)
            for row_idx, master_idx in self._visible_row_master_pairs()
            if query in self._events[master_idx].text.lower()
        ]

    def _refresh_find_status(self, *, custom_text: str | None = None) -> None:
        if custom_text is not None:
            self.lbl_find_status.setText(custom_text)
            return
        needle = self.find_edit.text()
        if not needle:
            self.lbl_find_status.setText("")
            return
        matches = self._matching_visible_rows(needle)
        if not matches:
            self.lbl_find_status.setText("0/0")
            return
        cur_row = self._selected_visible_row()
        pos = next((idx for idx, (row, _) in enumerate(matches) if row == cur_row), None)
        if pos is not None:
            self.lbl_find_status.setText(f"{pos + 1}/{len(matches)}")
        else:
            self.lbl_find_status.setText(f"0/{len(matches)}")

    def _select_and_scroll_row(self, row: int) -> None:
        self.table.selectRow(row)
        item = self.table.item(row, _COL_TEXT)
        if item is not None:
            self.table.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtCenter)

    def find_next(self) -> bool:
        """Select the next visible row matching ``find_edit`` (wrapping around)."""
        matches = self._matching_visible_rows(self.find_edit.text())
        if not matches:
            self._refresh_find_status()
            return False
        cur_row = self._selected_visible_row()
        target = next(((row, idx) for idx, (row, _) in enumerate(matches) if row > cur_row), None)
        if target is None:
            target_row, pos = matches[0][0], 0
        else:
            target_row, pos = target
        self._select_and_scroll_row(target_row)
        self.lbl_find_status.setText(f"{pos + 1}/{len(matches)}")
        return True

    def find_prev(self) -> bool:
        """Select the previous visible row matching ``find_edit`` (wrapping around)."""
        matches = self._matching_visible_rows(self.find_edit.text())
        if not matches:
            self._refresh_find_status()
            return False
        cur_row = self._selected_visible_row()
        target = next(
            ((row, idx) for idx, (row, _) in reversed(list(enumerate(matches))) if row < cur_row),
            None,
        )
        if target is None:
            target_row, pos = matches[-1][0], len(matches) - 1
        else:
            target_row, pos = target
        self._select_and_scroll_row(target_row)
        self.lbl_find_status.setText(f"{pos + 1}/{len(matches)}")
        return True

    def replace_current(self) -> bool:
        """Replace matches in the selected row (or jump to next match) in one undoable step."""
        needle = self.find_edit.text()
        if not needle:
            return False
        decoded_needle = _decode_display_text(needle)
        decoded_repl = _decode_display_text(self.replace_edit.text())
        pattern = re.compile(re.escape(decoded_needle), flags=re.IGNORECASE)

        sel = self._selected_row_and_master_idx()
        if sel is None or not pattern.search(self._events[sel[1]].text):
            if not self.find_next():
                return False
            sel = self._selected_row_and_master_idx()
            if sel is None:
                return False

        row, master_idx = sel
        ev = self._events[master_idx]
        new_text, count = pattern.subn(lambda _: decoded_repl, ev.text)
        if count == 0 or new_text == ev.text:
            return False

        self._push_undo_snapshot()
        ev.text = new_text
        if ev.flagged:
            ev.flagged = False
            ev.flag_reason = ""
        text_item = self.table.item(row, _COL_TEXT)
        if text_item is not None:
            self._populating = True
            try:
                text_item.setText(_encode_display_text(ev.text))
            finally:
                self._populating = False
        self._refresh_row_in_place(row, master_idx)
        self._refresh_find_status()
        self.events_changed.emit()
        self.events_modified.emit()
        return True

    def replace_all(
        self,
        needle: str | None = None,
        replacement: str | None = None,
    ) -> int:
        """Replace all matches across visible rows in a single undoable step."""
        raw_needle = self.find_edit.text() if needle is None else needle
        raw_repl = self.replace_edit.text() if replacement is None else replacement
        if not raw_needle:
            return 0

        decoded_needle = _decode_display_text(raw_needle)
        decoded_repl = _decode_display_text(raw_repl)
        pattern = re.compile(re.escape(decoded_needle), flags=re.IGNORECASE)

        pending: list[tuple[int, int, str, int]] = []
        for row_idx, master_idx in self._visible_row_master_pairs():
            ev = self._events[master_idx]
            new_text, n_subs = pattern.subn(lambda _: decoded_repl, ev.text)
            if n_subs > 0 and new_text != ev.text:
                pending.append((row_idx, master_idx, new_text, n_subs))

        if not pending:
            self._refresh_find_status(custom_text="0 thay thế")
            return 0

        self._push_undo_snapshot()
        total_replaced = 0
        self._populating = True
        try:
            for row_idx, master_idx, new_text, n_subs in pending:
                ev = self._events[master_idx]
                ev.text = new_text
                if ev.flagged:
                    ev.flagged = False
                    ev.flag_reason = ""
                text_item = self.table.item(row_idx, _COL_TEXT)
                if text_item is not None:
                    text_item.setText(_encode_display_text(new_text))
                self._apply_row_style(row_idx, ev)
                total_replaced += n_subs
        finally:
            self._populating = False

        self._update_filter_counts()
        self._update_inspector_for_row(self._selected_visible_row())
        self._update_action_buttons_state()
        self._refresh_find_status(custom_text=f"Đã thay {total_replaced}")
        self.events_changed.emit()
        self.events_modified.emit()
        return total_replaced

    # === Table Rendering & Styling ===

    def _update_filter_counts(self) -> None:
        total = len(self._events)
        flagged = sum(1 for ev in self._events if ev.flagged)
        self.radio_all.setText(f"Tất cả ({total})")
        self.radio_flagged.setText(f"Chỉ dòng nghi ngờ ({flagged})")

    def _on_filter_radio_toggled(self, checked_all: bool) -> None:
        new_mode = "all" if checked_all else "flagged_only"
        if new_mode != self._filter_mode:
            self._filter_mode = new_mode
            self._rebuild_rows()
            self._refresh_find_status()

    def _apply_row_style(self, row_idx: int, ev: SubtitleEvent) -> None:
        stt_item = self.table.item(row_idx, _COL_STT)
        if stt_item is not None:
            stt_item.setText(f"{ev.index} !" if ev.flagged else str(ev.index))

        tooltip = (ev.flag_reason.strip() or _FALLBACK_FLAG_REASON) if ev.flagged else ""
        for col_idx in range(5):
            cell = self.table.item(row_idx, col_idx)
            if cell is None:
                continue
            cell.setToolTip(tooltip)
            if ev.flagged:
                cell.setBackground(QColor(43, 33, 16))
                cell.setForeground(QColor(252, 211, 77))
            else:
                cell.setData(Qt.ItemDataRole.BackgroundRole, None)
                if col_idx in (_COL_STT, _COL_CONF):
                    cell.setForeground(QColor(155, 164, 181))
                else:
                    cell.setData(Qt.ItemDataRole.ForegroundRole, None)

    def _refresh_row_in_place(self, row_idx: int, master_idx: int) -> None:
        """Update row visual styling, filter counters, and inspector without rebuilding rows."""
        self._populating = True
        try:
            self._apply_row_style(row_idx, self._events[master_idx])
        finally:
            self._populating = False
        self._update_filter_counts()
        if row_idx == self._selected_visible_row():
            self._update_inspector_for_row(row_idx)
        self._update_action_buttons_state()

    def _rebuild_rows(self, preferred_row: int = 0) -> None:
        visible_pairs = [
            (master_idx, ev)
            for master_idx, ev in enumerate(self._events)
            if self._filter_mode == "all" or ev.flagged
        ]

        self._populating = True
        try:
            self.table.setRowCount(len(visible_pairs))
            for row_idx, (master_idx, ev) in enumerate(visible_pairs):
                stt_item = QTableWidgetItem(f"{ev.index} !" if ev.flagged else str(ev.index))
                stt_item.setData(Qt.ItemDataRole.UserRole, master_idx)
                stt_item.setFlags(stt_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                stt_item.setFont(self._mono_font)

                start_item = QTableWidgetItem(_format_cell_time(ev.start_ms))
                start_item.setFont(self._mono_font)
                end_item = QTableWidgetItem(_format_cell_time(ev.end_ms))
                end_item.setFont(self._mono_font)

                conf_item = QTableWidgetItem(f"{ev.confidence:.2f}")
                conf_item.setFlags(conf_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                conf_item.setFont(self._mono_font)

                text_item = QTableWidgetItem(_encode_display_text(ev.text))
                text_item.setFont(self._cjk_font)

                for col_idx, cell in enumerate(
                    (stt_item, start_item, end_item, conf_item, text_item)
                ):
                    self.table.setItem(row_idx, col_idx, cell)
                self._apply_row_style(row_idx, ev)
        finally:
            self._populating = False

        if visible_pairs:
            target_row = max(0, min(preferred_row, len(visible_pairs) - 1))
            self.table.selectRow(target_row)
            self._render_preview_for_row(target_row)
            self._update_inspector_for_row(target_row)
        else:
            self._raw_pixmap = None
            self.preview_label.clear()
            self.preview_label.setText("Không có dòng phù hợp")
            self._update_inspector_for_row(-1)
        self._update_action_buttons_state()

    def _on_selection_changed(self) -> None:
        row = self._selected_visible_row()
        self._render_preview_for_row(row)
        self._update_inspector_for_row(row)
        self._update_action_buttons_state()

    def _update_inspector_for_row(self, row: int) -> None:
        if row < 0 or row >= self.table.rowCount():
            self.lbl_cue_meta.setText("")
            self.lbl_cue_meta.setVisible(False)
            self._set_meta_warning(False)
            self.flag_banner.setVisible(False)
            self.lbl_flag_reason.setText("")
            self._set_timing_buttons_enabled(False)
            return

        stt_item = self.table.item(row, _COL_STT)
        if stt_item is None:
            self._set_timing_buttons_enabled(False)
            return
        master_idx = stt_item.data(Qt.ItemDataRole.UserRole)
        if master_idx is None or not (0 <= master_idx < len(self._events)):
            self._set_timing_buttons_enabled(False)
            return

        ev = self._events[master_idx]
        dur_ms = max(0, ev.end_ms - ev.start_ms)
        char_count = len(ev.text.replace("\n", ""))
        is_short = dur_ms < _MIN_DURATION_WARN_MS
        is_overlap = (
            (master_idx > 0 and self._events[master_idx - 1].end_ms > ev.start_ms)
            or (
                master_idx + 1 < len(self._events)
                and ev.end_ms > self._events[master_idx + 1].start_ms
            )
        )

        meta_parts = [f"{dur_ms / 1000.0:.2f}s", f"{char_count} ký tự"]
        if is_short:
            meta_parts.append("Quá ngắn")
        if is_overlap:
            meta_parts.append("Trùng giờ")
        self.lbl_cue_meta.setText(" · ".join(meta_parts))
        self.lbl_cue_meta.setVisible(True)
        self._set_meta_warning(is_short or is_overlap)

        if ev.flagged:
            reason = ev.flag_reason.strip() or _FALLBACK_FLAG_REASON
            self.lbl_flag_reason.setText(reason)
            self.lbl_flag_reason.setToolTip(reason)
            self.flag_banner.setVisible(True)
        else:
            self.lbl_flag_reason.setText("")
            self.lbl_flag_reason.setToolTip("")
            self.flag_banner.setVisible(False)

        self._set_timing_buttons_enabled(True)

    def _update_preview_pixmap(self) -> None:
        if self._raw_pixmap is None or self._raw_pixmap.isNull():
            return
        target_size = self.preview_label.size()
        if target_size.width() > 10 and target_size.height() > 10:
            scaled = self._raw_pixmap.scaled(
                target_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        else:
            scaled = self._raw_pixmap
        self.preview_label.setPixmap(scaled)

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Rescale active ROI thumbnail smoothly when the review panel resizes."""
        super().resizeEvent(event)
        self._update_preview_pixmap()

    def _handle_shortcut_key(self, event: QKeyEvent) -> bool:
        key = event.key()
        mods = event.modifiers()
        no_mod = mods == Qt.KeyboardModifier.NoModifier
        alt_only = mods == Qt.KeyboardModifier.AltModifier
        ctrl_only = mods == Qt.KeyboardModifier.ControlModifier
        ctrl_shift = mods == (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
        )

        if key == Qt.Key.Key_Delete and no_mod:
            self.delete_selected_rows()
            return True
        if key == Qt.Key.Key_Escape and no_mod and self.find_bar.isVisible():
            self.close_find_bar()
            return True
        if key == Qt.Key.Key_S and alt_only:
            self.jump_to_selected_start()
            return True
        if key == Qt.Key.Key_E and alt_only:
            self.jump_to_selected_end()
            return True
        if key == Qt.Key.Key_BracketLeft and ctrl_only:
            self.snap_start_requested.emit()
            return True
        if key == Qt.Key.Key_BracketRight and ctrl_only:
            self.snap_end_requested.emit()
            return True
        if key == Qt.Key.Key_D and ctrl_only:
            self.mark_selected_ok()
            return True
        if key == Qt.Key.Key_M and ctrl_only:
            self.merge_selected_rows()
            return True
        if key == Qt.Key.Key_K and ctrl_only:
            self._trigger_split()
            return True
        if key == Qt.Key.Key_Z and ctrl_only:
            self.undo()
            return True
        if (key == Qt.Key.Key_Y and ctrl_only) or (key == Qt.Key.Key_Z and ctrl_shift):
            self.redo()
            return True
        if key == Qt.Key.Key_F and ctrl_only:
            self.open_find_bar(focus_replace=False)
            return True
        if key == Qt.Key.Key_H and ctrl_only:
            self.open_find_bar(focus_replace=True)
            return True
        return False

    def _handle_find_bar_key(self, watched: QObject, event: QKeyEvent) -> bool:
        key = event.key()
        mods = event.modifiers()
        no_mod = mods == Qt.KeyboardModifier.NoModifier
        shift_only = mods == Qt.KeyboardModifier.ShiftModifier
        ctrl_only = mods == Qt.KeyboardModifier.ControlModifier

        if key == Qt.Key.Key_Escape and no_mod:
            self.close_find_bar()
            return True
        if key == Qt.Key.Key_F and ctrl_only:
            self.open_find_bar(focus_replace=False)
            return True
        if key == Qt.Key.Key_H and ctrl_only:
            self.open_find_bar(focus_replace=True)
            return True
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if watched is self.find_edit:
                if shift_only:
                    self.find_prev()
                else:
                    self.find_next()
                return True
            if watched is self.replace_edit:
                self.replace_current()
                return True
        return False

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.preview_label and event.type() == QEvent.Type.Resize:
            self._update_preview_pixmap()
        elif (
            watched in (self.find_edit, self.replace_edit)
            and event.type() == QEvent.Type.KeyPress
            and isinstance(event, QKeyEvent)
        ):
            if self._handle_find_bar_key(watched, event):
                event.accept()
                return True
        elif (
            watched is self.table
            and event.type() == QEvent.Type.KeyPress
            and isinstance(event, QKeyEvent)
            and self.table.state() != QAbstractItemView.State.EditingState
        ):
            if self._handle_shortcut_key(event):
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def _render_preview_for_row(self, row: int) -> None:
        if row < 0 or row >= self.table.rowCount():
            self._raw_pixmap = None
            self.preview_label.clear()
            self.preview_label.setText("Chưa chọn dòng")
            return

        stt_item = self.table.item(row, _COL_STT)
        if stt_item is None:
            return
        master_idx = stt_item.data(Qt.ItemDataRole.UserRole)
        if master_idx is None or not (0 <= master_idx < len(self._events)):
            return

        png_bytes = self._events[master_idx].thumbnail_png
        pixmap = QPixmap()
        if png_bytes and pixmap.loadFromData(png_bytes, "PNG") and not pixmap.isNull():
            self._raw_pixmap = pixmap
            self._update_preview_pixmap()
        else:
            self._raw_pixmap = None
            self.preview_label.clear()
            self.preview_label.setText("Không có ảnh thumbnail")

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._populating:
            return
        row = item.row()
        col = item.column()
        stt_item = self.table.item(row, _COL_STT)
        if stt_item is None:
            return
        master_idx = stt_item.data(Qt.ItemDataRole.UserRole)
        if master_idx is None or not (0 <= master_idx < len(self._events)):
            return

        event = self._events[master_idx]
        if col == _COL_TEXT:
            decoded_text = _decode_display_text(item.text())
            encoded_display = _encode_display_text(decoded_text)
            if item.text() != encoded_display:
                self._populating = True
                try:
                    item.setText(encoded_display)
                finally:
                    self._populating = False
            if decoded_text != event.text:
                self._push_undo_snapshot()
                event.text = decoded_text
                if event.flagged:
                    event.flagged = False
                    event.flag_reason = ""
                    self._refresh_row_in_place(row, master_idx)
                elif row == self._selected_visible_row():
                    self._update_inspector_for_row(row)
                self._refresh_find_status()
                self.events_changed.emit()
                self.events_modified.emit()
            return

        if col in (_COL_START, _COL_END):
            prev_ms = event.start_ms if col == _COL_START else event.end_ms
            parsed_ms = parse_subtitle_time(item.text())
            new_start_ms = parsed_ms if col == _COL_START else event.start_ms
            new_end_ms = parsed_ms if col == _COL_END else event.end_ms
            if parsed_ms is None or new_start_ms >= new_end_ms:
                self._populating = True
                try:
                    item.setText(_format_cell_time(prev_ms))
                finally:
                    self._populating = False
                return

            formatted = _format_cell_time(parsed_ms)
            if parsed_ms == prev_ms:
                if item.text() != formatted:
                    self._populating = True
                    try:
                        item.setText(formatted)
                    finally:
                        self._populating = False
                return

            self._push_undo_snapshot()
            if col == _COL_START:
                event.start_ms = parsed_ms
            else:
                event.end_ms = parsed_ms

            if item.text() != formatted:
                self._populating = True
                try:
                    item.setText(formatted)
                finally:
                    self._populating = False
            if row == self._selected_visible_row():
                self._update_inspector_for_row(row)
            self._update_action_buttons_state()
            self.events_changed.emit()
            self.events_modified.emit()
