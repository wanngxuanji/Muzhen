"""In-memory subtitle review table with flagged filtering and ROI crop preview."""

from __future__ import annotations

import re

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QColor, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QRadioButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from transapp.core.models import SubtitleEvent
from transapp.exporters.ass_exporter import format_ass_time
from transapp.exporters.srt_exporter import format_srt_time

_COL_STT = 0
_COL_START = 1
_COL_END = 2
_COL_CONF = 3
_COL_TEXT = 4

_TIME_RE = re.compile(r"^\s*(\d+):([0-5]\d):([0-5]\d)(?:[.,](\d{1,3}))?\s*$")


def parse_subtitle_time(value: str) -> int | None:
    """Parse ASS ``H:MM:SS.cc`` or SRT ``HH:MM:SS,mmm`` into non-negative milliseconds."""
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
    clamped = max(0, int(ms))
    return format_srt_time(clamped) if (clamped % 10) != 0 else format_ass_time(clamped)


class ReviewTable(QWidget):
    """Editable subtitle review table with flagged filter and in-memory thumbnail preview."""

    events_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._events: list[SubtitleEvent] = []
        self._filter_mode: str = "all"
        self._populating: bool = False
        self._raw_pixmap: QPixmap | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        title = QLabel("3. Rà soát & sửa nhanh")
        title.setStyleSheet("font-weight: 600;")
        header.addWidget(title)
        header.addSpacing(16)
        header.addWidget(QLabel("Lọc:"))

        self.radio_all = QRadioButton("Tất cả (0)")
        self.radio_all.setChecked(True)
        self.radio_flagged = QRadioButton("Chỉ dòng nghi ngờ (0)")
        header.addWidget(self.radio_all)
        header.addWidget(self.radio_flagged)
        header.addStretch(1)
        root.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["STT", "Start", "End", "Conf", "Text"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.AnyKeyPressed
        )
        self.table.verticalHeader().setVisible(False)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(_COL_STT, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(_COL_START, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(_COL_END, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(_COL_CONF, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(_COL_TEXT, QHeaderView.ResizeMode.Stretch)

        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(6, 0, 0, 0)
        preview_layout.setSpacing(4)
        preview_title = QLabel("Ảnh crop ROI dòng đang chọn")
        preview_title.setStyleSheet("font-weight: 600;")
        preview_layout.addWidget(preview_title)

        self.preview_label = QLabel("Chưa chọn dòng")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(280, 120)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.preview_label.setStyleSheet(
            "background-color: #14161A; color: #9AA3B2; border: 1px solid #2E3440;"
        )
        self.preview_label.installEventFilter(self)
        preview_layout.addWidget(self.preview_label, 1)

        splitter.addWidget(self.table)
        splitter.addWidget(preview_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        self.radio_all.toggled.connect(self._on_filter_radio_toggled)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        self.table.itemChanged.connect(self._on_item_changed)

    def filter_mode(self) -> str:
        """Return active filter mode (``"all"`` or ``"flagged_only"``)."""
        return self._filter_mode

    def set_events(self, events: list[SubtitleEvent]) -> None:
        """Replace in-memory subtitle events and refresh the table."""
        self._events = list(events)
        self._update_filter_counts()
        self._rebuild_rows()

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

    def clear(self) -> None:
        """Clear all events, table rows, and thumbnail pixmap from memory."""
        self._events = []
        self._raw_pixmap = None
        self._populating = True
        try:
            self.table.setRowCount(0)
        finally:
            self._populating = False
        self.preview_label.clear()
        self.preview_label.setText("Chưa chọn dòng")
        self._update_filter_counts()

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

    def _rebuild_rows(self) -> None:
        visible_pairs = [
            (master_idx, ev)
            for master_idx, ev in enumerate(self._events)
            if self._filter_mode == "all" or ev.flagged
        ]

        self._populating = True
        try:
            self.table.setRowCount(len(visible_pairs))
            for row_idx, (master_idx, ev) in enumerate(visible_pairs):
                stt_text = f"{ev.index} !" if ev.flagged else str(ev.index)
                stt_item = QTableWidgetItem(stt_text)
                stt_item.setData(Qt.ItemDataRole.UserRole, master_idx)
                stt_item.setFlags(stt_item.flags() & ~Qt.ItemFlag.ItemIsEditable)

                start_item = QTableWidgetItem(_format_cell_time(ev.start_ms))
                end_item = QTableWidgetItem(_format_cell_time(ev.end_ms))

                conf_item = QTableWidgetItem(f"{ev.confidence:.2f}")
                conf_item.setFlags(conf_item.flags() & ~Qt.ItemFlag.ItemIsEditable)

                text_item = QTableWidgetItem(ev.text)

                row_items = (stt_item, start_item, end_item, conf_item, text_item)
                for col_idx, cell in enumerate(row_items):
                    if ev.flagged:
                        cell.setBackground(QColor(74, 56, 12))
                        cell.setForeground(QColor(255, 229, 153))
                    self.table.setItem(row_idx, col_idx, cell)
        finally:
            self._populating = False

        if visible_pairs:
            self.table.selectRow(0)
            self._render_preview_for_row(0)
        else:
            self._raw_pixmap = None
            self.preview_label.clear()
            self.preview_label.setText("Không có dòng phù hợp")

    def _on_selection_changed(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            selected = self.table.selectionModel().selectedRows()
            row = selected[0].row() if selected else -1
        self._render_preview_for_row(row)

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

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self.preview_label and event.type() == QEvent.Type.Resize:
            self._update_preview_pixmap()
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
            normalized_text = item.text().replace("\r\n", "\n").replace("\r", "\n")
            if normalized_text != event.text:
                event.text = normalized_text
                self.events_changed.emit()
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

            if col == _COL_START:
                event.start_ms = parsed_ms
            else:
                event.end_ms = parsed_ms

            formatted = _format_cell_time(parsed_ms)
            if item.text() != formatted:
                self._populating = True
                try:
                    item.setText(formatted)
                finally:
                    self._populating = False
            self.events_changed.emit()
