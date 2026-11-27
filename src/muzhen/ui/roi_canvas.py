"""Interactive video frame preview and normalized ROI selection widget."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PySide6.QtCore import QMimeData, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QImage,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from muzhen.core.models import RoiRect

VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".mkv", ".avi", ".mov", ".ts", ".webm")

_DEFAULT_ROI = RoiRect(0.10, 0.78, 0.80, 0.16)


def extract_dropped_video_path(mime_data: QMimeData | None) -> Path | None:
    """Return the first local video file path from drag-and-drop mime data, or None."""
    if mime_data is None or not mime_data.hasUrls():
        return None
    for url in mime_data.urls():
        if not url.isLocalFile():
            continue
        candidate = Path(url.toLocalFile())
        if candidate.suffix.lower() in VIDEO_EXTENSIONS:
            return candidate
    return None
_HANDLE_HIT_PX = 8.0
_HANDLE_DRAW_PX = 6.0
_MIN_ROI_SPAN = 0.01

_DragMode = Literal[
    "draw",
    "move",
    "nw",
    "n",
    "ne",
    "e",
    "se",
    "s",
    "sw",
    "w",
]


class RoiCanvas(QWidget):
    """Displays a video frame, optional stroke-mask overlay, and an editable normalized ROI."""

    roi_changed = Signal(object)
    video_dropped = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._roi: RoiRect = _DEFAULT_ROI
        self._frame_pixmap: QPixmap | None = None
        self._mask_pixmap: QPixmap | None = None
        self._mask_is_full_frame: bool = False

        self._drag_mode: _DragMode | None = None
        self._drag_anchor_norm: tuple[float, float] = (0.0, 0.0)
        self._drag_start_roi: RoiRect = self._roi

        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def roi(self) -> RoiRect:
        """Return the current normalized ROI rectangle."""
        return self._roi

    def set_roi(self, roi: RoiRect) -> None:
        """Set and clamp the current normalized ROI rectangle."""
        clamped = roi.clamped()
        if clamped == self._roi:
            return
        self._roi = clamped
        self._mask_pixmap = None
        self.update()
        self.roi_changed.emit(self._roi)

    def set_frame(
        self,
        bgr_frame: np.ndarray | None,
        mask: np.ndarray | None = None,
    ) -> None:
        """Update the displayed BGR video frame and optional binary mask overlay."""
        if bgr_frame is None or bgr_frame.size == 0 or bgr_frame.ndim < 2:
            self._frame_pixmap = None
            self._mask_pixmap = None
            self._mask_is_full_frame = False
            self.update()
            return

        if bgr_frame.ndim == 2:
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        fh, fw = rgb.shape[:2]
        qimg = QImage(rgb.data, fw, fh, fw * 3, QImage.Format.Format_RGB888).copy()
        self._frame_pixmap = QPixmap.fromImage(qimg)

        self._mask_pixmap = None
        self._mask_is_full_frame = False
        if mask is not None and mask.size > 0 and mask.ndim >= 2:
            mh, mw = mask.shape[:2]
            rgba = np.zeros((mh, mw, 4), dtype=np.uint8)
            active = mask > 0
            rgba[active] = (0, 230, 140, 155)
            mask_qimg = QImage(rgba.data, mw, mh, mw * 4, QImage.Format.Format_RGBA8888).copy()
            self._mask_pixmap = QPixmap.fromImage(mask_qimg)
            self._mask_is_full_frame = (mh == fh and mw == fw)

        self.update()

    def _display_rect(self) -> QRectF:
        w = max(1, self.width())
        h = max(1, self.height())
        if self._frame_pixmap is None or self._frame_pixmap.isNull():
            return QRectF(0.0, 0.0, float(w), float(h))

        fw = max(1, self._frame_pixmap.width())
        fh = max(1, self._frame_pixmap.height())
        scale = min(w / float(fw), h / float(fh))
        draw_w = fw * scale
        draw_h = fh * scale
        left = (w - draw_w) * 0.5
        top = (h - draw_h) * 0.5
        return QRectF(left, top, draw_w, draw_h)

    @staticmethod
    def _roi_to_rect(roi: RoiRect, disp: QRectF) -> QRectF:
        return QRectF(
            disp.left() + roi.x * disp.width(),
            disp.top() + roi.y * disp.height(),
            roi.w * disp.width(),
            roi.h * disp.height(),
        )

    @staticmethod
    def _pos_to_norm(pos: QPointF, disp: QRectF) -> tuple[float, float]:
        nx = (pos.x() - disp.left()) / max(1.0, disp.width())
        ny = (pos.y() - disp.top()) / max(1.0, disp.height())
        return (max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny)))

    @staticmethod
    def _handle_points(rect: QRectF) -> dict[_DragMode, QPointF]:
        cx = (rect.left() + rect.right()) * 0.5
        cy = (rect.top() + rect.bottom()) * 0.5
        return {
            "nw": rect.topLeft(),
            "n": QPointF(cx, rect.top()),
            "ne": rect.topRight(),
            "e": QPointF(rect.right(), cy),
            "se": rect.bottomRight(),
            "s": QPointF(cx, rect.bottom()),
            "sw": rect.bottomLeft(),
            "w": QPointF(rect.left(), cy),
        }

    def _hit_test(self, pos: QPointF, disp: QRectF) -> _DragMode:
        roi_rect = self._roi_to_rect(self._roi, disp)
        for mode, pt in self._handle_points(roi_rect).items():
            if (
                abs(pos.x() - pt.x()) <= _HANDLE_HIT_PX
                and abs(pos.y() - pt.y()) <= _HANDLE_HIT_PX
            ):
                return mode

        if roi_rect.contains(pos):
            return "move"
        return "draw"

    @staticmethod
    def _cursor_for_mode(mode: _DragMode) -> Qt.CursorShape:
        if mode in ("nw", "se"):
            return Qt.CursorShape.SizeFDiagCursor
        if mode in ("ne", "sw"):
            return Qt.CursorShape.SizeBDiagCursor
        if mode in ("n", "s"):
            return Qt.CursorShape.SizeVerCursor
        if mode in ("e", "w"):
            return Qt.CursorShape.SizeHorCursor
        if mode == "move":
            return Qt.CursorShape.SizeAllCursor
        return Qt.CursorShape.CrossCursor

    def _apply_roi_update(self, new_roi: RoiRect) -> None:
        clamped = new_roi.clamped()
        if clamped != self._roi:
            self._roi = clamped
            self._mask_pixmap = None
            self.update()
            self.roi_changed.emit(self._roi)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        disp = self._display_rect()
        pos = event.position()
        self._drag_mode = self._hit_test(pos, disp)
        self._drag_anchor_norm = self._pos_to_norm(pos, disp)
        self._drag_start_roi = self._roi
        self.setCursor(self._cursor_for_mode(self._drag_mode))
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        disp = self._display_rect()
        pos = event.position()

        if self._drag_mode is None:
            hover_mode = self._hit_test(pos, disp)
            self.setCursor(self._cursor_for_mode(hover_mode))
            super().mouseMoveEvent(event)
            return

        nx, ny = self._pos_to_norm(pos, disp)
        ax, ay = self._drag_anchor_norm
        start = self._drag_start_roi

        if self._drag_mode == "draw":
            x1, x2 = min(ax, nx), max(ax, nx)
            y1, y2 = min(ay, ny), max(ay, ny)
            if (x2 - x1) >= _MIN_ROI_SPAN and (y2 - y1) >= _MIN_ROI_SPAN:
                self._apply_roi_update(RoiRect(x=x1, y=y1, w=x2 - x1, h=y2 - y1))
        elif self._drag_mode == "move":
            dx = nx - ax
            dy = ny - ay
            new_x = max(0.0, min(1.0 - start.w, start.x + dx))
            new_y = max(0.0, min(1.0 - start.h, start.y + dy))
            self._apply_roi_update(RoiRect(x=new_x, y=new_y, w=start.w, h=start.h))
        else:
            left = start.x
            top = start.y
            right = start.x + start.w
            bottom = start.y + start.h

            if "w" in self._drag_mode:
                left = min(nx, right - _MIN_ROI_SPAN)
            if "e" in self._drag_mode:
                right = max(nx, left + _MIN_ROI_SPAN)
            if "n" in self._drag_mode:
                top = min(ny, bottom - _MIN_ROI_SPAN)
            if "s" in self._drag_mode:
                bottom = max(ny, top + _MIN_ROI_SPAN)

            self._apply_roi_update(
                RoiRect(
                    x=left,
                    y=top,
                    w=max(_MIN_ROI_SPAN, right - left),
                    h=max(_MIN_ROI_SPAN, bottom - top),
                )
            )
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._drag_mode is not None:
            if self._drag_mode == "draw":
                disp = self._display_rect()
                nx, ny = self._pos_to_norm(event.position(), disp)
                ax, ay = self._drag_anchor_norm
                x1, x2 = min(ax, nx), max(ax, nx)
                y1, y2 = min(ay, ny), max(ay, ny)
                if (x2 - x1) >= _MIN_ROI_SPAN and (y2 - y1) >= _MIN_ROI_SPAN:
                    self._apply_roi_update(RoiRect(x=x1, y=y1, w=x2 - x1, h=y2 - y1))
                else:
                    self._apply_roi_update(self._drag_start_roi)
            self._drag_mode = None
            hover_mode = self._hit_test(event.position(), self._display_rect())
            self.setCursor(self._cursor_for_mode(hover_mode))
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self.isEnabled() and extract_dropped_video_path(event.mimeData()) is not None:
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        if not self.isEnabled():
            event.ignore()
            return
        video_path = extract_dropped_video_path(event.mimeData())
        if video_path is not None:
            event.acceptProposedAction()
            self.video_dropped.emit(str(video_path))
            return
        event.ignore()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.fillRect(self.rect(), QColor("#0B0B0B"))

            disp = self._display_rect()
            has_frame = self._frame_pixmap is not None and not self._frame_pixmap.isNull()
            if has_frame:
                painter.drawPixmap(disp, self._frame_pixmap, QRectF(self._frame_pixmap.rect()))
            else:
                painter.setPen(QColor("#6E6E6E"))
                painter.drawText(
                    self.rect(),
                    Qt.AlignmentFlag.AlignCenter,
                    "Drop a video file here or click Open Video",
                )

            roi_rect = self._roi_to_rect(self._roi, disp)

            if has_frame:
                scrim = QPainterPath()
                scrim.addRect(disp)
                hole = QPainterPath()
                hole.addRect(roi_rect)
                painter.fillPath(scrim.subtracted(hole), QColor(0, 0, 0, 84))

            if self._mask_pixmap is not None and not self._mask_pixmap.isNull():
                target_rect = disp if self._mask_is_full_frame else roi_rect
                painter.drawPixmap(
                    target_rect, self._mask_pixmap, QRectF(self._mask_pixmap.rect())
                )

            painter.fillRect(roi_rect, QColor(16, 185, 129, 22))
            pen = QPen(QColor("#10B981"), 1.6)
            painter.setPen(pen)
            painter.drawRect(roi_rect)

            half = _HANDLE_DRAW_PX * 0.5
            painter.setBrush(QColor("#10B981"))
            painter.setPen(QPen(QColor("#0B0B0B"), 1.0))
            for pt in self._handle_points(roi_rect).values():
                painter.drawRect(
                    QRectF(pt.x() - half, pt.y() - half, _HANDLE_DRAW_PX, _HANDLE_DRAW_PX)
                )

            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#262626"), 1.0))
            painter.drawRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5))
        finally:
            painter.end()
