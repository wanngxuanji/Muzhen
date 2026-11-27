"""Embedded Lucide-style SVG vector icons rendered cleanly for HiDPI displays."""

from __future__ import annotations

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

_SVG_PATHS: dict[str, str] = {
    "film": (
        '<rect x="2" y="4" width="20" height="16" rx="2.5"/>'
        '<line x1="7" y1="4" x2="7" y2="20"/>'
        '<line x1="17" y1="4" x2="17" y2="20"/>'
        '<line x1="2" y1="9" x2="7" y2="9"/>'
        '<line x1="2" y1="15" x2="7" y2="15"/>'
        '<line x1="17" y1="9" x2="22" y2="9"/>'
        '<line x1="17" y1="15" x2="22" y2="15"/>'
    ),
    "sidebar": (
        '<rect x="3" y="3" width="18" height="18" rx="2"/>'
        '<line x1="9" y1="3" x2="9" y2="21"/>'
    ),
    "folder": (
        '<path d="M3 6a2 2 0 0 1 2-2h4.2a2 2 0 0 1 1.6.8l1.2 1.6a1 1 0 0 0 .8.4H19a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'
    ),
    "save": (
        '<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/>'
        '<polyline points="17 21 17 13 7 13 7 21"/>'
        '<polyline points="7 3 7 8 15 8"/>'
    ),
    "trash": (
        '<polyline points="3 6 5 6 21 6"/>'
        '<path d="M8 6V4a1.5 1.5 0 0 1 1.5-1.5h5A1.5 1.5 0 0 1 16 4v2"/>'
        '<path d="M19 6l-1 13a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>'
        '<line x1="10" y1="11" x2="10" y2="17"/>'
        '<line x1="14" y1="11" x2="14" y2="17"/>'
    ),
    "skip_back": (
        '<line x1="5" y1="5" x2="5" y2="19"/>'
        '<polygon points="19 5 9 12 19 19 19 5"/>'
    ),
    "step_back": '<polyline points="15 18 9 12 15 6"/>',
    "step_fwd": '<polyline points="9 18 15 12 9 6"/>',
    "skip_fwd": (
        '<polygon points="5 5 15 12 5 19 5 5"/>'
        '<line x1="19" y1="5" x2="19" y2="19"/>'
    ),
    "scan": (
        '<path d="M3 7V5a2 2 0 0 1 2-2h2"/>'
        '<path d="M17 3h2a2 2 0 0 1 2 2v2"/>'
        '<path d="M21 17v2a2 2 0 0 1-2 2h-2"/>'
        '<path d="M7 21H5a2 2 0 0 1-2-2v-2"/>'
        '<line x1="7" y1="12" x2="17" y2="12"/>'
        '<line x1="12" y1="7" x2="12" y2="17"/>'
    ),
    "play": '<polygon points="6 4 20 12 6 20 6 4"/>',
    "pause": (
        '<rect x="6" y="4" width="4" height="16" rx="1"/>'
        '<rect x="14" y="4" width="4" height="16" rx="1"/>'
    ),
    "stop": '<rect x="5" y="5" width="14" height="14" rx="2"/>',
    "export": (
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        '<polyline points="7 10 12 15 17 10"/>'
        '<line x1="12" y1="15" x2="12" y2="3"/>'
    ),
    "trim_in": (
        '<path d="M9 4H5v16h4"/>'
        '<polyline points="14 8 10 12 14 16"/>'
        '<line x1="10" y1="12" x2="20" y2="12"/>'
    ),
    "trim_out": (
        '<path d="M15 4h4v16h-4"/>'
        '<polyline points="10 8 14 12 10 16"/>'
        '<line x1="4" y1="12" x2="14" y2="12"/>'
    ),
    "x_mark": (
        '<line x1="18" y1="6" x2="6" y2="18"/>'
        '<line x1="6" y1="6" x2="18" y2="18"/>'
    ),
    "check": '<polyline points="20 6 9 17 4 12"/>',
    "folder_open": (
        '<path d="m6 14 1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.54 6a2 2 0 0 1-1.95 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2 0 0 1 2 2v2"/>'
    ),
    "undo": (
        '<path d="M9 14 4 9l5-5"/>'
        '<path d="M4 9h10.5a5.5 5.5 0 0 1 5.5 5.5a5.5 5.5 0 0 1-5.5 5.5H11"/>'
    ),
    "redo": (
        '<path d="m15 14 5-5-5-5"/>'
        '<path d="M20 9H9.5A5.5 5.5 0 0 0 4 14.5A5.5 5.5 0 0 0 9.5 20H13"/>'
    ),
    "search": (
        '<circle cx="11" cy="11" r="7"/>'
        '<line x1="21" y1="21" x2="16.65" y2="16.65"/>'
    ),
    "merge": (
        '<path d="m8 6 4-4 4 4"/>'
        '<path d="M12 2v10.3a4 4 0 0 1-1.172 2.872L4 22"/>'
        '<path d="m20 22-5-5"/>'
    ),
    "split": (
        '<path d="M16 3h5v5"/>'
        '<path d="M8 3H3v5"/>'
        '<path d="M12 22v-8.3a4 4 0 0 0-1.172-2.872L3 3"/>'
        '<path d="m15 9 6-6"/>'
    ),
}

ICON_NAMES: tuple[str, ...] = tuple(_SVG_PATHS.keys())


def _render_svg_pixmap(svg_body: str, color: str, size: int) -> QPixmap:
    svg_xml = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'fill="none" stroke="{color}" stroke-width="1.5" '
        'stroke-linecap="round" stroke-linejoin="round">'
        f"{svg_body}</svg>"
    )
    scale = 2
    px = max(8, int(size)) * scale
    pixmap = QPixmap(px, px)
    pixmap.fill(Qt.GlobalColor.transparent)
    renderer = QSvgRenderer(QByteArray(svg_xml.encode("utf-8")))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    renderer.render(painter)
    painter.end()
    pixmap.setDevicePixelRatio(float(scale))
    return pixmap


def make_icon(
    name: str,
    *,
    color: str = "#B8B8B8",
    disabled_color: str = "#5A5A5A",
    size: int = 16,
) -> QIcon:
    """Create a HiDPI QIcon from the built-in vector icon set."""
    body = _SVG_PATHS.get(name)
    if body is None:
        raise ValueError(f"Unknown icon name: {name!r}")
    icon = QIcon()
    icon.addPixmap(_render_svg_pixmap(body, color, size), QIcon.Mode.Normal, QIcon.State.Off)
    icon.addPixmap(
        _render_svg_pixmap(body, disabled_color, size), QIcon.Mode.Disabled, QIcon.State.Off
    )
    return icon
