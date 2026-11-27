"""Adobe Premiere Pro / After Effects neutral graphite dark theme for Muzhen."""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

from muzhen.core import paths as runtime_paths

PALETTE_TOKENS: dict[str, str] = {
    "bg_gutter": "#111111",
    "bg_root": "#161616",
    "bg_panel": "#1D1D1D",
    "bg_surface": "#1D1D1D",
    "bg_header": "#232323",
    "bg_elevated": "#262626",
    "bg_control": "#2A2A2A",
    "bg_control_hover": "#333333",
    "bg_control_pressed": "#1F1F1F",
    "bg_input": "#141414",
    "border_hairline": "#2E2E2E",
    "border_subtle": "#2E2E2E",
    "border_strong": "#424242",
    "text_primary": "#E6E6E6",
    "text_secondary": "#A0A0A0",
    "text_muted": "#6E6E6E",
    "accent": "#2D8CEB",
    "accent_hover": "#439BF0",
    "accent_pressed": "#1F75CC",
    "timecode_blue": "#3EA6FF",
    "roi_emerald": "#10B981",
    "warn_bg": "#2E2514",
    "warn_text": "#E5B54D",
    "danger_bg": "#2D1919",
    "danger_border": "#7A2E2E",
    "danger_text": "#F08080",
}

MONO_FONT_FAMILY = "Cascadia Mono, Consolas, Courier New, monospace"

_STUDIO_DARK_QSS_TEMPLATE = """
QMainWindow, QDialog {
    background-color: #161616;
    color: #E6E6E6;
}

QWidget {
    color: #E6E6E6;
    font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
    font-size: 12px;
}

QToolTip {
    background-color: #262626;
    color: #E6E6E6;
    border: 1px solid #424242;
    border-radius: 3px;
    padding: 4px 8px;
}

QFrame#topBar {
    background-color: #181818;
    border: none;
    border-bottom: 1px solid #2E2E2E;
    border-radius: 0px;
}

QLabel#brandTitle {
    font-size: 13px;
    font-weight: 600;
    color: #E6E6E6;
}

QLabel#versionBadge {
    background-color: #262626;
    color: #A0A0A0;
    border: 1px solid #363636;
    border-radius: 3px;
    padding: 1px 5px;
    font-size: 10px;
    font-weight: 500;
}

QLabel#metaBadge {
    background-color: #141414;
    color: #A0A0A0;
    border: 1px solid #2E2E2E;
    border-radius: 3px;
    padding: 2px 7px;
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 11px;
}

QFrame[panel="true"], QFrame[card="true"] {
    background-color: #1D1D1D;
    border: 1px solid #2E2E2E;
    border-radius: 2px;
}

QFrame[panel="true"][activePanel="true"] {
    border: 1px solid #2D8CEB;
}

QFrame#panelHeader {
    background-color: #232323;
    border: none;
    border-bottom: 1px solid #2E2E2E;
    border-top-left-radius: 1px;
    border-top-right-radius: 1px;
    min-height: 28px;
    max-height: 28px;
}

QLabel#panelTitle {
    color: #A0A0A0;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 8px 3px 8px;
    border-bottom: 2px solid transparent;
    background: transparent;
}

QFrame[panel="true"][activePanel="true"] QLabel#panelTitle {
    color: #E6E6E6;
    border-bottom: 2px solid #2D8CEB;
}

QFrame[divider="true"] {
    background-color: #2E2E2E;
    max-height: 1px;
    min-height: 1px;
    border: none;
}

QLabel[role="sectionTitle"] {
    color: #A0A0A0;
    font-size: 11px;
    font-weight: 600;
}

QLabel[role="fieldLabel"] {
    color: #A0A0A0;
    font-size: 12px;
}

QLabel[role="mono"] {
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 11px;
    color: #C4C4C4;
}

QLabel#timecodeBadge {
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 11.5px;
    font-weight: 600;
    color: #3EA6FF;
    background-color: #141414;
    border: 1px solid #2E2E2E;
    border-radius: 3px;
    padding: 2px 8px;
}

QLabel#roiPill {
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 11px;
    color: #A0A0A0;
    background-color: #141414;
    border: 1px solid #2E2E2E;
    border-radius: 3px;
    padding: 2px 7px;
}

QLabel#trimRangeBadge {
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 11px;
    color: #A0A0A0;
    background-color: #141414;
    border: 1px solid #2E2E2E;
    border-radius: 3px;
    padding: 2px 7px;
}

QLabel#trimRangeBadge[active="true"] {
    color: #10B981;
    background-color: #14221D;
    border: 1px solid #1E4D3B;
}

QLabel#cueMetaBadge {
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 10px;
    color: #A0A0A0;
    background-color: #141414;
    border: 1px solid #2E2E2E;
    border-radius: 3px;
    padding: 1px 6px;
}

QLabel#cueMetaBadge[warning="true"] {
    color: #E5B54D;
    background-color: #2E2514;
    border: 1px solid #6E5320;
}

QFrame#flagBanner {
    background-color: #2E2514;
    border: 1px solid #6E5320;
    border-radius: 3px;
}

QLabel#flagReasonLabel {
    color: #E5B54D;
    font-size: 11px;
    font-weight: 500;
}

QFrame#findReplaceBar {
    background-color: #161616;
    border: 1px solid #2E2E2E;
    border-radius: 3px;
}

QFrame#findReplaceBar QLineEdit {
    background-color: #141414;
    padding: 2px 7px;
    min-height: 16px;
    border-radius: 3px;
    font-size: 11px;
}

QLabel#findStatusLabel {
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 10px;
    color: #A0A0A0;
    padding: 0 4px;
}

QLabel#testResultLabel {
    color: #D4D4D4;
    font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
    font-size: 12px;
    padding: 0 4px;
}

QLabel#statusLabel {
    color: #A0A0A0;
    font-size: 11px;
}

QScrollArea#inspectorScroll {
    background: transparent;
    border: none;
}

QScrollArea#inspectorScroll > QWidget > QWidget {
    background: transparent;
}

QLineEdit, QComboBox, QDoubleSpinBox {
    background-color: #141414;
    color: #E6E6E6;
    border: 1px solid #363636;
    border-radius: 3px;
    padding: 3px 8px;
    min-height: 18px;
    selection-background-color: #2D8CEB;
}

QLineEdit:hover, QComboBox:hover, QDoubleSpinBox:hover {
    border-color: #4A4A4A;
}

QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus {
    border-color: #2D8CEB;
}

QLineEdit:disabled, QComboBox:disabled, QDoubleSpinBox:disabled {
    background-color: #181818;
    color: #6E6E6E;
    border-color: #262626;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 22px;
    border-left: 1px solid #2E2E2E;
    border-top-right-radius: 2px;
    border-bottom-right-radius: 2px;
    background-color: #232323;
}

QComboBox::down-arrow {
    image: url("__CHEVRON_DOWN__");
    width: 12px;
    height: 12px;
}

QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    width: 20px;
    border-left: 1px solid #2E2E2E;
    background-color: #232323;
}

QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #333333;
}

QDoubleSpinBox::up-arrow {
    image: url("__CHEVRON_UP__");
    width: 10px;
    height: 10px;
}

QDoubleSpinBox::down-arrow {
    image: url("__CHEVRON_DOWN__");
    width: 10px;
    height: 10px;
}

QComboBox QAbstractItemView {
    background-color: #232323;
    color: #E6E6E6;
    border: 1px solid #424242;
    selection-background-color: #2D8CEB;
    selection-color: #FFFFFF;
    outline: none;
    padding: 2px;
}

QPushButton {
    background-color: #2A2A2A;
    color: #E6E6E6;
    border: 1px solid #3A3A3A;
    border-radius: 3px;
    padding: 4px 10px;
    font-weight: 500;
    min-height: 18px;
}

QPushButton:hover {
    background-color: #333333;
    border-color: #4A4A4A;
}

QPushButton:pressed {
    background-color: #1F1F1F;
}

QPushButton:disabled {
    background-color: #1F1F1F;
    color: #6E6E6E;
    border-color: #2A2A2A;
}

QPushButton[variant="primary"] {
    background-color: #2D8CEB;
    color: #FFFFFF;
    border: 1px solid #2D8CEB;
    font-weight: 600;
    padding: 5px 12px;
}

QPushButton[variant="primary"]:hover {
    background-color: #439BF0;
    border-color: #439BF0;
}

QPushButton[variant="primary"]:pressed {
    background-color: #1F75CC;
    border-color: #1F75CC;
}

QPushButton[variant="primary"]:disabled {
    background-color: #233547;
    color: #6E7E8F;
    border-color: #2A3E52;
}

QPushButton[variant="danger"] {
    background-color: #2D1919;
    color: #F08080;
    border: 1px solid #7A2E2E;
    font-weight: 500;
}

QPushButton[variant="danger"]:hover {
    background-color: #3D2121;
    border-color: #9E3A3A;
    color: #F8B4B4;
}

QPushButton[variant="danger"]:pressed {
    background-color: #221313;
}

QPushButton[variant="danger"]:disabled {
    background-color: #1F1F1F;
    color: #6E6E6E;
    border-color: #2A2A2A;
}

QPushButton[variant="tool"] {
    background-color: #262626;
    color: #D4D4D4;
    border: 1px solid #333333;
    border-radius: 3px;
    padding: 3px 6px;
    min-height: 18px;
}

QPushButton[variant="tool"]:hover {
    background-color: #333333;
    border-color: #424242;
}

QPushButton[variant="tool"]:pressed {
    background-color: #1F1F1F;
}

QPushButton[variant="tool"]:disabled {
    background-color: #1A1A1A;
    color: #6E6E6E;
    border-color: #262626;
}

QPushButton[variant="trim"] {
    background-color: #262626;
    color: #E6E6E6;
    border: 1px solid #333333;
    border-radius: 3px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 500;
    min-height: 18px;
}

QPushButton[variant="trim"]:hover {
    background-color: #333333;
    border-color: #424242;
}

QPushButton[variant="trim"]:pressed {
    background-color: #1F1F1F;
}

QPushButton[variant="trim"]:disabled {
    background-color: #1A1A1A;
    color: #6E6E6E;
    border-color: #262626;
}

QPushButton[variant="timing"] {
    background-color: #262626;
    color: #E6E6E6;
    border: 1px solid #333333;
    border-radius: 3px;
    padding: 2px 6px;
    font-size: 11px;
    font-weight: 500;
    min-height: 18px;
}

QPushButton[variant="timing"]:hover {
    background-color: #333333;
    border-color: #424242;
}

QPushButton[variant="timing"]:pressed {
    background-color: #1F1F1F;
}

QPushButton[variant="timing"]:disabled {
    background-color: #1A1A1A;
    color: #6E6E6E;
    border-color: #262626;
}

QPushButton[variant="toggle"] {
    background-color: #262626;
    color: #D4D4D4;
    border: 1px solid #333333;
    border-radius: 3px;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: 500;
    min-height: 18px;
}

QPushButton[variant="toggle"]:hover {
    background-color: #333333;
    border-color: #424242;
}

QPushButton[variant="toggle"]:pressed {
    background-color: #1F1F1F;
}

QPushButton[variant="toggle"][checked="true"],
QPushButton[variant="toggle"]:checked {
    background-color: #2A3A4D;
    color: #E6E6E6;
    border: 1px solid #2D8CEB;
}

QPushButton[variant="toggle"][checked="true"]:hover,
QPushButton[variant="toggle"]:checked:hover {
    background-color: #32465E;
    border-color: #439BF0;
}

QPushButton[variant="ok"] {
    background-color: #14261E;
    color: #10B981;
    border: 1px solid #1E4D3B;
    border-radius: 3px;
    padding: 1px 7px;
    font-size: 11px;
    font-weight: 600;
    min-height: 16px;
}

QPushButton[variant="ok"]:hover {
    background-color: #193328;
    border-color: #10B981;
    color: #6EE7B7;
}

QPushButton[variant="ok"]:disabled {
    background-color: #1A1A1A;
    color: #6E6E6E;
    border-color: #262626;
}

QPushButton[variant="update"] {
    background-color: rgba(45, 140, 235, 0.18);
    color: #3EA6FF;
    border: 1px solid #2D8CEB;
    border-radius: 3px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 600;
    min-height: 18px;
}

QPushButton[variant="update"]:hover {
    background-color: #2D8CEB;
    color: #FFFFFF;
    border-color: #439BF0;
}

QPushButton[variant="update"]:pressed {
    background-color: #1F75CC;
    color: #FFFFFF;
    border-color: #1F75CC;
}

QPushButton[variant="update"]:disabled {
    background-color: #1A1A1A;
    color: #6E6E6E;
    border-color: #262626;
}

QLabel#updateHeaderTitle {
    font-size: 14px;
    font-weight: 600;
    color: #E6E6E6;
}

QLabel#updateVersionPill {
    background-color: rgba(45, 140, 235, 0.18);
    color: #3EA6FF;
    border: 1px solid #2D8CEB;
    border-radius: 3px;
    padding: 2px 8px;
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 11px;
    font-weight: 600;
}

QLabel#updatePublishedLabel {
    color: #A0A0A0;
    font-size: 11px;
}

QPlainTextEdit#updateChangelog {
    background-color: #141414;
    border: 1px solid #2E2E2E;
    border-radius: 3px;
    color: #E6E6E6;
    font-family: "Cascadia Mono", "Consolas", "Segoe UI", monospace;
    font-size: 11.5px;
    padding: 8px;
    selection-background-color: #2D8CEB;
    selection-color: #FFFFFF;
}

QCheckBox, QRadioButton {
    spacing: 6px;
    color: #E6E6E6;
}

QCheckBox:disabled, QRadioButton:disabled {
    color: #6E6E6E;
}

QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border-radius: 2px;
    border: 1px solid #555555;
    background-color: #141414;
}

QCheckBox::indicator:hover {
    border-color: #2D8CEB;
}

QCheckBox::indicator:checked {
    background-color: #2D8CEB;
    border-color: #2D8CEB;
    image: url("__CHECK__");
}

QRadioButton::indicator {
    width: 14px;
    height: 14px;
    border-radius: 7px;
    border: 1px solid #555555;
    background-color: #141414;
}

QRadioButton::indicator:hover {
    border-color: #2D8CEB;
}

QRadioButton::indicator:checked {
    border: 1.5px solid #2D8CEB;
    background-color: qradialgradient(
        cx:0.5, cy:0.5, radius:0.5,
        fx:0.5, fy:0.5,
        stop:0 #FFFFFF,
        stop:0.42 #FFFFFF,
        stop:0.52 #2D8CEB,
        stop:1 #2D8CEB
    );
}

QFrame#filterPill {
    background-color: #141414;
    border: 1px solid #2E2E2E;
    border-radius: 3px;
    padding: 1px;
}

QFrame#filterPill QRadioButton {
    padding: 2px 9px;
    border-radius: 2px;
    color: #A0A0A0;
    font-weight: 500;
    spacing: 0px;
}

QFrame#filterPill QRadioButton::indicator {
    width: 0px;
    height: 0px;
    border: none;
    background: transparent;
    margin: 0px;
    padding: 0px;
}

QFrame#filterPill QRadioButton:checked {
    background-color: #2D8CEB;
    color: #FFFFFF;
}

QFrame#filterPill QRadioButton:hover:!checked {
    color: #E6E6E6;
    background-color: #262626;
}

QSlider::groove:horizontal {
    height: 4px;
    background: #2E2E2E;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: #2D8CEB;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #E6E6E6;
    border: 2px solid #2D8CEB;
    width: 10px;
    height: 10px;
    margin: -4px 0;
    border-radius: 2px;
}

QSlider::handle:horizontal:hover {
    background: #FFFFFF;
    border-color: #439BF0;
}

QSlider::handle:horizontal:disabled {
    background: #555555;
    border-color: #3A3A3A;
}

QProgressBar {
    background-color: #141414;
    border: 1px solid #2E2E2E;
    border-radius: 3px;
    max-height: 6px;
    min-height: 6px;
    text-align: center;
    color: transparent;
}

QProgressBar::chunk {
    background-color: #2D8CEB;
    border-radius: 2px;
}

QTableWidget {
    background-color: #191919;
    alternate-background-color: #1D1D1D;
    color: #E6E6E6;
    border: 1px solid #2E2E2E;
    border-radius: 2px;
    gridline-color: #262626;
    selection-background-color: #1F3B5C;
    selection-color: #E6E6E6;
    outline: none;
}

QTableWidget::item {
    padding: 4px 8px;
    border-bottom: 1px solid #262626;
}

QTableWidget::item:selected {
    background-color: #1F3B5C;
    color: #E6E6E6;
}

QHeaderView::section {
    background-color: #232323;
    color: #A0A0A0;
    border: none;
    border-bottom: 1px solid #2E2E2E;
    border-right: 1px solid #2B2B2B;
    padding: 4px 8px;
    font-size: 11px;
    font-weight: 600;
}

QHeaderView::section:last {
    border-right: none;
}

QSplitter::handle {
    background-color: #111111;
}

QSplitter::handle:horizontal {
    width: 3px;
}

QSplitter::handle:vertical {
    height: 3px;
}

QSplitter::handle:hover {
    background-color: #2D8CEB;
}

QScrollBar:vertical {
    background: #141414;
    width: 8px;
    margin: 1px;
    border-radius: 3px;
}

QScrollBar::handle:vertical {
    background: #363636;
    min-height: 24px;
    border-radius: 3px;
}

QScrollBar::handle:vertical:hover {
    background: #4C4C4C;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
    height: 0px;
}

QScrollBar:horizontal {
    background: #141414;
    height: 8px;
    margin: 1px;
    border-radius: 3px;
}

QScrollBar::handle:horizontal {
    background: #363636;
    min-width: 24px;
    border-radius: 3px;
}

QScrollBar::handle:horizontal:hover {
    background: #4C4C4C;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: none;
    width: 0px;
}
"""


def build_studio_dark_qss() -> str:
    assets_dir = runtime_paths.get_ui_assets_dir()
    return (
        _STUDIO_DARK_QSS_TEMPLATE.replace(
            "__CHEVRON_DOWN__", (assets_dir / "chevron_down.svg").as_posix()
        )
        .replace("__CHEVRON_UP__", (assets_dir / "chevron_up.svg").as_posix())
        .replace("__CHECK__", (assets_dir / "check.svg").as_posix())
    )


STUDIO_DARK_QSS = build_studio_dark_qss()


def build_studio_palette() -> QPalette:
    """Create a neutral graphite QPalette matching Muzhen's Premiere/AE tokens."""
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#161616"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#E6E6E6"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#1D1D1D"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#191919"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#262626"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#E6E6E6"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#E6E6E6"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#2A2A2A"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#E6E6E6"))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.Light, QColor("#424242"))
    palette.setColor(QPalette.ColorRole.Midlight, QColor("#333333"))
    palette.setColor(QPalette.ColorRole.Mid, QColor("#2E2E2E"))
    palette.setColor(QPalette.ColorRole.Dark, QColor("#1F1F1F"))
    palette.setColor(QPalette.ColorRole.Shadow, QColor("#0B0B0B"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#2D8CEB"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor("#6E6E6E"))

    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#6E6E6E"))
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#6E6E6E")
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor("#6E6E6E")
    )
    return palette


def apply_studio_theme(app: QApplication) -> None:
    """Apply Fusion style, Premiere/AE Graphite palette, default font, and QSS to QApplication."""
    app.setStyle("Fusion")
    app.setPalette(build_studio_palette())
    font = QFont("Segoe UI")
    font.setPointSizeF(9.5)
    app.setFont(font)
    app.setStyleSheet(build_studio_dark_qss())
