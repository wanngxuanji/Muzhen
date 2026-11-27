"""Pro Video Studio Dark theme (palette + QSS) for TransApp."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_CHEVRON_DOWN_URL = (_ASSETS_DIR / "chevron_down.svg").as_posix()
_CHEVRON_UP_URL = (_ASSETS_DIR / "chevron_up.svg").as_posix()
_CHECK_URL = (_ASSETS_DIR / "check.svg").as_posix()

PALETTE_TOKENS: dict[str, str] = {
    "bg_root": "#0E1014",
    "bg_surface": "#15181F",
    "bg_elevated": "#1B1F28",
    "bg_input": "#11141A",
    "border_subtle": "#252A34",
    "border_strong": "#333A48",
    "text_primary": "#EEF2F7",
    "text_secondary": "#9BA4B5",
    "text_muted": "#6E7789",
    "accent": "#2563EB",
    "accent_hover": "#3B82F6",
    "accent_pressed": "#1D4ED8",
    "roi_emerald": "#10B981",
    "warn_bg": "#2B2110",
    "warn_text": "#FCD34D",
    "danger_bg": "#2A161A",
    "danger_border": "#7F1D1D",
    "danger_text": "#FCA5A5",
}

MONO_FONT_FAMILY = "Cascadia Mono, Consolas, Courier New, monospace"

STUDIO_DARK_QSS = """
/* === Base Canvas & Typography === */
QMainWindow, QDialog {
    background-color: #0E1014;
    color: #EEF2F7;
}

QWidget {
    color: #EEF2F7;
    font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
    font-size: 12px;
}

QToolTip {
    background-color: #1F242E;
    color: #EEF2F7;
    border: 1px solid #333A48;
    border-radius: 4px;
    padding: 4px 8px;
}

/* === Top Header Bar === */
QFrame#topBar {
    background-color: #15181F;
    border: 1px solid #252A34;
    border-radius: 8px;
}

QLabel#brandTitle {
    font-size: 14px;
    font-weight: 700;
    color: #F8FAFC;
    letter-spacing: 0.2px;
}

QLabel#versionBadge {
    background-color: rgba(37, 99, 235, 0.18);
    color: #60A5FA;
    border: 1px solid rgba(59, 130, 246, 0.35);
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 10px;
    font-weight: 600;
}

QLabel#metaBadge {
    background-color: #1B1F28;
    color: #93C5FD;
    border: 1px solid #2A3346;
    border-radius: 4px;
    padding: 2px 8px;
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 11px;
}

/* === Studio Surface Cards === */
QFrame[card="true"] {
    background-color: #15181F;
    border: 1px solid #252A34;
    border-radius: 8px;
}

QFrame[divider="true"] {
    background-color: #222732;
    max-height: 1px;
    min-height: 1px;
    border: none;
}

QLabel[role="sectionTitle"] {
    color: #8A94A6;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.9px;
}

QLabel[role="fieldLabel"] {
    color: #A6B0C2;
    font-size: 12px;
}

QLabel[role="mono"] {
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 11px;
    color: #C7D0DF;
}

QLabel#timecodeBadge {
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 11px;
    color: #E2E8F0;
    background-color: #11141A;
    border: 1px solid #252A34;
    border-radius: 5px;
    padding: 3px 8px;
}

QLabel#roiPill {
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 11px;
    color: #6EE7B7;
    background-color: rgba(16, 185, 129, 0.10);
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: 4px;
    padding: 2px 8px;
}

QLabel#trimRangeBadge {
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 11px;
    color: #8A94A6;
    background-color: #11141A;
    border: 1px solid #252A34;
    border-radius: 4px;
    padding: 2px 8px;
}

QLabel#trimRangeBadge[active="true"] {
    color: #6EE7B7;
    background-color: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
}

QLabel#testResultLabel {
    color: #CBD5E1;
    font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
    font-size: 12px;
    padding: 0 4px;
}

QLabel#statusLabel {
    color: #9BA4B5;
    font-size: 11px;
}

/* === ScrollArea === */
QScrollArea#inspectorScroll {
    background: transparent;
    border: none;
}
QScrollArea#inspectorScroll > QWidget > QWidget {
    background: transparent;
}

/* === Inputs === */
QLineEdit, QComboBox, QDoubleSpinBox {
    background-color: #11141A;
    color: #EEF2F7;
    border: 1px solid #2D3442;
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 18px;
    selection-background-color: #2563EB;
}

QLineEdit:hover, QComboBox:hover, QDoubleSpinBox:hover {
    border-color: #465066;
}

QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus {
    border-color: #3B82F6;
}

QLineEdit:disabled, QComboBox:disabled, QDoubleSpinBox:disabled {
    background-color: #12151B;
    color: #5A6374;
    border-color: #202530;
}

QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 24px;
    border-left: 1px solid #252A34;
    border-top-right-radius: 5px;
    border-bottom-right-radius: 5px;
    background-color: #181C24;
}

QComboBox::down-arrow {
    image: url("__CHEVRON_DOWN__");
    width: 12px;
    height: 12px;
}

QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    width: 20px;
    border-left: 1px solid #252A34;
    background-color: #181C24;
}

QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #252C3A;
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
    background-color: #1B1F28;
    color: #EEF2F7;
    border: 1px solid #333A48;
    selection-background-color: #2563EB;
    selection-color: #FFFFFF;
    outline: none;
    padding: 2px;
}

/* === Buttons === */
QPushButton {
    background-color: #1E232E;
    color: #E2E8F0;
    border: 1px solid #303746;
    border-radius: 6px;
    padding: 5px 12px;
    font-weight: 500;
    min-height: 18px;
}

QPushButton:hover {
    background-color: #272D3B;
    border-color: #414B5F;
}

QPushButton:pressed {
    background-color: #191D26;
}

QPushButton:disabled {
    background-color: #161920;
    color: #525B6D;
    border-color: #222630;
}

QPushButton[variant="primary"] {
    background-color: #2563EB;
    color: #FFFFFF;
    border: 1px solid #3B82F6;
    font-weight: 600;
    padding: 6px 14px;
}

QPushButton[variant="primary"]:hover {
    background-color: #3B82F6;
    border-color: #60A5FA;
}

QPushButton[variant="primary"]:pressed {
    background-color: #1D4ED8;
}

QPushButton[variant="primary"]:disabled {
    background-color: #1A2742;
    color: #64748B;
    border-color: #1E3259;
}

QPushButton[variant="danger"] {
    background-color: #2A161A;
    color: #FCA5A5;
    border: 1px solid #5C2229;
    font-weight: 500;
}

QPushButton[variant="danger"]:hover {
    background-color: #3F1B22;
    border-color: #B91C1C;
    color: #FECACA;
}

QPushButton[variant="danger"]:pressed {
    background-color: #221115;
}

QPushButton[variant="danger"]:disabled {
    background-color: #161920;
    color: #525B6D;
    border-color: #222630;
}

QPushButton[variant="tool"] {
    background-color: #1B1F28;
    border: 1px solid #2A303C;
    border-radius: 5px;
    padding: 3px 6px;
    min-height: 20px;
}

QPushButton[variant="tool"]:hover {
    background-color: #252B37;
    border-color: #3D4657;
}

QPushButton[variant="tool"]:disabled {
    background-color: #14171D;
    border-color: #1E222B;
}

QPushButton[variant="trim"] {
    background-color: #1B1F28;
    color: #DCE3EE;
    border: 1px solid #2A303C;
    border-radius: 5px;
    padding: 2px 8px;
    font-size: 11px;
    font-weight: 500;
    min-height: 18px;
}

QPushButton[variant="trim"]:hover {
    background-color: #252B37;
    border-color: #3D4657;
}

QPushButton[variant="trim"]:disabled {
    background-color: #14171D;
    color: #525B6D;
    border-color: #1E222B;
}

/* === Checkboxes & Radios === */
QCheckBox, QRadioButton {
    spacing: 6px;
    color: #DCE3EE;
}

QCheckBox:disabled, QRadioButton:disabled {
    color: #5A6374;
}

QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border-radius: 4px;
    border: 1.5px solid #4B5563;
    background-color: #11141A;
}

QCheckBox::indicator:hover {
    border-color: #60A5FA;
}

QCheckBox::indicator:checked {
    background-color: #2563EB;
    border-color: #3B82F6;
    image: url("__CHECK__");
}

QRadioButton::indicator {
    width: 14px;
    height: 14px;
    border-radius: 8px;
    border: 1.5px solid #4B5563;
    background-color: #11141A;
}

QRadioButton::indicator:hover {
    border-color: #60A5FA;
}

QRadioButton::indicator:checked {
    border: 1.5px solid #2563EB;
    background-color: qradialgradient(
        cx:0.5, cy:0.5, radius:0.5,
        fx:0.5, fy:0.5,
        stop:0 #FFFFFF,
        stop:0.42 #FFFFFF,
        stop:0.52 #2563EB,
        stop:1 #2563EB
    );
}

/* === Filter Pill Radios === */
QFrame#filterPill {
    background-color: #11141A;
    border: 1px solid #252A34;
    border-radius: 6px;
    padding: 2px;
}

QFrame#filterPill QRadioButton {
    padding: 3px 10px;
    border-radius: 4px;
    color: #9BA4B5;
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
    background-color: #2563EB;
    color: #FFFFFF;
}

QFrame#filterPill QRadioButton:hover:!checked {
    color: #EEF2F7;
    background-color: #1B1F28;
}

/* === Transport Slider === */
QSlider::groove:horizontal {
    height: 4px;
    background: #252A34;
    border-radius: 2px;
}

QSlider::sub-page:horizontal {
    background: #2563EB;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #F8FAFC;
    border: 2px solid #2563EB;
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #FFFFFF;
    border-color: #60A5FA;
}

QSlider::handle:horizontal:disabled {
    background: #475569;
    border-color: #334155;
}

/* === Progress Bar === */
QProgressBar {
    background-color: #11141A;
    border: 1px solid #252A34;
    border-radius: 4px;
    max-height: 8px;
    min-height: 8px;
    text-align: center;
    color: transparent;
}

QProgressBar::chunk {
    background-color: #2563EB;
    border-radius: 3px;
}

/* === Review Table === */
QTableWidget {
    background-color: #11141A;
    alternate-background-color: #141820;
    color: #EEF2F7;
    border: 1px solid #252A34;
    border-radius: 6px;
    gridline-color: #1C212B;
    selection-background-color: #1E3A5F;
    selection-color: #FFFFFF;
    outline: none;
}

QTableWidget::item {
    padding: 4px 8px;
    border-bottom: 1px solid #1A1F28;
}

QTableWidget::item:selected {
    background-color: #1E3A5F;
    color: #FFFFFF;
}

QHeaderView::section {
    background-color: #191D26;
    color: #8A94A6;
    border: none;
    border-bottom: 1px solid #252A34;
    border-right: 1px solid #1F242E;
    padding: 5px 8px;
    font-size: 11px;
    font-weight: 600;
}

QHeaderView::section:last {
    border-right: none;
}

/* === Splitters === */
QSplitter::handle {
    background-color: transparent;
}

QSplitter::handle:horizontal {
    width: 6px;
}

QSplitter::handle:vertical {
    height: 6px;
}

QSplitter::handle:hover {
    background-color: rgba(37, 99, 235, 0.45);
    border-radius: 2px;
}

/* === Slim Dark Scrollbars === */
QScrollBar:vertical {
    background: #11141A;
    width: 8px;
    margin: 2px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: #2C3342;
    min-height: 24px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #3F495E;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
    height: 0px;
}

QScrollBar:horizontal {
    background: #11141A;
    height: 8px;
    margin: 2px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal {
    background: #2C3342;
    min-width: 24px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal:hover {
    background: #3F495E;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: none;
    width: 0px;
}
""".replace("__CHEVRON_DOWN__", _CHEVRON_DOWN_URL).replace(
    "__CHEVRON_UP__", _CHEVRON_UP_URL
).replace("__CHECK__", _CHECK_URL)


def build_studio_palette() -> QPalette:
    """Create a dark QPalette matching TransApp's Studio Dark tokens."""
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#0E1014"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#EEF2F7"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#1D222D"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#15181F"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#1F242E"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#EEF2F7"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#EEF2F7"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#1E232E"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#E2E8F0"))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.Light, QColor("#3A4254"))
    palette.setColor(QPalette.ColorRole.Midlight, QColor("#2D3442"))
    palette.setColor(QPalette.ColorRole.Mid, QColor("#4B5563"))
    palette.setColor(QPalette.ColorRole.Dark, QColor("#333A48"))
    palette.setColor(QPalette.ColorRole.Shadow, QColor("#090B0E"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#2563EB"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor("#64748B"))

    # Disabled state palette roles
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#5A6374"))
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#525B6D")
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor("#5A6374")
    )
    return palette


def apply_studio_theme(app: QApplication) -> None:
    """Apply Fusion style, Studio Dark palette, crisp default font, and QSS to QApplication."""
    app.setStyle("Fusion")
    app.setPalette(build_studio_palette())
    font = QFont("Segoe UI")
    font.setPointSizeF(9.5)
    app.setFont(font)
    app.setStyleSheet(STUDIO_DARK_QSS)
