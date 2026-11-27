"""PySide6 desktop UI components for TransApp."""

from transapp.ui.icons import ICON_NAMES, make_icon
from transapp.ui.main_window import MainWindow
from transapp.ui.review_table import ReviewTable
from transapp.ui.roi_canvas import RoiCanvas
from transapp.ui.theme import STUDIO_DARK_QSS, apply_studio_theme

__all__ = [
    "ICON_NAMES",
    "MainWindow",
    "ReviewTable",
    "RoiCanvas",
    "STUDIO_DARK_QSS",
    "apply_studio_theme",
    "make_icon",
]
