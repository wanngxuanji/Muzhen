"""PySide6 desktop UI components for Muzhen."""

from muzhen.ui.icons import ICON_NAMES, make_icon
from muzhen.ui.main_window import MainWindow
from muzhen.ui.review_table import ReviewTable
from muzhen.ui.roi_canvas import RoiCanvas
from muzhen.ui.theme import STUDIO_DARK_QSS, apply_studio_theme

__all__ = [
    "ICON_NAMES",
    "MainWindow",
    "ReviewTable",
    "RoiCanvas",
    "STUDIO_DARK_QSS",
    "apply_studio_theme",
    "make_icon",
]
