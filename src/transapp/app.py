"""Application entry point for launching the TransApp desktop GUI."""

from __future__ import annotations

import ctypes
from pathlib import Path
import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from transapp.ui.main_window import MainWindow
from transapp.ui.theme import apply_studio_theme

_APP_ICON_PATH = Path(__file__).resolve().parent / "ui" / "assets" / "transapp.ico"


def _set_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("TransApp.Desktop.1")
    except Exception:
        pass


def load_app_icon() -> QIcon:
    """Load the bundled multi-resolution TransApp window/taskbar icon."""
    if _APP_ICON_PATH.is_file():
        return QIcon(str(_APP_ICON_PATH))
    return QIcon()


def main() -> int:
    """Create or reuse QApplication, apply Studio Dark theme, open MainWindow, and run."""
    _set_windows_app_id()
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app.setApplicationName("TransApp")
    app.setOrganizationName("TransApp")
    apply_studio_theme(app)

    icon = load_app_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)

    window = MainWindow()
    if not icon.isNull():
        window.setWindowIcon(icon)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
