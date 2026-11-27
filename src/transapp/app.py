"""Application entry point for launching the TransApp desktop GUI."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from transapp.ui.main_window import MainWindow
from transapp.ui.theme import apply_studio_theme


def main() -> int:
    """Create or reuse QApplication, apply Studio Dark theme, open MainWindow, and run."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app.setApplicationName("TransApp")
    app.setOrganizationName("TransApp")
    apply_studio_theme(app)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
