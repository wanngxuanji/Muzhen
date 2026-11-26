"""Application entry point for launching the TransApp desktop GUI."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from transapp.ui.main_window import MainWindow


def main() -> int:
    """Create or reuse QApplication, open MainWindow, and enter the Qt event loop."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app.setApplicationName("TransApp")
    app.setOrganizationName("TransApp")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
