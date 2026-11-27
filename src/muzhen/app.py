from __future__ import annotations

import ctypes
from datetime import datetime
import os
from pathlib import Path
import sys
import traceback

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

from muzhen.core import paths as runtime_paths
from muzhen.ui.main_window import MainWindow
from muzhen.ui.theme import apply_studio_theme


def _crash_log_path() -> Path:
    return runtime_paths.get_app_data_dir() / "crash.log"


def _write_startup_crash_log(tb_text: str) -> Path | None:
    log_path = _crash_log_path()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"=== Muzhen startup crash ({timestamp}) ===\n{tb_text}\n")
        return log_path
    except Exception:
        return None


def _set_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Muzhen.Desktop.App")
    except Exception:
        pass


def load_app_icon() -> QIcon:
    icon_path = runtime_paths.get_ui_assets_dir() / "muzhen.ico"
    if icon_path.is_file():
        return QIcon(str(icon_path))
    return QIcon()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    smoke_test = "--smoke-test" in args
    app: QApplication | None = None
    try:
        if smoke_test:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        _set_windows_app_id()
        existing = QApplication.instance()
        if isinstance(existing, QApplication):
            app = existing
        else:
            qt_argv = [sys.argv[0], *args] if argv is not None else sys.argv
            app = QApplication(qt_argv)
        app.setApplicationName("Muzhen")
        app.setOrganizationName("Muzhen")
        apply_studio_theme(app)

        icon = load_app_icon()
        if not icon.isNull():
            app.setWindowIcon(icon)

        if smoke_test:
            assets_dir = runtime_paths.get_ui_assets_dir()
            if not assets_dir.is_dir():
                raise RuntimeError(f"UI assets directory not found: {assets_dir}")
            runtime_paths.get_models_dir()
            window = MainWindow()
            if not icon.isNull():
                window.setWindowIcon(icon)
            if not app.styleSheet().strip():
                raise RuntimeError("Application stylesheet is empty")
            window.close()
            return 0

        window = MainWindow()
        if not icon.isNull():
            window.setWindowIcon(icon)
        window.show()
        return app.exec()
    except Exception as exc:
        tb_text = traceback.format_exc()
        log_path = _write_startup_crash_log(tb_text)
        if smoke_test:
            sys.stderr.write(tb_text)
            return 1
        try:
            if app is None:
                existing = QApplication.instance()
                app = existing if isinstance(existing, QApplication) else QApplication(sys.argv)
            log_note = f"\n\nChi tiết lỗi đã lưu tại:\n{log_path}" if log_path is not None else ""
            QMessageBox.critical(
                None,
                "Muzhen - Lỗi khởi động",
                f"Không thể khởi động Muzhen:\n{exc}{log_note}",
            )
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
