from __future__ import annotations

import os
from pathlib import Path
import sys

APP_NAME = "Muzhen"
MODELS_DIR_ENV = "MUZHEN_MODELS_DIR"
DATA_DIR_ENV = "MUZHEN_DATA_DIR"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def get_app_data_dir(*, create: bool = True) -> Path:
    env_dir = os.environ.get(DATA_DIR_ENV, "").strip()
    if env_dir:
        data_dir = Path(env_dir)
    else:
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if local_app_data:
            data_dir = Path(local_app_data) / APP_NAME
        else:
            data_dir = Path.home() / ".muzhen"
    if create:
        data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_ui_assets_dir() -> Path:
    if is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass) / "muzhen" / "ui" / "assets"
    return Path(__file__).resolve().parent.parent / "ui" / "assets"


def _is_writable_dir(directory: Path) -> bool:
    probe = directory / f".muzhen_write_probe_{os.getpid()}"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe.write_bytes(b"")
    except OSError:
        return False
    else:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
        return True


def get_models_dir() -> Path | None:
    env_dir = os.environ.get(MODELS_DIR_ENV, "").strip()
    if env_dir:
        models_dir = Path(env_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir

    if not is_frozen():
        return None

    portable_dir = Path(sys.executable).resolve().parent / "models"
    if _is_writable_dir(portable_dir):
        return portable_dir

    fallback_dir = get_app_data_dir() / "models"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    return fallback_dir
