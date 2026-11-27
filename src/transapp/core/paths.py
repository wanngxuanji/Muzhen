from __future__ import annotations

import os
from pathlib import Path
import sys


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def get_ui_assets_dir() -> Path:
    if is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass) / "transapp" / "ui" / "assets"
    return Path(__file__).resolve().parent.parent / "ui" / "assets"


def _is_writable_dir(directory: Path) -> bool:
    probe = directory / f".transapp_write_probe_{os.getpid()}"
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
    env_dir = os.environ.get("TRANSAPP_MODELS_DIR", "").strip()
    if env_dir:
        models_dir = Path(env_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir

    if not is_frozen():
        return None

    portable_dir = Path(sys.executable).resolve().parent / "models"
    if _is_writable_dir(portable_dir):
        return portable_dir

    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    fallback_base = Path(local_app_data) if local_app_data else Path.home()
    fallback_dir = fallback_base / "TransApp" / "models"
    fallback_dir.mkdir(parents=True, exist_ok=True)
    return fallback_dir
