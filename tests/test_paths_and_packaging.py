from __future__ import annotations

from pathlib import Path
import sys

import pytest


def test_get_ui_assets_dir_resolves_dev_and_frozen_meipass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from transapp.core import paths

    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)

    assert paths.is_frozen() is False
    dev_dir = paths.get_ui_assets_dir()
    assert dev_dir.is_dir()
    for asset_name in ("transapp.ico", "chevron_down.svg", "chevron_up.svg", "check.svg"):
        assert (dev_dir / asset_name).is_file()

    frozen_assets = tmp_path / "transapp" / "ui" / "assets"
    frozen_assets.mkdir(parents=True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

    assert paths.is_frozen() is True
    assert paths.get_ui_assets_dir() == frozen_assets

    from transapp.ui.theme import build_studio_dark_qss

    frozen_qss = build_studio_dark_qss()
    assert (frozen_assets / "chevron_down.svg").as_posix() in frozen_qss
    assert (frozen_assets / "chevron_up.svg").as_posix() in frozen_qss
    assert (frozen_assets / "check.svg").as_posix() in frozen_qss

    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    assert paths.get_ui_assets_dir() == dev_dir


def test_get_models_dir_portable_beside_exe_fallback_and_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from transapp.core import paths

    monkeypatch.delenv("TRANSAPP_MODELS_DIR", raising=False)
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert paths.get_models_dir() is None

    env_dir = tmp_path / "override_models"
    monkeypatch.setenv("TRANSAPP_MODELS_DIR", str(env_dir))
    assert paths.get_models_dir() == env_dir
    assert env_dir.is_dir()

    monkeypatch.delenv("TRANSAPP_MODELS_DIR", raising=False)
    exe_dir = tmp_path / "portable_dist"
    exe_dir.mkdir()
    fake_exe = exe_dir / "TransApp.exe"
    fake_exe.write_bytes(b"")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(fake_exe))

    portable_models = paths.get_models_dir()
    assert portable_models == exe_dir / "models"
    assert portable_models.is_dir()
    assert list(portable_models.iterdir()) == []

    readonly_exe_dir = tmp_path / "program_files_app"
    readonly_exe_dir.mkdir()
    readonly_exe = readonly_exe_dir / "TransApp.exe"
    readonly_exe.write_bytes(b"")
    local_appdata = tmp_path / "local_appdata"
    monkeypatch.setattr(sys, "executable", str(readonly_exe))
    monkeypatch.setenv("LOCALAPPDATA", str(local_appdata))

    orig_write_bytes = Path.write_bytes

    def deny_program_files_write(self: Path, data: bytes) -> int:
        if readonly_exe_dir in self.parents:
            raise PermissionError("Access is denied")
        return orig_write_bytes(self, data)

    monkeypatch.setattr(Path, "write_bytes", deny_program_files_write)

    fallback_models = paths.get_models_dir()
    assert fallback_models == local_appdata / "TransApp" / "models"
    assert fallback_models.is_dir()


def test_ocr_params_include_model_root_dir_when_models_dir_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rapidocr
    from transapp.core import ocr_engine, paths

    monkeypatch.setattr(paths, "get_models_dir", lambda: None)
    dev_rec = ocr_engine._build_rapidocr_params("accurate", use_dml=False)
    dev_det = ocr_engine._build_det_mobile_params(use_dml=False)
    assert "Global.model_root_dir" not in dev_rec
    assert "Det.model_root_dir" not in dev_rec
    assert "Rec.model_root_dir" not in dev_rec
    assert "Global.model_root_dir" not in dev_det
    assert "Det.model_root_dir" not in dev_det
    assert "Rec.model_root_dir" not in dev_det

    custom_models = tmp_path / "portable_models"
    monkeypatch.setattr(paths, "get_models_dir", lambda: custom_models)

    frozen_rec = ocr_engine._build_rapidocr_params("accurate", use_dml=False)
    assert frozen_rec["Global.model_root_dir"] == str(custom_models)
    assert frozen_rec["Det.model_root_dir"] == str(custom_models)
    assert frozen_rec["Rec.model_root_dir"] == str(custom_models)

    frozen_fast = ocr_engine._build_rapidocr_params("fast", use_dml=True)
    assert frozen_fast["Global.model_root_dir"] == str(custom_models)
    assert frozen_fast["Det.model_root_dir"] == str(custom_models)
    assert frozen_fast["Rec.model_root_dir"] == str(custom_models)

    frozen_det = ocr_engine._build_det_mobile_params(use_dml=False)
    assert frozen_det["Global.model_root_dir"] == str(custom_models)
    assert frozen_det["Det.model_root_dir"] == str(custom_models)
    assert "Rec.model_root_dir" not in frozen_det

    engine_instance = rapidocr.RapidOCR(params=frozen_rec)
    assert Path(engine_instance.cfg.Global.model_root_dir) == custom_models
    assert Path(engine_instance.cfg.Det.model_root_dir) == custom_models
    assert Path(engine_instance.cfg.Rec.model_root_dir) == custom_models


def test_app_smoke_test_flag_boots_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QApplication
    import transapp.app as app_mod

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "smoke_appdata"))
    monkeypatch.delenv("TRANSAPP_MODELS_DIR", raising=False)

    def fail_if_exec_called(_self: QApplication) -> int:
        raise AssertionError("QApplication.exec must not be called in --smoke-test mode")

    monkeypatch.setattr(QApplication, "exec", fail_if_exec_called)

    rc = app_mod.main(["--smoke-test"])
    assert rc == 0

    app = QApplication.instance()
    assert isinstance(app, QApplication)
    assert len(app.styleSheet().strip()) > 200
