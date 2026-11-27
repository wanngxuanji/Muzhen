"""UTF-8 JSON persistence for ROI presets."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from transapp.core.models import RoiRect

logger = logging.getLogger(__name__)


def _parse_finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


@dataclass(frozen=True, slots=True)
class RoiPreset:
    """Named ROI preset with optional ASS template and confidence threshold."""

    name: str
    roi: RoiRect
    ass_template_path: str | None = None
    conf_threshold: float = 0.85


def _default_config_dir() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "TransApp"
    return Path.home() / ".transapp"


class PresetStore:
    """Persists named ROI presets in a UTF-8 JSON file outside video directories."""

    def __init__(self, config_dir: Path | None = None) -> None:
        self.config_dir = Path(config_dir) if config_dir is not None else _default_config_dir()
        self.presets_path = self.config_dir / "presets.json"

    def _read_all(self) -> dict[str, dict]:
        if not self.presets_path.exists():
            return {}
        try:
            raw = self.presets_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("Failed to read presets file %s: %s", self.presets_path, exc)
            return {}
        if not raw.strip():
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Corrupt presets JSON in %s: %s", self.presets_path, exc)
            return {}
        if not isinstance(data, dict):
            logger.warning("Invalid presets payload in %s: expected JSON object", self.presets_path)
            return {}
        return data

    def _write_all(self, data: dict[str, dict]) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        fd, tmp_name = tempfile.mkstemp(
            dir=self.config_dir,
            prefix=f".{self.presets_path.name}.",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, self.presets_path)
        except BaseException:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def list_names(self) -> list[str]:
        """Return saved preset names."""
        return list(self._read_all().keys())

    def load(self, name: str) -> RoiPreset | None:
        """Load a preset by name, or return None if absent or malformed."""
        entry = self._read_all().get(name)
        if entry is None:
            return None
        if not isinstance(entry, dict):
            logger.warning("Malformed preset %r: expected object", name)
            return None

        entry_name = entry.get("name")
        if not isinstance(entry_name, str) or not entry_name.strip():
            logger.warning("Malformed preset %r: invalid name", name)
            return None

        roi_data = entry.get("roi")
        if not isinstance(roi_data, dict):
            logger.warning("Malformed preset %r: invalid roi", name)
            return None

        coords: dict[str, float] = {}
        for key in ("x", "y", "w", "h"):
            parsed = _parse_finite_float(roi_data.get(key))
            if parsed is None:
                logger.warning("Malformed preset %r: invalid roi.%s", name, key)
                return None
            coords[key] = parsed

        conf_threshold = _parse_finite_float(entry.get("conf_threshold", 0.85))
        if conf_threshold is None or not (0.0 <= conf_threshold <= 1.0):
            logger.warning("Malformed preset %r: invalid conf_threshold", name)
            return None

        ass_template = entry.get("ass_template_path")
        if ass_template is not None and not isinstance(ass_template, str):
            logger.warning("Malformed preset %r: invalid ass_template_path", name)
            return None

        return RoiPreset(
            name=entry_name,
            roi=RoiRect(
                x=coords["x"],
                y=coords["y"],
                w=coords["w"],
                h=coords["h"],
            ),
            ass_template_path=ass_template,
            conf_threshold=conf_threshold,
        )

    def save(self, preset: RoiPreset) -> None:
        """Save or overwrite a preset."""
        if not isinstance(preset.name, str) or not preset.name.strip():
            raise ValueError("Preset name must be non-empty")
        for coord_name in ("x", "y", "w", "h"):
            val = getattr(preset.roi, coord_name)
            if _parse_finite_float(val) is None:
                raise ValueError(f"Preset ROI coordinate {coord_name!r} must be finite")
        conf = _parse_finite_float(preset.conf_threshold)
        if conf is None or not (0.0 <= conf <= 1.0):
            raise ValueError("Preset conf_threshold must be a finite float in [0.0, 1.0]")

        data = self._read_all()
        data[preset.name] = asdict(preset)
        self._write_all(data)

    def delete(self, name: str) -> bool:
        """Delete a preset by name; returns True if it existed."""
        data = self._read_all()
        if name not in data:
            return False
        del data[name]
        self._write_all(data)
        return True
