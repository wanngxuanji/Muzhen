"""UTF-8 JSON persistence for ROI presets."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from transapp.core.models import RoiRect


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
        raw = self.presets_path.read_text(encoding="utf-8")
        if not raw.strip():
            return {}
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}

    def _write_all(self, data: dict[str, dict]) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        self.presets_path.write_text(payload, encoding="utf-8")

    def list_names(self) -> list[str]:
        """Return saved preset names."""
        return list(self._read_all().keys())

    def load(self, name: str) -> RoiPreset | None:
        """Load a preset by name, or return None if absent."""
        entry = self._read_all().get(name)
        if entry is None:
            return None
        roi_data = entry["roi"]
        return RoiPreset(
            name=entry["name"],
            roi=RoiRect(
                x=float(roi_data["x"]),
                y=float(roi_data["y"]),
                w=float(roi_data["w"]),
                h=float(roi_data["h"]),
            ),
            ass_template_path=entry.get("ass_template_path"),
            conf_threshold=float(entry.get("conf_threshold", 0.85)),
        )

    def save(self, preset: RoiPreset) -> None:
        """Save or overwrite a preset."""
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
