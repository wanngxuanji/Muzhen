"""Plain text (.txt) subtitle script exporter."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from muzhen.core.models import SubtitleEvent
from muzhen.exporters.sanitize import (
    prepare_export_cues,
    write_text_atomic,
)


def export_txt(events: Iterable[SubtitleEvent], output_path: Path | str) -> Path:
    """Export subtitle events as clean UTF-8 plain text (one dialogue line per cue)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cues = prepare_export_cues(list(events))
    lines = [" ".join(event.text.splitlines()) for event in cues]

    content = ("\n".join(lines) + "\n") if lines else ""
    write_text_atomic(out, content, encoding="utf-8", newline="\n")
    return out
