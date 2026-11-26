"""Premiere Pro compatible SubRip (.srt) subtitle exporter."""

from __future__ import annotations

from pathlib import Path

from transapp.core.models import SubtitleEvent


def format_srt_time(ms: int) -> str:
    """Format milliseconds into SRT timestamp ``HH:MM:SS,mmm`` clamped to >= 0."""
    clamped_ms = max(0, int(ms))
    mmm = clamped_ms % 1000
    total_s = clamped_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{mmm:03d}"


def export_srt(events: list[SubtitleEvent], output_path: Path) -> Path:
    """Export subtitle events as clean UTF-8 ``.srt`` (no BOM, LF newlines)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    blocks: list[str] = []
    for seq, event in enumerate(events, start=1):
        start_ms = max(0, int(event.start_ms))
        end_ms = max(start_ms + 1, int(event.end_ms))
        clean_text = event.text.replace("\r\n", "\n").replace("\r", "\n")
        blocks.append(
            f"{seq}\n{format_srt_time(start_ms)} --> {format_srt_time(end_ms)}\n{clean_text}"
        )

    content = ("\n\n".join(blocks) + "\n") if blocks else ""
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(content)
    return out
