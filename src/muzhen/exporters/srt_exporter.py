"""Premiere Pro compatible SubRip (.srt) subtitle exporter."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from muzhen.core.models import SubtitleEvent
from muzhen.exporters.sanitize import (
    clamp_non_overlapping_intervals,
    prepare_export_cues,
    write_text_atomic,
)


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


def export_srt(events: Sequence[SubtitleEvent], output_path: Path) -> Path:
    """Export subtitle events as clean UTF-8 ``.srt`` (no BOM, LF newlines)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cues = prepare_export_cues(events)
    raw_timings = [(ev.start_ms, max(ev.start_ms + 1, ev.end_ms)) for ev in cues]
    resolved_timings = clamp_non_overlapping_intervals(raw_timings)

    blocks: list[str] = []
    for seq, (event, (start_ms, end_ms)) in enumerate(zip(cues, resolved_timings), start=1):
        blocks.append(
            f"{seq}\n{format_srt_time(start_ms)} --> {format_srt_time(end_ms)}\n{event.text}"
        )

    content = ("\n\n".join(blocks) + "\n") if blocks else ""
    write_text_atomic(out, content, encoding="utf-8", newline="\n")
    return out
