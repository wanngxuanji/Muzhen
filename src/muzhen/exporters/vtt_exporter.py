"""WebVTT (.vtt) subtitle exporter."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from muzhen.core.models import SubtitleEvent
from muzhen.exporters.sanitize import (
    clamp_non_overlapping_intervals,
    prepare_export_cues,
    write_text_atomic,
)


def format_vtt_time(ms: int) -> str:
    """Format milliseconds into WebVTT timestamp ``HH:MM:SS.mmm`` clamped to >= 0."""
    clamped_ms = max(0, int(ms))
    mmm = clamped_ms % 1000
    total_s = clamped_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{mmm:03d}"


def export_vtt(events: Iterable[SubtitleEvent], output_path: Path | str) -> Path:
    """Export subtitle events as clean UTF-8 WebVTT ``.vtt`` (no BOM, LF newlines)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    cues = prepare_export_cues(list(events))
    raw_timings = [(ev.start_ms, max(ev.start_ms + 1, ev.end_ms)) for ev in cues]
    resolved_timings = clamp_non_overlapping_intervals(raw_timings)

    blocks: list[str] = []
    for event, (start_ms, end_ms) in zip(cues, resolved_timings):
        safe_text = (
            event.text.replace("\r\n", "\n")
            .replace("\r", "\n")
            .replace("-->", "--&gt;")
        )
        blocks.append(f"{format_vtt_time(start_ms)} --> {format_vtt_time(end_ms)}\n{safe_text}")

    content = ("WEBVTT\n\n" + "\n\n".join(blocks) + "\n") if blocks else "WEBVTT\n"
    write_text_atomic(out, content, encoding="utf-8", newline="\n")
    return out
