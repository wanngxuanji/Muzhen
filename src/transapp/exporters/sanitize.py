"""Shared cue sanitization, Aegisub midpoint snapping, and non-overlap helpers."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import replace
import re

from transapp.core.models import SubtitleEvent

_ASS_ESCAPE_RE = re.compile(r"\\([Nnh])")


def clean_cue_lines(text: str) -> list[str]:
    """Normalize line endings, strip per-line whitespace, and drop blank lines."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return [stripped for line in normalized.split("\n") if (stripped := line.strip())]


def prepare_export_cues(events: Sequence[SubtitleEvent]) -> list[SubtitleEvent]:
    """Clean, drop empty cues, and stably sort fresh copies for ASS/SRT parity."""
    cleaned: list[SubtitleEvent] = []
    for ev in events:
        lines = clean_cue_lines(ev.text)
        if not lines:
            continue
        start_ms = max(0, int(ev.start_ms))
        end_ms = max(0, int(ev.end_ms))
        cleaned.append(
            replace(
                ev,
                start_ms=start_ms,
                end_ms=end_ms,
                text="\n".join(lines),
            )
        )
    # Python's Timsort is guaranteed stable, preserving original order on ties.
    cleaned.sort(key=lambda ev: (ev.start_ms, ev.end_ms, int(ev.index)))
    return cleaned


def ms_to_ass_cs_half_up(ms: int) -> int:
    """Convert milliseconds to ASS centiseconds using deterministic half-up rounding."""
    return (max(0, int(ms)) + 5) // 10


def _snap_boundary_cs(
    boundary_ms: int,
    frame_pts_ms: Sequence[int],
) -> int:
    """Snap a cue boundary to the Aegisub/libass inter-frame midpoint centisecond."""
    b_ms = max(0, int(boundary_ms))
    first_pts = int(frame_pts_ms[0])
    last_pts = int(frame_pts_ms[-1])

    if b_ms > last_pts:
        return max((last_pts // 10) + 1, ms_to_ass_cs_half_up(b_ms))

    idx = bisect_left(frame_pts_ms, b_ms)
    if idx == 0:
        return min(max(0, first_pts // 10), ms_to_ass_cs_half_up(b_ms))

    prev_pts = int(frame_pts_ms[idx - 1])
    curr_pts = int(frame_pts_ms[idx])
    lo_cs = (prev_pts // 10) + 1
    hi_cs = curr_pts // 10
    if lo_cs <= hi_cs:
        mid_cs = (prev_pts + curr_pts + 10) // 20
        return max(lo_cs, min(hi_cs, mid_cs))
    return ms_to_ass_cs_half_up(b_ms)


def snap_ass_centiseconds(
    start_ms: int,
    end_ms: int,
    frame_pts_ms: Sequence[int] | None,
) -> tuple[int, int]:
    """Snap ``(start_ms, end_ms)`` to ASS centiseconds with exact libass frame visibility.

    When ``frame_pts_ms`` has at least 2 sorted frame timestamps, boundaries snap to
    inter-frame midpoints satisfying ``pts[f-1] < cs * 10 <= pts[f]`` so libass's
    ``start_cs * 10 <= pts[f] < end_cs * 10`` rule matches the intended frame set
    with 0-frame error across 23.976, 25, 29.97, 59.94, and VFR timelines.
    """
    s_ms = max(0, int(start_ms))
    e_ms = max(0, int(end_ms))
    if frame_pts_ms is None or len(frame_pts_ms) < 2:
        start_cs = ms_to_ass_cs_half_up(s_ms)
        end_cs = max(start_cs + 1, ms_to_ass_cs_half_up(e_ms))
        return (start_cs, end_cs)

    start_cs = _snap_boundary_cs(s_ms, frame_pts_ms)
    end_cs = _snap_boundary_cs(e_ms, frame_pts_ms)
    end_cs = max(start_cs + 1, end_cs)
    return (start_cs, end_cs)


def clamp_non_overlapping_intervals(
    intervals: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Enforce strictly positive, non-overlapping integer intervals ``0 <= s_k < e_k <= s_{k+1}``."""
    resolved = [[max(0, int(s)), max(max(0, int(s)) + 1, int(e))] for s, e in intervals]
    for idx in range(1, len(resolved)):
        prev_s, prev_e = resolved[idx - 1]
        curr_s, curr_e = resolved[idx]
        if prev_e > curr_s:
            prev_e = curr_s
            if prev_e <= prev_s:
                prev_e = prev_s + 1
                curr_s = prev_e
                curr_e = max(curr_s + 1, curr_e)
            resolved[idx - 1][1] = prev_e
            resolved[idx][0] = curr_s
            resolved[idx][1] = curr_e
    return [(s, e) for s, e in resolved]


def sanitize_ass_dialogue_text(text: str) -> str:
    """Escape ASS override braces and literal ``\\N``/``\\n``/``\\h`` before joining lines with ``\\N``."""
    escaped_lines: list[str] = []
    for line in clean_cue_lines(text):
        safe = line.replace("{", "\uff5b").replace("}", "\uff5d")
        safe = _ASS_ESCAPE_RE.sub("\\\\\u200b\\1", safe)
        escaped_lines.append(safe)
    return r"\N".join(escaped_lines)
