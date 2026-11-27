"""CJK subtitle text normalization, similarity merging, and review flagging."""

from __future__ import annotations

from dataclasses import dataclass
import re

from transapp.core.models import ExtractConfig, OcrSegment, SubtitleEvent

_HORIZONTAL_WS_RE = re.compile(r"[ \t\f\v\u00a0\u3000]+")
_ALL_WS_RE = re.compile(r"\s+")

_CJK_CHARS = (
    r"\u2e80-\u2eff"  # CJK Radicals Supplement
    r"\u2f00-\u2fdf"  # Kangxi Radicals
    r"\u3005\u3007"   # Ideographic iteration mark (々) & zero (〇)
    r"\u3040-\u30ff"  # Hiragana & Katakana
    r"\u3400-\u4dbf"  # CJK Extension A
    r"\u4e00-\u9fff"  # CJK Unified Ideographs
    r"\uac00-\ud7af"  # Hangul Syllables
    r"\uf900-\ufaff"  # CJK Compatibility Ideographs
    r"\U00020000-\U0002ebef"  # CJK Extensions B-G
)

_CJK_FULLWIDTH_PUNCT = "，。！？；：「」『』（）【】《》、…—～·“”‘’〈〉〔〕"

_CJK_OR_PUNCT_CLASS = f"{_CJK_CHARS}{re.escape(_CJK_FULLWIDTH_PUNCT)}"
_CJK_ADJACENT_SPACE_RE = re.compile(
    rf"(?<=[{_CJK_OR_PUNCT_CLASS}]) +| +(?=[{_CJK_OR_PUNCT_CLASS}])"
)

_DISALLOWED_TEXT_CHAR_RE = re.compile(
    rf"[^0-9a-zA-Z,.!?%:'\"()/;\-\n {_CJK_CHARS}{re.escape(_CJK_FULLWIDTH_PUNCT)}]"
)


@dataclass(slots=True)
class _MergedCue:
    start_ms: int
    end_ms: int
    text: str
    confidence: float
    thumbnail_png: bytes


def normalize_cjk_text(text: str) -> str:
    """Normalize OCR subtitle text by removing spurious CJK spaces and trimming lines."""
    normalized_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines: list[str] = []
    for raw_line in normalized_newlines.split("\n"):
        collapsed = _HORIZONTAL_WS_RE.sub(" ", raw_line).strip(" ")
        if not collapsed:
            continue
        cleaned = _CJK_ADJACENT_SPACE_RE.sub("", collapsed).strip(" ")
        if cleaned:
            cleaned_lines.append(cleaned)
    return "\n".join(cleaned_lines)


def _levenshtein_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def text_similarity(a: str, b: str) -> float:
    """Compute normalized Levenshtein similarity in [0.0, 1.0] on whitespace-stripped normalized text."""
    na = _ALL_WS_RE.sub("", normalize_cjk_text(a))
    nb = _ALL_WS_RE.sub("", normalize_cjk_text(b))
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    dist = _levenshtein_distance(na, nb)
    return 1.0 - dist / max(len(na), len(nb))


def _has_suspicious_noise(text: str) -> bool:
    return bool(_DISALLOWED_TEXT_CHAR_RE.search(text))


def postprocess_segments(
    segments: list[OcrSegment],
    config: ExtractConfig,
) -> list[SubtitleEvent]:
    """Filter short/empty OCR segments, merge flickering duplicates, flag noisy cues, and re-index."""
    kept: list[_MergedCue] = []
    for seg in segments:
        if (seg.end_ms - seg.start_ms) < config.min_duration_ms:
            continue
        norm_text = normalize_cjk_text(seg.text)
        if not norm_text:
            continue
        kept.append(
            _MergedCue(
                start_ms=seg.start_ms,
                end_ms=seg.end_ms,
                text=norm_text,
                confidence=float(seg.confidence),
                thumbnail_png=seg.thumbnail_png,
            )
        )

    if not kept:
        return []

    merged: list[_MergedCue] = [kept[0]]
    for curr in kept[1:]:
        prev = merged[-1]
        gap_ms = curr.start_ms - prev.end_ms
        if (
            gap_ms <= config.merge_gap_ms
            and text_similarity(prev.text, curr.text) >= config.merge_similarity
        ):
            if (
                curr.confidence > prev.confidence
                or (curr.confidence == prev.confidence and len(curr.text) > len(prev.text))
            ):
                prev.text = curr.text
                prev.thumbnail_png = curr.thumbnail_png
            prev.confidence = max(prev.confidence, curr.confidence)
            prev.start_ms = min(prev.start_ms, curr.start_ms)
            prev.end_ms = max(prev.end_ms, curr.end_ms)
        else:
            merged.append(curr)

    return [
        SubtitleEvent(
            index=idx,
            start_ms=cue.start_ms,
            end_ms=cue.end_ms,
            text=cue.text,
            confidence=cue.confidence,
            flagged=(cue.confidence < config.conf_threshold) or _has_suspicious_noise(cue.text),
            thumbnail_png=cue.thumbnail_png,
        )
        for idx, cue in enumerate(merged, start=1)
    ]
