"""CJK subtitle text normalization, 2-tier smart merging, and review flagging."""

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

_CJK_FULLWIDTH_PUNCT = (
    "，。！？；：「」『』（）【】《》、…—―～·“”‘’〈〉〔〕〖〗．［］＋－／％・"
)

_CJK_OR_PUNCT_CLASS = f"{_CJK_CHARS}{re.escape(_CJK_FULLWIDTH_PUNCT)}"
_HAS_CJK_RE = re.compile(rf"[{_CJK_CHARS}]")
_HANZI_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002ebef]")
_EDGE_LATIN_NOISE_RE = re.compile(
    r"(?:^[A-Za-z]{1,2}[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff][A-Za-z]{1,2}$)"
)
_HAS_CONTENT_CHAR_RE = re.compile(rf"[0-9A-Za-z\uff10-\uff19\uff21-\uff3a\uff41-\uff5a{_CJK_CHARS}]")
_SINGLE_KANA_NOISE_RE = re.compile(r"^[\u3040-\u30ff]$")
_PURE_KANA_NOISE_RE = re.compile(r"^[\u3040-\u30ff]+$")
_CJK_ADJACENT_SPACE_RE = re.compile(
    rf"(?<=[{_CJK_OR_PUNCT_CLASS}]) +| +(?=[{_CJK_OR_PUNCT_CLASS}])"
)

_ALLOWED_ASCII_PUNCT = r",.!?%:'\"()/;\-\[\]~&+=°℃￥$×÷♪♫★☆–•"
_DISALLOWED_TEXT_CHAR_RE = re.compile(
    rf"[^0-9a-zA-Z\uff10-\uff19\uff21-\uff3a\uff41-\uff5a{_ALLOWED_ASCII_PUNCT}\n {_CJK_CHARS}{re.escape(_CJK_FULLWIDTH_PUNCT)}]"
)

# Punctuation stripped during similarity comparison (per spec) while preserved in display text.
_SIMILARITY_STRIP_RE = re.compile(
    rf"[\s{re.escape(_CJK_FULLWIDTH_PUNCT)},.!?%:'\"()/;\-\[\]~&+=–•]+"
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
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("―", "—")
    cleaned_lines: list[str] = []
    for raw_line in normalized.split("\n"):
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


def _strip_for_similarity(text: str) -> str:
    norm = normalize_cjk_text(text)
    stripped = _SIMILARITY_STRIP_RE.sub("", norm)
    return stripped if stripped else _ALL_WS_RE.sub("", norm)


def text_similarity(a: str, b: str) -> float:
    """Compute normalized Levenshtein similarity in [0.0, 1.0] ignoring whitespace and noise punctuation."""
    na = _strip_for_similarity(a)
    nb = _strip_for_similarity(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    dist = _levenshtein_distance(na, nb)
    return 1.0 - dist / max(len(na), len(nb))


def _find_suspicious_chars(text: str) -> list[str]:
    return list(dict.fromkeys(_DISALLOWED_TEXT_CHAR_RE.findall(text)))


def _has_suspicious_noise(text: str) -> bool:
    return bool(_DISALLOWED_TEXT_CHAR_RE.search(text))


def _has_edge_latin_noise(text: str) -> bool:
    return any(_EDGE_LATIN_NOISE_RE.search(line) for line in text.split("\n"))


def _is_clean_hanzi_minimal_pair(
    prev: _MergedCue, curr: _MergedCue, config: ExtractConfig
) -> bool:
    """True when two clean high-conf cues (>=4 stripped chars) differ by exactly 1 Hanzi char."""
    if prev.confidence < config.conf_threshold or curr.confidence < config.conf_threshold:
        return False
    if (
        _has_suspicious_noise(prev.text)
        or _has_suspicious_noise(curr.text)
        or _has_edge_latin_noise(prev.text)
        or _has_edge_latin_noise(curr.text)
    ):
        return False
    sa = _strip_for_similarity(prev.text)
    sb = _strip_for_similarity(curr.text)
    if len(sa) != len(sb) or len(sa) < 4:
        return False
    diffs = [(ca, cb) for ca, cb in zip(sa, sb) if ca != cb]
    if len(diffs) != 1:
        return False
    ca, cb = diffs[0]
    return bool(_HANZI_CHAR_RE.fullmatch(ca) and _HANZI_CHAR_RE.fullmatch(cb))


def _is_strict_containment(a: str, b: str) -> bool:
    """True when the shorter text (>=4 chars) is a strict prefix/suffix of the longer by 1-2 chars."""
    sa = _strip_for_similarity(a)
    sb = _strip_for_similarity(b)
    if not sa or not sb or sa == sb:
        return False
    short, long = (sa, sb) if len(sa) < len(sb) else (sb, sa)
    diff = len(long) - len(short)
    if len(short) < 4 or not (1 <= diff <= 2):
        return False
    return long.startswith(short) or long.endswith(short)


def _should_merge(prev: _MergedCue, curr: _MergedCue, config: ExtractConfig) -> bool:
    gap_ms = curr.start_ms - prev.end_ms
    if gap_ms < 0:
        # Guard against blindly merging overlapping distinct cues; only coalesce exact duplicates.
        return _strip_for_similarity(prev.text) == _strip_for_similarity(curr.text)

    if gap_ms <= config.merge_gap_ms:
        if _is_clean_hanzi_minimal_pair(prev, curr, config):
            return False
        if text_similarity(prev.text, curr.text) >= config.merge_similarity:
            return True
        if _is_strict_containment(prev.text, curr.text) and not (
            _has_suspicious_noise(prev.text) or _has_suspicious_noise(curr.text)
        ):
            return True
        return False

    # Tier-2 fallback (merge_gap_ms < gap <= flash_bridge_max_ms):
    # Disallow Levenshtein character substitutions (protects "放过你的" vs "放过他的") and
    # disallow identical repeats (protects genuine repeated cues like "快跑!"..."快跑!").
    # Only merge clean prefix/suffix truncation pairs (e.g. glare-dropped leading "一").
    if gap_ms <= config.flash_bridge_max_ms:
        if _is_strict_containment(prev.text, curr.text) and not (
            _has_suspicious_noise(prev.text) or _has_suspicious_noise(curr.text)
        ):
            return True

    return False


def _pick_merged_winner(
    prev: _MergedCue, curr: _MergedCue, config: ExtractConfig
) -> tuple[str, float, bytes]:
    """Choose the best display text, true winner confidence, and thumbnail when merging two cues."""
    if prev.text == curr.text:
        win_conf = max(prev.confidence, curr.confidence)
        win_thumb = curr.thumbnail_png if curr.confidence > prev.confidence else prev.thumbnail_png
        return (prev.text, win_conf, win_thumb)

    # Prefer the fuller clean text when one is a strict prefix/suffix containment of the other
    # (e.g. "一个蕴藏着神秘力量的" over glare-clipped "个蕴藏着神秘力量的") if confidence is healthy.
    if _is_strict_containment(prev.text, curr.text):
        prev_clean = not _has_suspicious_noise(prev.text)
        curr_clean = not _has_suspicious_noise(curr.text)
        if len(curr.text) > len(prev.text) and curr_clean:
            min_ok = max(config.conf_threshold - 0.05, prev.confidence - 0.08)
            if curr.confidence >= min_ok:
                return (curr.text, curr.confidence, curr.thumbnail_png)
        elif len(prev.text) > len(curr.text) and prev_clean:
            min_ok = max(config.conf_threshold - 0.05, curr.confidence - 0.08)
            if prev.confidence >= min_ok:
                return (prev.text, prev.confidence, prev.thumbnail_png)

    if (
        curr.confidence > prev.confidence
        or (curr.confidence == prev.confidence and len(curr.text) > len(prev.text))
    ):
        return (curr.text, curr.confidence, curr.thumbnail_png)
    return (prev.text, prev.confidence, prev.thumbnail_png)


def _is_ghost_cue(cue: _MergedCue, config: ExtractConfig) -> bool:
    """Return True if cue is a known non-CJK, kana, 1-char ultra-low-conf, or short symbol ghost."""
    if not _HAS_CONTENT_CHAR_RE.search(cue.text):
        return True
    if _SINGLE_KANA_NOISE_RE.match(cue.text):
        return True
    if _PURE_KANA_NOISE_RE.match(cue.text) and cue.confidence < config.conf_threshold:
        return True
    if not _HAS_CJK_RE.search(cue.text) and cue.confidence < config.conf_threshold:
        return True

    stripped = _strip_for_similarity(cue.text)
    dur_ms = cue.end_ms - cue.start_ms
    # Rule A: 1-char ultra-low-conf transient ghost
    if len(stripped) == 1 and cue.confidence < 0.72 and dur_ms < 400:
        return True
    # Rule B: short disallowed-symbol ghost below confidence threshold
    if (
        len(stripped) <= 3
        and _has_suspicious_noise(cue.text)
        and cue.confidence < config.conf_threshold
    ):
        return True
    return False


def _build_flag_info(cue: _MergedCue, config: ExtractConfig) -> tuple[bool, str]:
    reasons: list[str] = []
    if cue.confidence < config.conf_threshold:
        reasons.append(f"Độ tin cậy thấp ({cue.confidence:.2f})")
    bad_chars = _find_suspicious_chars(cue.text)
    if bad_chars:
        sample = " ".join(bad_chars[:4])
        reasons.append(f"Ký tự lạ: {sample}")
    return (bool(reasons), "; ".join(reasons))


def postprocess_segments(
    segments: list[OcrSegment],
    config: ExtractConfig,
    *,
    duration_ms: int | None = None,
) -> list[SubtitleEvent]:
    """Normalize text, merge flickering/truncated cues, filter short/noise cues, flag, and re-index."""
    kept: list[_MergedCue] = []
    for seg in segments:
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
        if _should_merge(prev, curr, config):
            win_text, win_conf, win_thumb = _pick_merged_winner(prev, curr, config)
            prev.text = win_text
            prev.thumbnail_png = win_thumb
            prev.confidence = win_conf
            prev.start_ms = min(prev.start_ms, curr.start_ms)
            prev.end_ms = max(prev.end_ms, curr.end_ms)
        else:
            merged.append(curr)

    if duration_ms is not None and duration_ms > 0:
        clamped: list[_MergedCue] = []
        for cue in merged:
            if cue.start_ms >= duration_ms:
                continue
            cue.end_ms = min(cue.end_ms, duration_ms)
            if cue.end_ms > cue.start_ms:
                clamped.append(cue)
        merged = clamped

    # Filter out zero-duration cues and cues shorter than min_duration_ms AFTER merging so valid
    # leading/trailing fragments attached to a real cue are not discarded prematurely.
    merged = [
        cue
        for cue in merged
        if cue.end_ms > cue.start_ms and (cue.end_ms - cue.start_ms) >= config.min_duration_ms
    ]

    if config.filter_non_cjk_low_conf:
        merged = [cue for cue in merged if not _is_ghost_cue(cue, config)]

    # Clamp overlaps AFTER duration and ghost filtering so discarded ghosts cannot truncate valid cues.
    non_overlapping: list[_MergedCue] = []
    for cue in merged:
        while non_overlapping and non_overlapping[-1].end_ms > cue.start_ms:
            prev_cue = non_overlapping[-1]
            prev_cue.end_ms = cue.start_ms
            if prev_cue.end_ms <= prev_cue.start_ms:
                non_overlapping.pop()
            else:
                break
        if cue.end_ms > cue.start_ms:
            non_overlapping.append(cue)
    merged = non_overlapping

    events: list[SubtitleEvent] = []
    for idx, cue in enumerate(merged, start=1):
        flagged, flag_reason = _build_flag_info(cue, config)
        events.append(
            SubtitleEvent(
                index=idx,
                start_ms=cue.start_ms,
                end_ms=cue.end_ms,
                text=cue.text,
                confidence=cue.confidence,
                flagged=flagged,
                thumbnail_png=cue.thumbnail_png,
                flag_reason=flag_reason,
            )
        )
    return events
