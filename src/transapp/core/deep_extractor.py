"""2-Stage Deep Learning hardsub extractor (ch_PP-OCRv5 Det Mobile Grid + Server Majority Voting)."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import re

import cv2
import numpy as np

from transapp.core.models import ExtractConfig, ExtractionCancelled, OcrSegment
from transapp.core.ocr_engine import OcrEngine
from transapp.core.postprocess import normalize_cjk_text, text_similarity
from transapp.core.video_reader import FrameSlice

_MAX_SPAN_SAMPLES = 12
_MAX_OCR_VOTES_PER_SPAN = 5
_DET_BATCH_SIZE = 8
_RECT_SIMILARITY_TOL = 0.12
_HANZI_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_EDGE_LATIN_NOISE_RE = re.compile(
    r"(?:^[A-Za-z0-9]{1,3}[\u4e00-\u9fff]|[\u4e00-\u9fff][A-Za-z0-9]{1,3}$)"
)


@dataclass(slots=True)
class _CompressedSample:
    pts_ms: int
    jpeg_bytes: bytes


@dataclass(slots=True)
class _DetectedSpan:
    start_ms: int
    end_ms: int
    union_rect: tuple[float, float, float, float]
    samples: list[_CompressedSample] = field(default_factory=list)
    total_seen: int = 0
    stride: int = 1
    hit_count: int = 0

    def add_sample(self, pts_ms: int, bgr_roi: np.ndarray) -> None:
        self.hit_count += 1
        idx = self.total_seen
        self.total_seen += 1
        if idx % self.stride != 0:
            return
        if len(self.samples) >= _MAX_SPAN_SAMPLES:
            # Keep first, decimate interior, keep latest coverage bounded in RAM
            self.samples = self.samples[::2]
            self.stride *= 2
            if idx % self.stride != 0:
                return
        ok, buf = cv2.imencode(".jpg", bgr_roi, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if ok:
            self.samples.append(_CompressedSample(pts_ms=pts_ms, jpeg_bytes=buf.tobytes()))


@dataclass(slots=True)
class _FrameVote:
    pts_ms: int
    text: str
    conf: float
    bgr: np.ndarray


@dataclass(slots=True)
class _VoteCluster:
    start_ms: int
    end_ms: int
    votes: list[_FrameVote] = field(default_factory=list)


def _merge_boxes_to_union_rect(
    boxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (x1, y1, x2, y2)


def _rects_similar(
    r1: tuple[float, float, float, float],
    r2: tuple[float, float, float, float],
    tol: float = _RECT_SIMILARITY_TOL,
) -> bool:
    """Compare horizontal span stability + vertical overlap (tolerant of DBNet vertical jitter)."""
    x_inter = max(0.0, min(r1[2], r2[2]) - max(r1[0], r2[0]))
    x_union = max(1e-3, max(r1[2], r2[2]) - min(r1[0], r2[0]))
    x_iou = x_inter / x_union

    y_inter = max(0.0, min(r1[3], r2[3]) - max(r1[1], r2[1]))
    min_h = max(1e-3, min(r1[3] - r1[1], r2[3] - r2[1]))
    y_overlap = y_inter / min_h
    if y_overlap < 0.45:
        return False

    if x_iou >= 0.72:
        return True

    w1 = max(1e-3, r1[2] - r1[0])
    w2 = max(1e-3, r2[2] - r2[0])
    cx1 = 0.5 * (r1[0] + r1[2])
    cx2 = 0.5 * (r2[0] + r2[2])
    max_w = max(w1, w2)
    return abs(w1 - w2) <= max(0.06, tol * 2.0 * max_w) and abs(cx1 - cx2) <= tol * 1.25


def _decode_sample(sample: _CompressedSample) -> np.ndarray | None:
    arr = np.frombuffer(sample.jpeg_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _is_same_cue_text(a: str, b: str) -> bool:
    if a == b:
        return True
    if text_similarity(a, b) >= 0.72:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) >= 3 and (long.startswith(short) or long.endswith(short)):
        return (len(long) - len(short)) <= max(2, int(0.40 * len(long)))
    return False


def _choose_majority_vote(
    votes: list[_FrameVote],
    ocr_engine: OcrEngine,
    config: ExtractConfig,
) -> tuple[str, float, np.ndarray]:
    counts: Counter[str] = Counter()
    confs: dict[str, list[float]] = defaultdict(list)
    best_frame_for_text: dict[str, tuple[float, np.ndarray]] = {}

    for v in votes:
        counts[v.text] += 1
        confs[v.text].append(v.conf)
        prev = best_frame_for_text.get(v.text)
        if prev is None or v.conf > prev[0]:
            best_frame_for_text[v.text] = (v.conf, v.bgr)

    def rank_key(txt: str) -> tuple[int, int, float, int]:
        clean_bonus = 0 if _EDGE_LATIN_NOISE_RE.search(txt) else 1
        avg_c = sum(confs[txt]) / len(confs[txt])
        return (counts[txt], clean_bonus, round(avg_c, 3), len(txt))

    winner = max(counts.keys(), key=rank_key)

    # Prefer a clean fuller containment candidate when strongly supported (e.g. '一个...' over '个...')
    for cand, cnt in counts.items():
        if cand == winner or _EDGE_LATIN_NOISE_RE.search(cand):
            continue
        if len(cand) > len(winner) and (cand.startswith(winner) or cand.endswith(winner)):
            added = cand[: len(cand) - len(winner)] if cand.endswith(winner) else cand[len(winner) :]
            if (len(cand) - len(winner)) <= 2 and _HANZI_RE.search(added) and max(confs[cand]) >= 0.90:
                if cnt >= max(1, counts[winner] // 2):
                    winner = cand
                    break

    win_conf, win_bgr = best_frame_for_text[winner]

    # Selective brightness-filtered retry only when winner is still low-confidence or has edge Latin noise
    if win_conf < config.conf_threshold or _EDGE_LATIN_NOISE_RE.search(winner):
        retry_raw, retry_conf = ocr_engine.recognize_roi(win_bgr, brightness_filter=True)
        retry_text = normalize_cjk_text(retry_raw)
        if retry_text and _HANZI_RE.search(retry_text):
            winner_noisy = bool(_EDGE_LATIN_NOISE_RE.search(winner))
            retry_noisy = bool(_EDGE_LATIN_NOISE_RE.search(retry_text))
            cleaned_edge = winner_noisy and not retry_noisy and retry_conf >= max(0.80, win_conf - 0.05)
            higher_conf = (not retry_noisy) and (retry_conf >= win_conf + 0.05) and (
                len(retry_text) >= max(1, int(0.70 * len(winner)))
            )
            if cleaned_edge or higher_conf:
                winner = retry_text
                win_conf = retry_conf

    return (winner, win_conf, win_bgr)


def detect_subtitle_spans(
    frames: Iterable[FrameSlice],
    fps: float,
    start_pts_ms: int,
    ocr_engine: OcrEngine,
    config: ExtractConfig,
    *,
    scan_progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[list[_DetectedSpan], int]:
    """Stage 1: Scan video ROI stream via ch_PP-OCRv5 Det Mobile Grid into candidate spans."""
    safe_fps = max(1.0, float(fps))
    sample_step = max(1, int(round(safe_fps / 7.5)))
    if config.coarse_step == 1:
        sample_step = 1
    step_ms = max(1, int(round(1000.0 * sample_step / safe_fps)))

    spans: list[_DetectedSpan] = []
    active: _DetectedSpan | None = None
    batch_slices: list[tuple[int, np.ndarray]] = []

    def process_batch() -> None:
        nonlocal active
        if not batch_slices:
            return
        rois = [b[1] for b in batch_slices]
        batch_boxes = ocr_engine.detect_grid_batch(rois)
        for (rel_ms, bgr_roi), boxes in zip(batch_slices, batch_boxes):
            if not boxes:
                if active is not None and (rel_ms - active.end_ms) > config.flash_bridge_max_ms:
                    spans.append(active)
                    active = None
                continue

            u_rect = _merge_boxes_to_union_rect(boxes)

            if active is not None:
                gap_ms = rel_ms - active.end_ms
                same_layout = gap_ms <= config.flash_bridge_max_ms and _rects_similar(
                    active.union_rect, u_rect
                )
                if same_layout:
                    active.end_ms = rel_ms + step_ms
                    # Smooth rect with EMA instead of monotonic min/max union growth
                    alpha = 0.35
                    r_prev = active.union_rect
                    active.union_rect = (
                        (1.0 - alpha) * r_prev[0] + alpha * u_rect[0],
                        (1.0 - alpha) * r_prev[1] + alpha * u_rect[1],
                        (1.0 - alpha) * r_prev[2] + alpha * u_rect[2],
                        (1.0 - alpha) * r_prev[3] + alpha * u_rect[3],
                    )
                    active.add_sample(rel_ms, bgr_roi)
                    continue

                spans.append(active)
                active = None

            active = _DetectedSpan(
                start_ms=rel_ms,
                end_ms=rel_ms + step_ms,
                union_rect=u_rect,
            )
            active.add_sample(rel_ms, bgr_roi)

        batch_slices.clear()

    for frame_slice in frames:
        if cancel_check is not None and cancel_check():
            raise ExtractionCancelled("Deep subtitle scan cancelled")

        rel_ms = max(0, frame_slice.pts_ms - start_pts_ms)
        if scan_progress_cb is not None:
            scan_progress_cb(frame_slice.frame_index, frame_slice.pts_ms)

        if frame_slice.frame_index % sample_step != 0:
            continue

        batch_slices.append((rel_ms, frame_slice.bgr_roi))
        if len(batch_slices) >= _DET_BATCH_SIZE:
            process_batch()

    if cancel_check is not None and cancel_check():
        raise ExtractionCancelled("Deep subtitle scan cancelled")
    process_batch()
    if active is not None:
        spans.append(active)

    neighbor_gap = max(config.merge_gap_ms, config.flash_bridge_max_ms)
    eligible: list[_DetectedSpan] = []
    for idx, sp in enumerate(spans):
        dur = sp.end_ms - sp.start_ms
        if not sp.samples:
            continue
        if dur >= config.min_duration_ms:
            eligible.append(sp)
            continue
        near_prev = idx > 0 and (sp.start_ms - spans[idx - 1].end_ms) <= neighbor_gap
        near_next = idx + 1 < len(spans) and (spans[idx + 1].start_ms - sp.end_ms) <= neighbor_gap
        if near_prev or near_next:
            eligible.append(sp)

    return eligible, step_ms


def recognize_spans_with_voting(
    eligible: list[_DetectedSpan],
    step_ms: int,
    ocr_engine: OcrEngine,
    config: ExtractConfig,
    *,
    ocr_progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[OcrSegment]:
    """Stage 2: Run ch_PP-OCRv5 Server/Mobile on representative span frames with Majority Voting."""
    total_spans = len(eligible)
    if total_spans == 0:
        return []

    clusters: list[_VoteCluster] = []

    for sp_idx, span in enumerate(eligible):
        if cancel_check is not None and cancel_check():
            raise ExtractionCancelled("Deep OCR recognition cancelled")

        samples = span.samples
        if len(samples) > _MAX_OCR_VOTES_PER_SPAN:
            idxs = np.linspace(0, len(samples) - 1, num=_MAX_OCR_VOTES_PER_SPAN)
            chosen_samples = [samples[int(round(i))] for i in dict.fromkeys(int(round(x)) for x in idxs)]
        else:
            chosen_samples = samples

        span_votes: list[_FrameVote] = []
        for s in chosen_samples:
            if cancel_check is not None and cancel_check():
                raise ExtractionCancelled("Deep OCR recognition cancelled")
            bgr = _decode_sample(s)
            if bgr is None:
                continue
            raw_text, conf = ocr_engine.recognize_roi(bgr)
            norm = normalize_cjk_text(raw_text)
            if norm:
                span_votes.append(_FrameVote(pts_ms=s.pts_ms, text=norm, conf=conf, bgr=bgr))

        if span_votes:
            sub_groups: list[list[_FrameVote]] = [[span_votes[0]]]
            for v in span_votes[1:]:
                if _is_same_cue_text(sub_groups[-1][-1].text, v.text):
                    sub_groups[-1].append(v)
                else:
                    sub_groups.append([v])

            for g_idx, grp in enumerate(sub_groups):
                g_start = span.start_ms if g_idx == 0 else grp[0].pts_ms
                g_end = span.end_ms if g_idx == len(sub_groups) - 1 else max(
                    g_start + step_ms, sub_groups[g_idx + 1][0].pts_ms
                )
                rep_text = grp[0].text
                if clusters:
                    prev_cluster = clusters[-1]
                    gap_ms = g_start - prev_cluster.end_ms
                    prev_rep = prev_cluster.votes[-1].text
                    if gap_ms <= config.flash_bridge_max_ms and _is_same_cue_text(prev_rep, rep_text):
                        prev_cluster.end_ms = max(prev_cluster.end_ms, g_end)
                        prev_cluster.votes.extend(grp)
                        continue
                clusters.append(_VoteCluster(start_ms=g_start, end_ms=g_end, votes=list(grp)))

        span.samples.clear()
        if ocr_progress_cb is not None:
            ocr_progress_cb(sp_idx + 1, total_spans)

    ocr_segments: list[OcrSegment] = []
    for cl in clusters:
        win_text, win_conf, win_bgr = _choose_majority_vote(cl.votes, ocr_engine, config)
        dur = cl.end_ms - cl.start_ms
        if len(cl.votes) == 1 and dur < max(200, config.min_duration_ms) and win_conf < 0.88:
            continue
        thumb_png = ocr_engine.encode_thumbnail_png(win_bgr)
        ocr_segments.append(
            OcrSegment(
                start_ms=cl.start_ms,
                end_ms=max(cl.start_ms + 1, cl.end_ms),
                text=win_text,
                confidence=win_conf,
                thumbnail_png=thumb_png,
            )
        )

    return ocr_segments


def extract_subtitles_deep(
    frames: Iterable[FrameSlice],
    fps: float,
    start_pts_ms: int,
    ocr_engine: OcrEngine,
    config: ExtractConfig,
    *,
    scan_progress_cb: Callable[[int, int], None] | None = None,
    ocr_progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[OcrSegment]:
    """Convenience wrapper executing Stage 1 detection followed by Stage 2 Majority Voting."""
    spans, step_ms = detect_subtitle_spans(
        frames,
        fps,
        start_pts_ms,
        ocr_engine,
        config,
        scan_progress_cb=scan_progress_cb,
        cancel_check=cancel_check,
    )
    return recognize_spans_with_voting(
        spans,
        step_ms,
        ocr_engine,
        config,
        ocr_progress_cb=ocr_progress_cb,
        cancel_check=cancel_check,
    )
