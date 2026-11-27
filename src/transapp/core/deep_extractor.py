"""2-Stage Deep Learning hardsub extractor (4D Det Mobile + Ring-Buffer Refinement + Server Voting)."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import math
import re
import time

import cv2
import numpy as np

from transapp.core.models import ExtractConfig, ExtractionCancelled, OcrSegment
from transapp.core.ocr_engine import OcrEngine
from transapp.core.postprocess import normalize_cjk_text, text_similarity
from transapp.core.video_reader import FrameSlice

_MAX_SPAN_SAMPLES = 24
_MAX_OCR_VOTES_PER_SPAN = 5
_DET_BATCH_SIZE = 8
_RECT_SIMILARITY_TOL = 0.08
_SIG_W = 64
_SIG_H = 16
_PROGRESS_INTERVAL_SEC = 0.08
_PROGRESS_UNTHROTTLED_FRAMES = 16
_HANZI_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_EDGE_LATIN_NOISE_RE = re.compile(
    r"(?:^[A-Za-z0-9]{1,3}[\u4e00-\u9fff]|[\u4e00-\u9fff][A-Za-z0-9]{1,3}$)"
)


@dataclass(slots=True)
class TightSignature:
    """Visual & geometric signature computed strictly inside Det Mobile's union text box."""

    boxes: list[tuple[float, float, float, float]]
    union_rect: tuple[float, float, float, float]
    edge_mask: np.ndarray
    luma_patch: np.ndarray


@dataclass(slots=True)
class _CompressedSample:
    frame_idx: int
    pts_ms: int
    jpeg_bytes: bytes


@dataclass(slots=True)
class _DetectedSpan:
    start_idx: int
    end_idx: int
    start_ms: int
    end_ms: int
    union_rect: tuple[float, float, float, float]
    samples: list[_CompressedSample] = field(default_factory=list)
    total_seen: int = 0
    hit_count: int = 0
    last_sig: TightSignature | None = None

    def add_sample(self, frame_idx: int, pts_ms: int, bgr_roi: np.ndarray) -> None:
        if self.samples and self.samples[-1].frame_idx == frame_idx:
            return
        self.hit_count += 1
        self.total_seen += 1
        ok, buf = cv2.imencode(".jpg", bgr_roi, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            return
        new_sample = _CompressedSample(frame_idx=frame_idx, pts_ms=pts_ms, jpeg_bytes=buf.tobytes())
        if len(self.samples) < _MAX_SPAN_SAMPLES:
            self.samples.append(new_sample)
            return

        # Keep first sample (samples[0]), latest sample (new_sample), and drop the most
        # redundant interior slot whose removal produces the smallest adjacent frame gap.
        candidates = self.samples + [new_sample]
        best_drop = 1
        best_span = candidates[2].frame_idx - candidates[0].frame_idx
        for k in range(2, len(candidates) - 1):
            gap = candidates[k + 1].frame_idx - candidates[k - 1].frame_idx
            if gap < best_span:
                best_span = gap
                best_drop = k
        del candidates[best_drop]
        self.samples = candidates


@dataclass(slots=True)
class _FrameVote:
    pts_ms: int
    text: str
    conf: float
    bgr: np.ndarray
    frame_idx: int = -1


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


def _rect_geometry_metrics(
    r1: tuple[float, float, float, float],
    r2: tuple[float, float, float, float],
) -> tuple[float, float]:
    x_inter = max(0.0, min(r1[2], r2[2]) - max(r1[0], r2[0]))
    x_union = max(1e-3, max(r1[2], r2[2]) - min(r1[0], r2[0]))
    x_iou = x_inter / x_union

    y_inter = max(0.0, min(r1[3], r2[3]) - max(r1[1], r2[1]))
    min_h = max(1e-3, min(r1[3] - r1[1], r2[3] - r2[1]))
    y_overlap = min(1.0, y_inter / min_h)
    return (x_iou, y_overlap)


def _rects_similar(
    r1: tuple[float, float, float, float],
    r2: tuple[float, float, float, float],
    tol: float = _RECT_SIMILARITY_TOL,
) -> bool:
    """Compare horizontal span stability + vertical overlap (tolerant of DBNet vertical jitter)."""
    x_iou, y_overlap = _rect_geometry_metrics(r1, r2)
    if y_overlap < 0.45:
        return False

    if x_iou >= 0.78:
        return True

    w1 = max(1e-3, r1[2] - r1[0])
    w2 = max(1e-3, r2[2] - r2[0])
    cx1 = 0.5 * (r1[0] + r1[2])
    cx2 = 0.5 * (r2[0] + r2[2])
    max_w = max(w1, w2)
    return abs(w1 - w2) <= max(0.05, tol * 2.0 * max_w) and abs(cx1 - cx2) <= tol * 1.25


def _compute_tight_signature(
    bgr_roi: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
) -> TightSignature:
    """Compute edge-stroke mask and normalized luma patch strictly inside the union text box."""
    union_rect = _merge_boxes_to_union_rect(boxes)
    if bgr_roi is None or bgr_roi.size == 0 or bgr_roi.ndim < 2:
        return TightSignature(
            boxes=list(boxes),
            union_rect=union_rect,
            edge_mask=np.zeros((_SIG_H, _SIG_W), dtype=np.uint8),
            luma_patch=np.zeros((_SIG_H, _SIG_W), dtype=np.float32),
        )

    h, w = bgr_roi.shape[:2]
    nx1, ny1, nx2, ny2 = union_rect
    x1 = max(0, min(w - 1, int(math.floor(nx1 * w)) - 2))
    y1 = max(0, min(h - 1, int(math.floor(ny1 * h)) - 2))
    x2 = max(x1 + 1, min(w, int(math.ceil(nx2 * w)) + 2))
    y2 = max(y1 + 1, min(h, int(math.ceil(ny2 * h)) + 2))

    crop = bgr_roi[y1:y2, x1:x2]
    if crop.ndim == 3 and crop.shape[2] >= 3:
        gray = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        gray = crop if crop.ndim == 2 else crop[:, :, 0]

    # Compute strokes directly on the compact 64x16 grid so internal character holes stay crisp
    # instead of blurring into solid square blobs across distinct equal-width CJK sentences.
    small = cv2.resize(gray, (_SIG_W, _SIG_H), interpolation=cv2.INTER_AREA).astype(np.float32)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    # Local bright-stroke signal suppresses smooth/moving background fills and abrupt
    # background cuts while preserving white, yellow, colored, and light-outlined strokes.
    local_mean = cv2.GaussianBlur(small, (5, 5), 0)
    bright_stroke = np.maximum(0.0, small - local_mean)
    sig_map = bright_stroke * 2.5 + grad_mag * 0.25

    max_val = float(sig_map.max())
    if max_val < 8.0:
        edge_mask = np.zeros((_SIG_H, _SIG_W), dtype=np.uint8)
    else:
        thresh = max(10.0, float(np.percentile(sig_map, 72)), 0.30 * max_val)
        edge_mask = (sig_map >= thresh).astype(np.uint8)

    # High-pass zero-mean luma patch removes low-frequency background brightness jumps
    # and moving shapes while retaining exact signed glyph-fill (+) and outline (-) layout.
    luma = small - cv2.GaussianBlur(small, (7, 7), 0)
    luma -= float(luma.mean())
    luma_norm = float(np.linalg.norm(luma))
    if luma_norm > 1e-3:
        luma /= luma_norm
    else:
        luma.fill(0.0)

    return TightSignature(
        boxes=list(boxes),
        union_rect=union_rect,
        edge_mask=edge_mask,
        luma_patch=luma,
    )


def _compare_tight_visual(
    sig_a: TightSignature,
    sig_b: TightSignature,
) -> tuple[float, float, bool, bool]:
    """Compare edge_mask IoU and luma_patch NCC at the best shared alignment shift on 64x16."""
    ea = sig_a.edge_mask > 0
    eb = sig_b.edge_mask > 0
    la = sig_a.luma_patch
    lb = sig_b.luma_patch

    a_flat = (not bool(ea.any())) and float(np.linalg.norm(la)) <= 1e-3
    b_flat = (not bool(eb.any())) and float(np.linalg.norm(lb)) <= 1e-3
    if a_flat and b_flat:
        return (1.0, 1.0, True, True)
    if a_flat != b_flat:
        return (0.0, 0.0, a_flat, b_flat)

    best_score = -2.0
    best_edge_iou = 0.0
    best_luma_ncc = -1.0
    h, w = ea.shape

    for dy in (-1, 0, 1):
        ya0 = max(0, dy)
        ya1 = min(h, h + dy)
        yb0 = max(0, -dy)
        yb1 = min(h, h - dy)
        for dx in (-2, -1, 0, 1, 2):
            xa0 = max(0, dx)
            xa1 = min(w, w + dx)
            xb0 = max(0, -dx)
            xb1 = min(w, w - dx)

            sub_ea = ea[ya0:ya1, xa0:xa1]
            sub_eb = eb[yb0:yb1, xb0:xb1]
            union = int(np.count_nonzero(sub_ea | sub_eb))
            iou = (int(np.count_nonzero(sub_ea & sub_eb)) / float(union)) if union > 0 else 0.0

            sub_la = la[ya0:ya1, xa0:xa1]
            sub_lb = lb[yb0:yb1, xb0:xb1]
            denom = float(np.linalg.norm(sub_la) * np.linalg.norm(sub_lb))
            ncc = (float(np.sum(sub_la * sub_lb)) / denom) if denom > 1e-6 else 0.0
            ncc = max(-1.0, min(1.0, ncc))

            combined = iou + max(0.0, ncc)
            if combined > best_score:
                best_score = combined
                best_edge_iou = iou
                best_luma_ncc = ncc

    return (best_edge_iou, best_luma_ncc, False, False)


def _tight_signature_score(sig_a: TightSignature, sig_b: TightSignature) -> float:
    """Return a [0.0, 1.0] visual+geometric similarity score between two tight signatures."""
    x_iou, y_overlap = _rect_geometry_metrics(sig_a.union_rect, sig_b.union_rect)
    geom_sim = 0.5 * x_iou + 0.5 * y_overlap
    if y_overlap <= 0.0 or x_iou <= 0.0:
        return 0.0

    edge_iou, luma_ncc, a_flat, b_flat = _compare_tight_visual(sig_a, sig_b)
    if a_flat and b_flat:
        return geom_sim if _rects_similar(sig_a.union_rect, sig_b.union_rect) else 0.5 * geom_sim
    if a_flat != b_flat:
        return 0.0

    luma_sim = max(0.0, luma_ncc)
    visual_score = 0.55 * edge_iou + 0.45 * luma_sim
    if not _rects_similar(sig_a.union_rect, sig_b.union_rect):
        return visual_score * geom_sim
    return 0.20 * geom_sim + 0.80 * visual_score


def _tight_signatures_same_cue(sig_a: TightSignature, sig_b: TightSignature) -> bool:
    """Return True only when both tight box geometry AND inner text strokes match."""
    if not _rects_similar(sig_a.union_rect, sig_b.union_rect):
        return False
    edge_iou, luma_ncc, a_flat, b_flat = _compare_tight_visual(sig_a, sig_b)
    if a_flat and b_flat:
        return True
    if a_flat != b_flat:
        return False
    return (edge_iou >= 0.45 and luma_ncc >= 0.55) or (edge_iou >= 0.55) or (luma_ncc >= 0.72)


def _compute_median_frame_ms(frame_pts_ms: list[int], fallback_fps: float = 25.0) -> int:
    deltas = [
        frame_pts_ms[i + 1] - frame_pts_ms[i]
        for i in range(len(frame_pts_ms) - 1)
        if frame_pts_ms[i + 1] > frame_pts_ms[i]
    ]
    if deltas:
        return max(1, int(round(float(np.median(deltas)))))
    return max(1, int(round(1000.0 / max(1.0, float(fallback_fps)))))


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


def _open_span(
    start_idx: int,
    curr_idx: int,
    curr_rel_ms: int,
    curr_bgr: np.ndarray,
    curr_sig: TightSignature,
    onset_frames: list[tuple[int, int, np.ndarray]],
    frame_pts_ms: list[int],
) -> _DetectedSpan:
    span = _DetectedSpan(
        start_idx=start_idx,
        end_idx=curr_idx,
        start_ms=frame_pts_ms[start_idx],
        end_ms=curr_rel_ms,
        union_rect=curr_sig.union_rect,
        last_sig=curr_sig,
    )
    for f_idx, f_rel_ms, f_bgr in onset_frames:
        span.add_sample(f_idx, f_rel_ms, f_bgr)
    span.add_sample(curr_idx, curr_rel_ms, curr_bgr)
    return span


def detect_subtitle_spans(
    frames: Iterable[FrameSlice],
    fps: float,
    start_pts_ms: int,
    ocr_engine: OcrEngine,
    config: ExtractConfig,
    *,
    scan_progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[list[_DetectedSpan], list[int]]:
    """Stage 1: Single-pass streaming 4D Det Mobile + ring-buffer exact frame refinement."""
    safe_fps = max(1.0, float(fps))
    if config.coarse_step == 1:
        sample_step = 1
    elif config.coarse_step > 2:
        sample_step = int(config.coarse_step)
    elif safe_fps <= 30.5:
        sample_step = 2
    else:
        sample_step = max(2, int(round(safe_fps / 12.5)))

    det_batch_size = max(1, int(getattr(ocr_engine, "det_batch_size", _DET_BATCH_SIZE)))

    spans: list[_DetectedSpan] = []
    frame_pts_ms: list[int] = []
    frame_buf: dict[int, tuple[int, np.ndarray]] = {}
    batch_coarse_idxs: list[int] = []

    active: _DetectedSpan | None = None
    prev_coarse_idx: int | None = None
    last_queued_coarse_idx = -1

    last_progress_ts = 0.0
    last_emitted_idx = -1
    last_frame_index = -1
    last_raw_pts_ms = 0

    def process_batch() -> None:
        nonlocal active, prev_coarse_idx
        if not batch_coarse_idxs:
            return

        rois = [frame_buf[idx][1] for idx in batch_coarse_idxs]
        batch_boxes = ocr_engine.detect_grid_batch(rois)

        for curr_idx, boxes in zip(batch_coarse_idxs, batch_boxes):
            curr_rel_ms, curr_bgr = frame_buf[curr_idx]
            curr_sig = _compute_tight_signature(curr_bgr, boxes) if boxes else None
            prev_idx = prev_coarse_idx

            if active is None:
                if curr_sig is not None:
                    # EMPTY -> TEXT onset refinement
                    start_idx = curr_idx
                    onset_sig = curr_sig
                    onset_frames_rev: list[tuple[int, int, np.ndarray]] = []
                    if prev_idx is not None and curr_idx - prev_idx > 1:
                        mid_idxs = list(range(prev_idx + 1, curr_idx))
                        mid_boxes_list = ocr_engine.detect_grid_batch(
                            [frame_buf[i][1] for i in mid_idxs]
                        )
                        for i, m_boxes in reversed(list(zip(mid_idxs, mid_boxes_list))):
                            if not m_boxes:
                                break
                            m_rel_ms, m_bgr = frame_buf[i]
                            m_sig = _compute_tight_signature(m_bgr, m_boxes)
                            if _tight_signatures_same_cue(m_sig, onset_sig):
                                start_idx = i
                                onset_sig = m_sig
                                onset_frames_rev.append((i, m_rel_ms, m_bgr))
                            else:
                                break
                    onset_frames_rev.reverse()
                    active = _open_span(
                        start_idx,
                        curr_idx,
                        curr_rel_ms,
                        curr_bgr,
                        curr_sig,
                        onset_frames_rev,
                        frame_pts_ms,
                    )
            else:
                if curr_sig is None:
                    # TEXT -> EMPTY offset refinement (no Stage 1 blind flash gluing!)
                    if prev_idx is not None and curr_idx - prev_idx > 1:
                        mid_idxs = list(range(prev_idx + 1, curr_idx))
                        mid_boxes_list = ocr_engine.detect_grid_batch(
                            [frame_buf[i][1] for i in mid_idxs]
                        )
                        for i, m_boxes in zip(mid_idxs, mid_boxes_list):
                            if not m_boxes:
                                break
                            m_rel_ms, m_bgr = frame_buf[i]
                            m_sig = _compute_tight_signature(m_bgr, m_boxes)
                            if active.last_sig is None or _tight_signatures_same_cue(
                                active.last_sig, m_sig
                            ):
                                active.end_idx = i
                                active.end_ms = m_rel_ms
                                active.last_sig = m_sig
                                active.add_sample(i, m_rel_ms, m_bgr)
                            else:
                                break
                    spans.append(active)
                    active = None
                elif active.last_sig is None or _tight_signatures_same_cue(
                    active.last_sig, curr_sig
                ):
                    # TEXT_A -> TEXT_A continuation
                    active.end_idx = curr_idx
                    active.end_ms = curr_rel_ms
                    active.last_sig = curr_sig
                    alpha = 0.35
                    r_prev = active.union_rect
                    u_rect = curr_sig.union_rect
                    active.union_rect = (
                        (1.0 - alpha) * r_prev[0] + alpha * u_rect[0],
                        (1.0 - alpha) * r_prev[1] + alpha * u_rect[1],
                        (1.0 - alpha) * r_prev[2] + alpha * u_rect[2],
                        (1.0 - alpha) * r_prev[3] + alpha * u_rect[3],
                    )
                    active.add_sample(curr_idx, curr_rel_ms, curr_bgr)
                else:
                    # TEXT_A -> TEXT_B (0ms back-to-back cut or short gap + new cue)
                    sig_a = active.last_sig
                    sig_b = curr_sig
                    b_start_idx = curr_idx
                    b_onset_frames: list[tuple[int, int, np.ndarray]] = []

                    if prev_idx is not None and curr_idx - prev_idx > 1:
                        mid_idxs = list(range(prev_idx + 1, curr_idx))
                        mid_boxes_list = ocr_engine.detect_grid_batch(
                            [frame_buf[i][1] for i in mid_idxs]
                        )
                        classified: list[tuple[int, str, TightSignature | None]] = []
                        for i, m_boxes in zip(mid_idxs, mid_boxes_list):
                            if not m_boxes:
                                classified.append((i, "EMPTY", None))
                                continue
                            m_sig = _compute_tight_signature(frame_buf[i][1], m_boxes)
                            score_a = _tight_signature_score(m_sig, sig_a)
                            score_b = _tight_signature_score(m_sig, sig_b)
                            label = "A" if score_a >= score_b else "B"
                            classified.append((i, label, m_sig))

                        a_end_pos = 0
                        for pos, (i, label, m_sig) in enumerate(classified):
                            if label != "A":
                                break
                            m_rel_ms, m_bgr = frame_buf[i]
                            active.end_idx = i
                            active.end_ms = m_rel_ms
                            active.last_sig = m_sig
                            active.add_sample(i, m_rel_ms, m_bgr)
                            a_end_pos = pos + 1

                        b_onset_rev: list[tuple[int, int, np.ndarray]] = []
                        for pos in range(len(classified) - 1, a_end_pos - 1, -1):
                            i, label, _m_sig = classified[pos]
                            if label != "B":
                                break
                            m_rel_ms, m_bgr = frame_buf[i]
                            b_start_idx = i
                            b_onset_rev.append((i, m_rel_ms, m_bgr))
                        b_onset_rev.reverse()
                        b_onset_frames = b_onset_rev

                    spans.append(active)
                    active = _open_span(
                        b_start_idx,
                        curr_idx,
                        curr_rel_ms,
                        curr_bgr,
                        curr_sig,
                        b_onset_frames,
                        frame_pts_ms,
                    )

            prev_coarse_idx = curr_idx

        batch_coarse_idxs.clear()
        if prev_coarse_idx is not None:
            stale = [k for k in frame_buf if k < prev_coarse_idx]
            for k in stale:
                del frame_buf[k]

    for frame_slice in frames:
        if cancel_check is not None and cancel_check():
            raise ExtractionCancelled("Deep subtitle scan cancelled")

        idx = len(frame_pts_ms)
        rel_ms = max(0, frame_slice.pts_ms - start_pts_ms)
        frame_pts_ms.append(rel_ms)
        frame_buf[idx] = (rel_ms, frame_slice.bgr_roi)
        last_frame_index = frame_slice.frame_index
        last_raw_pts_ms = frame_slice.pts_ms

        if scan_progress_cb is not None:
            now = time.monotonic()
            if idx < _PROGRESS_UNTHROTTLED_FRAMES or (now - last_progress_ts) >= _PROGRESS_INTERVAL_SEC:
                scan_progress_cb(frame_slice.frame_index, frame_slice.pts_ms)
                last_progress_ts = now
                last_emitted_idx = idx

        if idx % sample_step == 0:
            batch_coarse_idxs.append(idx)
            last_queued_coarse_idx = idx
            if len(batch_coarse_idxs) >= det_batch_size:
                process_batch()

    if cancel_check is not None and cancel_check():
        raise ExtractionCancelled("Deep subtitle scan cancelled")

    if frame_pts_ms:
        last_idx = len(frame_pts_ms) - 1
        if scan_progress_cb is not None and last_emitted_idx != last_idx:
            scan_progress_cb(last_frame_index, last_raw_pts_ms)
        if last_idx > last_queued_coarse_idx:
            batch_coarse_idxs.append(last_idx)

    process_batch()
    frame_buf.clear()

    if active is not None:
        spans.append(active)

    if not frame_pts_ms:
        return ([], [])

    median_frame_ms = _compute_median_frame_ms(frame_pts_ms, safe_fps)
    total_frames = len(frame_pts_ms)
    for sp in spans:
        sp.start_ms = frame_pts_ms[sp.start_idx]
        if sp.end_idx + 1 < total_frames:
            sp.end_ms = max(sp.start_ms + 1, frame_pts_ms[sp.end_idx + 1])
        else:
            sp.end_ms = max(sp.start_ms + 1, frame_pts_ms[sp.end_idx] + median_frame_ms)
        sp.last_sig = None

    neighbor_gap = max(config.merge_gap_ms, config.flash_bridge_max_ms)
    eligible: list[_DetectedSpan] = []
    for idx, sp in enumerate(spans):
        dur = sp.end_ms - sp.start_ms
        if not sp.samples:
            continue
        if dur >= config.min_duration_ms:
            eligible.append(sp)
            continue
        # Drop isolated 1-frame artifacts (e.g. 40ms flash) below min_duration_ms
        if sp.start_idx == sp.end_idx:
            continue
        near_prev = idx > 0 and (sp.start_ms - spans[idx - 1].end_ms) <= neighbor_gap
        near_next = idx + 1 < len(spans) and (spans[idx + 1].start_ms - sp.end_ms) <= neighbor_gap
        if near_prev or near_next:
            eligible.append(sp)

    return (eligible, frame_pts_ms)


def recognize_spans_with_voting(
    eligible: list[_DetectedSpan],
    frame_pts_ms: list[int],
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

    total_frames = len(frame_pts_ms)
    clusters: list[_VoteCluster] = []

    for sp_idx, span in enumerate(eligible):
        if cancel_check is not None and cancel_check():
            raise ExtractionCancelled("Deep OCR recognition cancelled")

        samples = span.samples
        if len(samples) > _MAX_OCR_VOTES_PER_SPAN:
            idxs = np.linspace(0, len(samples) - 1, num=_MAX_OCR_VOTES_PER_SPAN)
            chosen_samples = [
                samples[int(round(i))] for i in dict.fromkeys(int(round(x)) for x in idxs)
            ]
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
                span_votes.append(
                    _FrameVote(
                        pts_ms=s.pts_ms,
                        text=norm,
                        conf=conf,
                        bgr=bgr,
                        frame_idx=s.frame_idx,
                    )
                )

        if span_votes:
            sub_groups: list[list[_FrameVote]] = [[span_votes[0]]]
            for v in span_votes[1:]:
                if _is_same_cue_text(sub_groups[-1][-1].text, v.text):
                    sub_groups[-1].append(v)
                else:
                    sub_groups.append([v])

            for g_idx, grp in enumerate(sub_groups):
                if g_idx == 0:
                    g_start = span.start_ms
                elif 0 <= grp[0].frame_idx < total_frames:
                    g_start = frame_pts_ms[grp[0].frame_idx]
                else:
                    g_start = grp[0].pts_ms

                if g_idx == len(sub_groups) - 1:
                    g_end = span.end_ms
                else:
                    next_first = sub_groups[g_idx + 1][0]
                    if 0 <= next_first.frame_idx < total_frames:
                        next_start_ms = frame_pts_ms[next_first.frame_idx]
                    else:
                        next_start_ms = next_first.pts_ms
                    g_end = max(g_start + 1, next_start_ms)

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
        span.last_sig = None
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
    spans, frame_pts_ms = detect_subtitle_spans(
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
        frame_pts_ms,
        ocr_engine,
        config,
        ocr_progress_cb=ocr_progress_cb,
        cancel_check=cancel_check,
    )
