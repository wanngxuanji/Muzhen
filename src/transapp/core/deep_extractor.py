"""2-Stage Deep Learning hardsub extractor (4D Det Mobile + Ring-Buffer Refinement + Server Voting)."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import difflib
import math
import re
import time

import cv2
import numpy as np

from transapp.core.models import ExtractConfig, ExtractionCancelled, OcrSegment
from transapp.core.ocr_engine import OcrEngine
from transapp.core.postprocess import (
    _has_suspicious_noise,
    _strip_for_similarity,
    normalize_cjk_text,
    text_similarity,
)
from transapp.core.video_reader import FrameSlice

_MAX_SPAN_SAMPLES = 24
_DET_BATCH_SIZE = 8
_RECT_SIMILARITY_TOL = 0.08
_SIG_W = 64
_SIG_H = 16
_ROI_GRAY_W = 640
_ROI_GRAY_H = 64
_PROGRESS_INTERVAL_SEC = 0.08
_PROGRESS_UNTHROTTLED_FRAMES = 16
_HANZI_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_EDGE_NOISE_RE = re.compile(
    r"(?:^[A-Za-z0-9#|/\\_^~`]{1,3}[\u4e00-\u9fff]|[\u4e00-\u9fff][A-Za-z0-9#|/\\_^~`]{1,3}$)"
)
_EDGE_NOISE_PREFIX_RE = re.compile(r"^[A-Za-z0-9#|/\\_^~`]{1,3}(?=[\u4e00-\u9fff])")
_EDGE_NOISE_SUFFIX_RE = re.compile(r"(?<=[\u4e00-\u9fff])[A-Za-z0-9#|/\\_^~`]{1,3}$")
_EDGE_LATIN_NOISE_RE = _EDGE_NOISE_RE


@dataclass(slots=True)
class TightSignature:
    """Visual & geometric signature computed strictly inside Det Mobile's union text box."""

    boxes: list[tuple[float, float, float, float]]
    union_rect: tuple[float, float, float, float]
    edge_mask: np.ndarray
    luma_patch: np.ndarray
    roi_gray: np.ndarray | None = None


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
    return abs(w1 - w2) <= max(0.06, tol * 2.0 * max_w) and abs(cx1 - cx2) <= tol * 1.25


def _boundary_rects_compatible(
    r1: tuple[float, float, float, float],
    r2: tuple[float, float, float, float],
) -> bool:
    """Return True when two boxes spatially overlap in the same subtitle band."""
    x_inter = max(0.0, min(r1[2], r2[2]) - max(r1[0], r2[0]))
    if x_inter <= 0.0:
        return False
    y_inter = max(0.0, min(r1[3], r2[3]) - max(r1[1], r2[1]))
    min_h = max(1e-3, min(r1[3] - r1[1], r2[3] - r2[1]))
    y_overlap = min(1.0, y_inter / min_h)
    if y_overlap < 0.45:
        return False
    w1 = max(1e-3, r1[2] - r1[0])
    w2 = max(1e-3, r2[2] - r2[0])
    return 0.45 <= (w1 / w2) <= 2.2


def _extract_patch_features(
    gray_img: np.ndarray,
    rect: tuple[float, float, float, float],
    *,
    pad_px: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute stroke edge mask and high-pass zero-mean luma patch inside normalized rect."""
    h, w = gray_img.shape[:2]
    nx1, ny1, nx2, ny2 = rect
    x1 = max(0, min(w - 1, int(math.floor(nx1 * w)) - pad_px))
    y1 = max(0, min(h - 1, int(math.floor(ny1 * h)) - pad_px))
    x2 = max(x1 + 1, min(w, int(math.ceil(nx2 * w)) + pad_px))
    y2 = max(y1 + 1, min(h, int(math.ceil(ny2 * h)) + pad_px))

    crop = gray_img[y1:y2, x1:x2]
    interp = (
        cv2.INTER_LINEAR
        if crop.shape[1] < _SIG_W or crop.shape[0] < _SIG_H
        else cv2.INTER_AREA
    )
    # Compute strokes directly on the compact 64x16 grid so internal character holes stay crisp
    # instead of blurring into solid square blobs across distinct equal-width CJK sentences.
    small = cv2.resize(crop, (_SIG_W, _SIG_H), interpolation=interp).astype(np.float32)
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

    return (edge_mask, luma)


def _compute_tight_signature(
    bgr_roi: np.ndarray,
    boxes: list[tuple[float, float, float, float]],
) -> TightSignature:
    """Compute edge-stroke mask and normalized luma patch strictly inside the union text box."""
    if not boxes:
        return TightSignature(
            boxes=[],
            union_rect=(0.0, 0.0, 0.0, 0.0),
            edge_mask=np.zeros((_SIG_H, _SIG_W), dtype=np.uint8),
            luma_patch=np.zeros((_SIG_H, _SIG_W), dtype=np.float32),
            roi_gray=None,
        )
    union_rect = _merge_boxes_to_union_rect(boxes)
    if bgr_roi is None or bgr_roi.size == 0 or bgr_roi.ndim < 2:
        return TightSignature(
            boxes=list(boxes),
            union_rect=union_rect,
            edge_mask=np.zeros((_SIG_H, _SIG_W), dtype=np.uint8),
            luma_patch=np.zeros((_SIG_H, _SIG_W), dtype=np.float32),
            roi_gray=None,
        )

    if bgr_roi.ndim == 3 and bgr_roi.shape[2] >= 3:
        full_gray = cv2.cvtColor(bgr_roi[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        full_gray = bgr_roi if bgr_roi.ndim == 2 else bgr_roi[:, :, 0]

    if full_gray.shape[:2] == (_ROI_GRAY_H, _ROI_GRAY_W):
        roi_gray = full_gray.copy()
    else:
        interp = (
            cv2.INTER_LINEAR
            if full_gray.shape[1] < _ROI_GRAY_W or full_gray.shape[0] < _ROI_GRAY_H
            else cv2.INTER_AREA
        )
        roi_gray = cv2.resize(full_gray, (_ROI_GRAY_W, _ROI_GRAY_H), interpolation=interp)

    # _compare_tight_visual extracts shared-rect features directly from roi_gray when both
    # signatures retain roi_gray, so skip redundant full-res patch extraction here.
    return TightSignature(
        boxes=list(boxes),
        union_rect=union_rect,
        edge_mask=np.zeros((_SIG_H, _SIG_W), dtype=np.uint8),
        luma_patch=np.zeros((_SIG_H, _SIG_W), dtype=np.float32),
        roi_gray=roi_gray,
    )


def _compare_tight_visual(
    sig_a: TightSignature,
    sig_b: TightSignature,
) -> tuple[float, float, bool, bool]:
    """Compare edge_mask IoU and luma_patch NCC at the best shared alignment shift on 64x16.

    When both signatures retain compact ROI grayscale thumbnails, extract both patches from the
    exact same shared ROI coordinate rectangle so stationary hardsubs remain pixel-aligned even
    when DBNet's predicted bounding box jitters by +/-10-15px on moving backgrounds.
    """
    if sig_a.roi_gray is not None and sig_b.roi_gray is not None:
        ra, rb = sig_a.union_rect, sig_b.union_rect
        shared_rect = (
            min(ra[0], rb[0]),
            min(ra[1], rb[1]),
            max(ra[2], rb[2]),
            max(ra[3], rb[3]),
        )
        ea_u8, la = _extract_patch_features(sig_a.roi_gray, shared_rect, pad_px=1)
        eb_u8, lb = _extract_patch_features(sig_b.roi_gray, shared_rect, pad_px=1)
        ea = ea_u8 > 0
        eb = eb_u8 > 0
    else:
        if sig_a.roi_gray is not None:
            ea_u8, la = _extract_patch_features(sig_a.roi_gray, sig_a.union_rect, pad_px=1)
            ea = ea_u8 > 0
        else:
            ea = sig_a.edge_mask > 0
            la = sig_a.luma_patch
        if sig_b.roi_gray is not None:
            eb_u8, lb = _extract_patch_features(sig_b.roi_gray, sig_b.union_rect, pad_px=1)
            eb = eb_u8 > 0
        else:
            eb = sig_b.edge_mask > 0
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
    return (
        (edge_iou >= 0.45 and luma_ncc >= 0.55)
        or (edge_iou >= 0.55 and luma_ncc >= 0.30)
        or (luma_ncc >= 0.72 and edge_iou >= 0.30)
    )


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


def _has_edge_noise(text: str) -> bool:
    return bool(_EDGE_NOISE_RE.search(text))


def _strip_edge_noise(text: str) -> str:
    stripped = _EDGE_NOISE_PREFIX_RE.sub("", text)
    stripped = _EDGE_NOISE_SUFFIX_RE.sub("", stripped)
    return stripped


def _is_strict_same_cue(
    a: str,
    b: str,
    conf_a: float = 1.0,
    conf_b: float = 1.0,
) -> bool:
    """Strict cue equivalence that protects 1-char minimal pairs from blind merging."""
    if a == b:
        return True
    if text_similarity(a, b) == 1.0:
        return True

    ca = _strip_edge_noise(a)
    cb = _strip_edge_noise(b)
    sca = _strip_for_similarity(ca)
    scb = _strip_for_similarity(cb)
    if not sca or not scb:
        return False
    if sca == scb:
        return True

    # Strict prefix/suffix containment (diff <= 2 chars, len(short) >= 3)
    short, long_ = (sca, scb) if len(sca) <= len(scb) else (scb, sca)
    if len(short) >= 3 and 1 <= (len(long_) - len(short)) <= 2:
        if long_.startswith(short) or long_.endswith(short):
            return True

    # Protect same-length 1-char minimal pairs on >= 4 chars (e.g. "我不会放过你的" vs "我不会放过他的")
    is_same_len_one_char_sub = (
        len(sca) == len(scb) and sum(1 for x, y in zip(sca, scb) if x != y) == 1
    )
    if is_same_len_one_char_sub and len(sca) >= 4:
        diff_idx = next(i for i, (x, y) in enumerate(zip(sca, scb)) if x != y)
        both_hanzi = bool(_HANZI_RE.match(sca[diff_idx]) and _HANZI_RE.match(scb[diff_idx]))
        both_clean = not (
            _has_edge_noise(a)
            or _has_edge_noise(b)
            or _has_suspicious_noise(a)
            or _has_suspicious_noise(b)
        )
        if len(sca) < 10 or (both_hanzi and both_clean and min(conf_a, conf_b) >= 0.90):
            return False

    if min(len(sca), len(scb)) >= 10:
        return text_similarity(ca, cb) >= 0.88

    return False


def _score_vote_candidates(
    votes: list[_FrameVote],
) -> tuple[
    Counter[str],
    dict[str, list[float]],
    dict[str, _FrameVote],
    dict[str, float],
    Counter[str],
]:
    """Tally votes with edge-noise weight pooling onto matching clean candidates."""
    counts: Counter[str] = Counter()
    confs: dict[str, list[float]] = defaultdict(list)
    best_vote_for_text: dict[str, _FrameVote] = {}

    for v in votes:
        counts[v.text] += 1
        confs[v.text].append(v.conf)
        prev = best_vote_for_text.get(v.text)
        if prev is None or v.conf > prev.conf:
            best_vote_for_text[v.text] = v

    pooled_bonus: dict[str, float] = defaultdict(float)
    pooled_counts: Counter[str] = Counter(counts)
    for txt, cnt in counts.items():
        if not _has_edge_noise(txt):
            continue
        stripped = _strip_edge_noise(txt)
        if stripped and stripped != txt and stripped in counts and not _has_edge_noise(stripped):
            pooled_bonus[stripped] += 0.85 * sum(confs[txt])
            pooled_counts[stripped] += cnt

    return (counts, confs, best_vote_for_text, pooled_bonus, pooled_counts)


def _rank_candidate_key(
    txt: str,
    confs: dict[str, list[float]],
    pooled_bonus: dict[str, float],
    pooled_counts: Counter[str],
) -> tuple[float, int, int, float, int]:
    is_edge_noisy = _has_edge_noise(txt)
    total_w = (sum(confs[txt]) + pooled_bonus[txt]) * (0.72 if is_edge_noisy else 1.0)
    clean_bonus = 0 if (is_edge_noisy or _has_suspicious_noise(txt)) else 1
    avg_c = sum(confs[txt]) / len(confs[txt])
    return (round(total_w, 4), clean_bonus, pooled_counts[txt], round(avg_c, 3), len(txt))


def _group_representative_vote(votes: list[_FrameVote]) -> _FrameVote:
    """Select the cleanest majority representative vote of a group without running OCR retries."""
    counts, confs, best_vote_for_text, pooled_bonus, pooled_counts = _score_vote_candidates(votes)
    winner = max(
        counts.keys(),
        key=lambda t: _rank_candidate_key(t, confs, pooled_bonus, pooled_counts),
    )
    return best_vote_for_text[winner]


def _char_level_consensus(votes: list[_FrameVote]) -> tuple[str, _FrameVote] | None:
    """Reconstruct character-level consensus via SequenceMatcher alignment when no candidate repeats."""
    clean_votes = [
        v for v in votes if not _has_edge_noise(v.text) and not _has_suspicious_noise(v.text)
    ]
    pivot_pool = clean_votes if clean_votes else votes
    pivot_vote = max(pivot_pool, key=lambda v: (v.conf, len(v.text)))
    pivot = pivot_vote.text
    if not pivot:
        return None

    n_chars = len(pivot)
    pos_char_weights: list[dict[str, float]] = [defaultdict(float) for _ in range(n_chars)]
    ins_weights: list[dict[str, float]] = [defaultdict(float) for _ in range(n_chars + 1)]
    total_weight = 0.0

    for v in votes:
        is_noisy = _has_edge_noise(v.text) or _has_suspicious_noise(v.text)
        w = max(0.05, float(v.conf)) * (0.72 if is_noisy else 1.0)
        total_weight += w
        sm = difflib.SequenceMatcher(None, pivot, v.text, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    pos_char_weights[i1 + k][pivot[i1 + k]] += w
            elif tag == "replace":
                len_p = i2 - i1
                len_v = j2 - j1
                m = min(len_p, len_v)
                for k in range(m):
                    pos_char_weights[i1 + k][v.text[j1 + k]] += w
                if len_p > m:
                    for k in range(m, len_p):
                        pos_char_weights[i1 + k][""] += w
                elif len_v > m:
                    ins_weights[i2][v.text[j1 + m : j2]] += w
            elif tag == "delete":
                for i in range(i1, i2):
                    pos_char_weights[i][""] += w
            elif tag == "insert":
                ins_weights[i1][v.text[j1:j2]] += w

    if total_weight <= 0.0:
        return None

    out_chars: list[str] = []
    for g in range(n_chars + 1):
        if ins_weights[g]:
            best_ins, best_ins_w = max(ins_weights[g].items(), key=lambda kv: kv[1])
            if best_ins and best_ins_w >= 0.60 * total_weight:
                out_chars.append(best_ins)

        if g < n_chars:
            orig_ch = pivot[g]
            if pos_char_weights[g]:
                best_ch, best_w = max(pos_char_weights[g].items(), key=lambda kv: kv[1])
                if best_ch != orig_ch and best_w >= 0.60 * total_weight:
                    if best_ch:
                        out_chars.append(best_ch)
                else:
                    out_chars.append(orig_ch)
            else:
                out_chars.append(orig_ch)

    reconstructed = normalize_cjk_text("".join(out_chars))
    if not reconstructed:
        return None
    return (reconstructed, pivot_vote)


def _choose_majority_vote(
    votes: list[_FrameVote],
    ocr_engine: OcrEngine,
    config: ExtractConfig,
) -> tuple[str, float, np.ndarray]:
    counts, confs, best_vote_for_text, pooled_bonus, pooled_counts = _score_vote_candidates(votes)

    winner = max(
        counts.keys(),
        key=lambda t: _rank_candidate_key(t, confs, pooled_bonus, pooled_counts),
    )
    win_vote = best_vote_for_text[winner]

    # Character-level alignment consensus when >= 3 votes have no repeated/pooled candidate
    if len(votes) >= 3 and max(pooled_counts.values()) < 2:
        cons = _char_level_consensus(votes)
        if cons is not None:
            cons_text, pivot_vote = cons
            if cons_text and cons_text != winner:
                winner = cons_text
                win_vote = best_vote_for_text.get(winner, pivot_vote)

    # Prefer a clean fuller Hanzi containment candidate when strongly supported
    best_fuller: tuple[tuple[float, int, int], str] | None = None
    for cand, cnt in counts.items():
        if cand == winner or _has_edge_noise(cand) or _has_suspicious_noise(cand):
            continue
        if len(cand) > len(winner) and (cand.startswith(winner) or cand.endswith(winner)):
            added = cand[: len(cand) - len(winner)] if cand.endswith(winner) else cand[len(winner) :]
            if 1 <= len(added) <= 2 and all(_HANZI_RE.match(ch) for ch in added):
                max_c = max(confs[cand])
                if max_c >= 0.92 or (max_c >= 0.90 and cnt >= max(1, pooled_counts.get(winner, 1) // 2)):
                    cand_key = (max_c, cnt, len(cand))
                    if best_fuller is None or cand_key > best_fuller[0]:
                        best_fuller = (cand_key, cand)
    if best_fuller is not None:
        winner = best_fuller[1]
        win_vote = best_vote_for_text[winner]

    win_conf, win_bgr = win_vote.conf, win_vote.bgr

    def _needs_retry(txt: str, conf: float) -> bool:
        return conf < config.conf_threshold or _has_edge_noise(txt) or _has_suspicious_noise(txt)

    # Selective brightness-filtered retry on up to 2 distinct frames when noisy or low-confidence
    if _needs_retry(winner, win_conf):
        retry_candidates: list[_FrameVote] = [win_vote]
        interior_pool = votes[1:-1] if len(votes) >= 3 else votes
        other_votes = sorted(
            [
                v
                for v in interior_pool
                if v.bgr is not win_bgr
                and not (v.frame_idx >= 0 and win_vote.frame_idx >= 0 and v.frame_idx == win_vote.frame_idx)
            ],
            key=lambda v: v.conf,
            reverse=True,
        )
        if not other_votes and interior_pool is not votes:
            other_votes = sorted(
                [
                    v
                    for v in votes
                    if v.bgr is not win_bgr
                    and not (
                        v.frame_idx >= 0 and win_vote.frame_idx >= 0 and v.frame_idx == win_vote.frame_idx
                    )
                ],
                key=lambda v: v.conf,
                reverse=True,
            )
        if other_votes:
            retry_candidates.append(other_votes[0])

        for rv in retry_candidates:
            retry_raw, retry_conf = ocr_engine.recognize_roi(rv.bgr, brightness_filter=True)
            retry_text = normalize_cjk_text(retry_raw)
            if retry_text and _HANZI_RE.search(retry_text):
                winner_noisy = _has_edge_noise(winner) or _has_suspicious_noise(winner)
                retry_noisy = _has_edge_noise(retry_text) or _has_suspicious_noise(retry_text)
                clean_base_len = len(_strip_edge_noise(winner)) or len(winner)
                cleaned_edge = (
                    winner_noisy
                    and not retry_noisy
                    and retry_conf >= max(0.78, win_conf - 0.05)
                    and len(retry_text) >= max(1, int(0.70 * clean_base_len))
                )
                higher_conf = (
                    (not retry_noisy)
                    and (retry_conf >= win_conf + 0.05)
                    and (len(retry_text) >= max(1, int(0.70 * len(winner))))
                )
                if cleaned_edge or higher_conf:
                    winner = retry_text
                    win_conf = retry_conf
                    win_bgr = rv.bgr
            if not _needs_retry(winner, win_conf):
                break

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
                            same_visual = _tight_signatures_same_cue(m_sig, onset_sig)
                            if same_visual or _boundary_rects_compatible(
                                m_sig.union_rect, onset_sig.union_rect
                            ):
                                start_idx = i
                                if same_visual:
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
                            same_visual = (
                                active.last_sig is None
                                or _tight_signatures_same_cue(active.last_sig, m_sig)
                            )
                            if same_visual or (
                                active.last_sig is not None
                                and _boundary_rects_compatible(
                                    active.last_sig.union_rect, m_sig.union_rect
                                )
                            ):
                                active.end_idx = i
                                active.end_ms = m_rel_ms
                                if same_visual:
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
                            if max(score_a, score_b) < 0.25:
                                classified.append((i, "EMPTY", None))
                            else:
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
    trim_end_ms = max(0, int(config.trim_end_ms)) if config.trim_end_ms is not None else None
    for sp in spans:
        sp.start_ms = frame_pts_ms[sp.start_idx]
        if sp.end_idx + 1 < total_frames:
            sp.end_ms = max(sp.start_ms + 1, frame_pts_ms[sp.end_idx + 1])
        else:
            sp.end_ms = max(sp.start_ms + 1, frame_pts_ms[sp.end_idx] + median_frame_ms)
        if trim_end_ms is not None:
            sp.end_ms = min(sp.end_ms, trim_end_ms)
        sp.last_sig = None

    neighbor_gap = max(config.merge_gap_ms, config.flash_bridge_max_ms)
    eligible: list[_DetectedSpan] = []
    for idx, sp in enumerate(spans):
        dur = sp.end_ms - sp.start_ms
        if not sp.samples or dur <= 0:
            continue
        if dur >= config.min_duration_ms:
            eligible.append(sp)
            continue
        # Keep short spans (including 1-frame 40ms boundary/jitter pieces) if adjacent to a
        # neighboring span so Stage 2 + postprocess can merge real fragments and filter noise;
        # drop only truly isolated sub-threshold flashes.
        near_prev = idx > 0 and (sp.start_ms - spans[idx - 1].end_ms) <= neighbor_gap
        near_next = idx + 1 < len(spans) and (spans[idx + 1].start_ms - sp.end_ms) <= neighbor_gap
        if near_prev or near_next:
            eligible.append(sp)

    return (eligible, frame_pts_ms)


def _find_sandwich_noise(evaluated: list[tuple[int, _FrameVote]]) -> set[int]:
    """Identify isolated middle noise votes sandwiched between two identical-cue votes."""
    absorbed: set[int] = set()
    for k in range(1, len(evaluated) - 1):
        v_prev = evaluated[k - 1][1]
        v_curr = evaluated[k][1]
        v_next = evaluated[k + 1][1]
        if (
            _is_strict_same_cue(v_prev.text, v_next.text, v_prev.conf, v_next.conf)
            and not _is_strict_same_cue(v_prev.text, v_curr.text, v_prev.conf, v_curr.conf)
            and not _is_strict_same_cue(v_curr.text, v_next.text, v_curr.conf, v_next.conf)
        ):
            absorbed.add(k)
    return absorbed


def _sample_mean_luma(bgr: np.ndarray | None) -> float | None:
    if bgr is None or bgr.size == 0:
        return None
    if bgr.ndim == 3 and bgr.shape[2] >= 3:
        gray = cv2.cvtColor(bgr[:, :, :3], cv2.COLOR_BGR2GRAY)
        return float(gray.mean())
    return float(bgr.mean())


def _select_initial_probes(
    n: int,
    get_bgr: Callable[[int], np.ndarray | None],
) -> list[int]:
    """Pick interior-first probe indices, dodging glare/white-flash frames when n >= 8."""
    if n < 4:
        return list(range(n))

    raw_anchors = [
        max(1, round(0.18 * (n - 1))),
        (n - 1) // 2,
        min(n - 2, round(0.82 * (n - 1))),
    ]
    anchors = sorted(dict.fromkeys(raw_anchors))
    if n < 8:
        return anchors

    lumas: dict[int, float] = {}
    for i in range(n):
        val = _sample_mean_luma(get_bgr(i))
        if val is not None:
            lumas[i] = val
    if not lumas:
        return anchors

    median_luma = float(np.median(list(lumas.values())))
    chosen: list[int] = []
    chosen_set: set[int] = set()
    for a in anchors:
        window = [c for c in (a - 1, a, a + 1) if 1 <= c <= n - 2 and c not in chosen_set]
        if not window:
            window = [c for c in (a - 1, a, a + 1) if 1 <= c <= n - 2]
        if window:
            best_c = min(
                window,
                key=lambda c: (abs(lumas.get(c, median_luma) - median_luma), abs(c - a)),
            )
            if best_c not in chosen_set:
                chosen.append(best_c)
                chosen_set.add(best_c)
    return sorted(chosen) if chosen else anchors


def _pick_expansion_probes(n: int, already_evaluated: set[int], max_votes: int) -> list[int]:
    """Pick additional evenly spaced sample indices up to max_votes, preferring interior frames."""
    needed = max_votes - len(already_evaluated)
    if needed <= 0:
        return []

    evaluated = set(already_evaluated)
    picked: list[int] = []
    primary_pool = list(range(1, n - 1)) if n >= 5 else list(range(n))
    fallback_pool = [0, n - 1] if n >= 5 else []

    for pool in (primary_pool, fallback_pool):
        while len(picked) < needed:
            candidates = [idx for idx in pool if idx not in evaluated]
            if not candidates:
                break
            if not evaluated:
                best = candidates[len(candidates) // 2]
            else:
                best = max(
                    candidates,
                    key=lambda c: (min(abs(c - e) for e in evaluated), -abs(c - (n - 1) / 2.0)),
                )
            picked.append(best)
            evaluated.add(best)
    return picked


def recognize_spans_with_voting(
    eligible: list[_DetectedSpan],
    frame_pts_ms: list[int],
    ocr_engine: OcrEngine,
    config: ExtractConfig,
    *,
    ocr_progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[OcrSegment]:
    """Stage 2: Interior-first weighted consensus OCR with adaptive early-exit & binary-search split."""
    total_spans = len(eligible)
    if total_spans == 0:
        return []

    total_frames = len(frame_pts_ms)
    ocr_segments: list[OcrSegment] = []
    pending_cluster: _VoteCluster | None = None

    def _flush_cluster(cl: _VoteCluster | None) -> None:
        if cl is None or not cl.votes:
            return
        if cancel_check is not None and cancel_check():
            raise ExtractionCancelled("Deep OCR recognition cancelled")
        win_text, win_conf, win_bgr = _choose_majority_vote(cl.votes, ocr_engine, config)
        dur_cl = cl.end_ms - cl.start_ms
        stripped = _strip_for_similarity(win_text)
        keep = not (
            (len(cl.votes) == 1 and dur_cl < max(200, config.min_duration_ms) and win_conf < 0.88)
            or (len(cl.votes) <= 2 and len(stripped) <= 2 and win_conf < 0.72)
        )
        if keep:
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
        cl.votes.clear()

    for sp_idx, span in enumerate(eligible):
        if cancel_check is not None and cancel_check():
            raise ExtractionCancelled("Deep OCR recognition cancelled")

        # Eagerly flush pending_cluster once span gap exceeds flash_bridge_max_ms (O(1) RAM)
        if (
            pending_cluster is not None
            and (span.start_ms - pending_cluster.end_ms) > config.flash_bridge_max_ms
        ):
            _flush_cluster(pending_cluster)
            pending_cluster = None

        samples = span.samples
        n = len(samples)
        dur = max(0, span.end_ms - span.start_ms)

        decoded_cache: dict[int, np.ndarray | None] = {}
        eval_cache: dict[int, _FrameVote | None] = {}

        def get_bgr(sample_idx: int) -> np.ndarray | None:
            if sample_idx not in decoded_cache:
                decoded_cache[sample_idx] = _decode_sample(samples[sample_idx])
            return decoded_cache[sample_idx]

        def eval_sample(sample_idx: int) -> _FrameVote | None:
            if sample_idx in eval_cache:
                return eval_cache[sample_idx]
            if cancel_check is not None and cancel_check():
                raise ExtractionCancelled("Deep OCR recognition cancelled")
            bgr = get_bgr(sample_idx)
            if bgr is None:
                eval_cache[sample_idx] = None
                return None
            raw_text, conf = ocr_engine.recognize_roi(bgr)
            norm = normalize_cjk_text(raw_text)
            if norm:
                s = samples[sample_idx]
                vote = _FrameVote(
                    pts_ms=s.pts_ms,
                    text=norm,
                    conf=conf,
                    bgr=bgr,
                    frame_idx=s.frame_idx,
                )
                eval_cache[sample_idx] = vote
                return vote
            eval_cache[sample_idx] = None
            return None

        if n > 0:
            initial_probes = _select_initial_probes(n, get_bgr)
            for idx in initial_probes:
                eval_sample(idx)

            init_votes = [eval_cache[i] for i in initial_probes if eval_cache.get(i) is not None]
            clean_early_exit = (
                n >= 4
                and len(initial_probes) >= 3
                and len(init_votes) == len(initial_probes)
                and len({v.text for v in init_votes}) == 1
                and min(v.conf for v in init_votes) >= 0.93
                and dur <= 2800
                and not _has_edge_noise(init_votes[0].text)
                and not _has_suspicious_noise(init_votes[0].text)
            )
            empty_early_exit = n >= 4 and len(initial_probes) >= 2 and len(init_votes) == 0 and dur <= 1200

            if n >= 4 and not clean_early_exit and not empty_early_exit:
                max_votes = min(n, max(5, min(9, int(dur // 700) + 2)))
                for idx in _pick_expansion_probes(n, set(eval_cache.keys()), max_votes):
                    eval_sample(idx)

            evaluated = sorted((idx, v) for idx, v in eval_cache.items() if v is not None)
            if evaluated:
                absorbed = _find_sandwich_noise(evaluated)
                anchors = [(idx, v) for k, (idx, v) in enumerate(evaluated) if k not in absorbed]

                # Targeted binary-search safety split on disagreeing adjacent evaluated samples
                for pair_idx in range(len(anchors) - 1):
                    lo_idx, lo_vote = anchors[pair_idx]
                    hi_idx, hi_vote = anchors[pair_idx + 1]
                    if hi_idx - lo_idx <= 1:
                        continue
                    if _is_strict_same_cue(lo_vote.text, hi_vote.text, lo_vote.conf, hi_vote.conf):
                        continue
                    for _ in range(5):
                        if hi_idx - lo_idx <= 1:
                            break
                        mid = (lo_idx + hi_idx) // 2
                        probe_idx = mid
                        mid_vote = eval_sample(probe_idx)
                        if mid_vote is None:
                            # Probe expanding offsets -1, +1, -2, +2, ... inside (lo_idx, hi_idx)
                            max_offset = max(mid - lo_idx - 1, hi_idx - mid - 1)
                            for offset in range(1, max_offset + 1):
                                for cand in (mid - offset, mid + offset):
                                    if lo_idx < cand < hi_idx:
                                        if cand in eval_cache and eval_cache[cand] is None:
                                            continue
                                        cand_vote = eval_sample(cand)
                                        if cand_vote is not None:
                                            probe_idx = cand
                                            mid_vote = cand_vote
                                            break
                                if mid_vote is not None:
                                    break
                            if mid_vote is None:
                                break

                        if _is_strict_same_cue(
                            lo_vote.text, mid_vote.text, lo_vote.conf, mid_vote.conf
                        ):
                            lo_idx, lo_vote = probe_idx, mid_vote
                        elif _is_strict_same_cue(
                            mid_vote.text, hi_vote.text, mid_vote.conf, hi_vote.conf
                        ):
                            hi_idx, hi_vote = probe_idx, mid_vote
                        else:
                            break

                evaluated = sorted((idx, v) for idx, v in eval_cache.items() if v is not None)
                absorbed = _find_sandwich_noise(evaluated)

                sub_groups: list[list[tuple[int, _FrameVote]]] = [[evaluated[0]]]
                anchor_vote = evaluated[0][1]
                for k in range(1, len(evaluated)):
                    if k in absorbed:
                        continue
                    idx_k, v_k = evaluated[k]
                    if _is_strict_same_cue(anchor_vote.text, v_k.text, anchor_vote.conf, v_k.conf):
                        sub_groups[-1].append((idx_k, v_k))
                        if not _has_edge_noise(v_k.text) or _has_edge_noise(anchor_vote.text):
                            anchor_vote = v_k
                    else:
                        sub_groups.append([(idx_k, v_k)])
                        anchor_vote = v_k

                prev_g_end: int | None = None
                for g_idx, grp_items in enumerate(sub_groups):
                    grp_votes = [v for _, v in grp_items]
                    first_vote = grp_votes[0]
                    if g_idx == 0:
                        g_start = span.start_ms
                    elif 0 <= first_vote.frame_idx < total_frames:
                        g_start = frame_pts_ms[first_vote.frame_idx]
                    else:
                        g_start = first_vote.pts_ms
                    if prev_g_end is not None and g_start < prev_g_end:
                        g_start = prev_g_end

                    if g_idx == len(sub_groups) - 1:
                        g_end = max(g_start + 1, span.end_ms)
                    else:
                        next_first = sub_groups[g_idx + 1][0][1]
                        if 0 <= next_first.frame_idx < total_frames:
                            next_start_ms = frame_pts_ms[next_first.frame_idx]
                        else:
                            next_start_ms = next_first.pts_ms
                        g_end = max(g_start + 1, next_start_ms)
                    prev_g_end = g_end

                    grp_rep = _group_representative_vote(grp_votes)
                    if pending_cluster is not None:
                        gap_ms = g_start - pending_cluster.end_ms
                        prev_rep = _group_representative_vote(pending_cluster.votes)
                        if gap_ms <= config.flash_bridge_max_ms and _is_strict_same_cue(
                            prev_rep.text, grp_rep.text, prev_rep.conf, grp_rep.conf
                        ):
                            pending_cluster.end_ms = max(pending_cluster.end_ms, g_end)
                            pending_cluster.votes.extend(grp_votes)
                            continue
                        _flush_cluster(pending_cluster)
                    pending_cluster = _VoteCluster(
                        start_ms=g_start, end_ms=g_end, votes=list(grp_votes)
                    )

        eval_cache.clear()
        decoded_cache.clear()
        span.samples.clear()
        span.last_sig = None
        if ocr_progress_cb is not None:
            ocr_progress_cb(sp_idx + 1, total_spans)

    _flush_cluster(pending_cluster)
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
