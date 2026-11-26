"""Coarse-to-fine ring-buffer hardsub frame segmenter."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import statistics

import cv2
import numpy as np

from transapp.core.models import ExtractConfig, ExtractionCancelled, RawSegment
from transapp.core.subtitle_mask import (
    build_stroke_mask,
    frame_sharpness,
    mask_fg_ratio,
    mask_iou,
)
from transapp.core.video_reader import FrameSlice

_DEFAULT_FRAME_STEP_MS = 40
_MAX_CANDIDATE_FRAMES = 64
_MAX_DELTA_SAMPLES = 256
_MAX_FADE_TAIL_STEPS = 2


@dataclass(slots=True)
class _BufferedFrame:
    pos: int
    slice: FrameSlice
    mask: np.ndarray | None = None
    fg_ratio: float | None = None

    def ensure_mask(self, config: ExtractConfig) -> tuple[np.ndarray, float]:
        if self.mask is None or self.fg_ratio is None:
            self.mask = build_stroke_mask(self.slice.bgr_roi, config)
            self.fg_ratio = mask_fg_ratio(self.mask)
        return self.mask, self.fg_ratio


@dataclass(slots=True)
class _CandidateSampler:
    """Bounded uniform sampler retaining up to _MAX_CANDIDATE_FRAMES from an active cue."""

    items: list[tuple[int, FrameSlice]] = field(default_factory=list)
    total_count: int = 0
    stride: int = 1
    _last_pos: int = -1

    def append(self, pos: int, frame_slice: FrameSlice) -> None:
        if pos <= self._last_pos:
            return
        self._last_pos = pos
        idx = self.total_count
        self.total_count += 1
        if idx % self.stride != 0:
            return
        if len(self.items) >= _MAX_CANDIDATE_FRAMES:
            self.items = self.items[::2]
            self.stride *= 2
            if idx % self.stride != 0:
                return
        self.items.append((pos, frame_slice))

    def pick_peak(self) -> FrameSlice:
        n = len(self.items)
        if n == 0:
            raise ValueError("Cannot select peak frame from an empty segment")
        if n <= 3:
            mid = (n - 1) / 2.0
            candidate_indices = sorted(range(n), key=lambda i: (abs(i - mid), i))
        else:
            # Evaluate 50% first so exact sharpness ties deterministically prefer the middle frame.
            candidate_indices = list(
                dict.fromkeys(int(round(p * (n - 1))) for p in (0.50, 0.35, 0.65))
            )

        best_slice = self.items[candidate_indices[0]][1]
        best_score = frame_sharpness(best_slice.bgr_roi)
        for idx in candidate_indices[1:]:
            candidate = self.items[idx][1]
            score = frame_sharpness(candidate.bgr_roi)
            if score > best_score:
                best_score = score
                best_slice = candidate
        return best_slice


@dataclass(slots=True)
class _ActiveSegment:
    start_pos: int
    start_slice: FrameSlice
    anchor_mask: np.ndarray
    anchor_fg: float
    last_confirmed_pos: int
    last_confirmed_slice: FrameSlice
    sampler: _CandidateSampler
    fade_tail_count: int = 0


def _mask_containment(small_mask: np.ndarray, ref_mask: np.ndarray, dilate_px: int = 1) -> float:
    """Fraction of small_mask foreground pixels covered by dilated ref_mask."""
    fg_small = small_mask > 0
    small_count = int(np.count_nonzero(fg_small))
    if small_count == 0:
        return 0.0
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        ref_bin = cv2.dilate((ref_mask > 0).astype(np.uint8), kernel) > 0
    else:
        ref_bin = ref_mask > 0
    covered = int(np.count_nonzero(fg_small & ref_bin))
    return float(covered) / float(small_count)


def _is_edge_match(
    mask: np.ndarray,
    fg_ratio: float,
    ref_mask: np.ndarray,
    ref_fg: float,
    config: ExtractConfig,
) -> bool:
    if fg_ratio < config.fg_ratio_edge:
        return False
    if mask_iou(mask, ref_mask, dilate_px=1) >= config.split_iou_min * 0.5:
        return True
    if fg_ratio < max(config.fg_ratio_min, ref_fg * 0.6):
        return _mask_containment(mask, ref_mask, dilate_px=1) >= 0.65
    return False


def _belongs_to_next_cue(
    mask: np.ndarray,
    fg_ratio: float,
    prev_anchor_mask: np.ndarray,
    next_mask: np.ndarray,
    next_fg: float,
    config: ExtractConfig,
) -> bool:
    if not _is_edge_match(mask, fg_ratio, next_mask, next_fg, config):
        return False
    iou_next = mask_iou(mask, next_mask, dilate_px=1)
    iou_prev = mask_iou(mask, prev_anchor_mask, dilate_px=1)
    if iou_next != iou_prev:
        return iou_next > iou_prev
    return _mask_containment(mask, next_mask, dilate_px=1) > _mask_containment(
        mask, prev_anchor_mask, dilate_px=1
    )


def _open_segment_from_buffer(
    buf: deque[_BufferedFrame],
    trigger: _BufferedFrame,
    trigger_mask: np.ndarray,
    trigger_fg: float,
    min_pos: int,
    config: ExtractConfig,
) -> _ActiveSegment:
    """Scan backward inside the ring buffer to attach fade-in / non-coarse leading frames."""
    core_mask = trigger_mask
    core_fg = trigger_fg
    matched: list[_BufferedFrame] = [trigger]

    # buf is strictly ordered by pos; iterate backward from frame before trigger.
    prior_frames = [b for b in buf if min_pos <= b.pos < trigger.pos]
    for b in reversed(prior_frames):
        b_mask, b_fg = b.ensure_mask(config)
        if not _is_edge_match(b_mask, b_fg, core_mask, core_fg, config):
            break
        matched.append(b)
        if b_fg >= core_fg and b_fg >= config.fg_ratio_min:
            core_mask = b_mask
            core_fg = b_fg

    matched.reverse()
    sampler = _CandidateSampler()
    for b in matched:
        sampler.append(b.pos, b.slice)

    return _ActiveSegment(
        start_pos=matched[0].pos,
        start_slice=matched[0].slice,
        anchor_mask=core_mask,
        anchor_fg=core_fg,
        last_confirmed_pos=trigger.pos,
        last_confirmed_slice=trigger.slice,
        sampler=sampler,
    )


def _finalize_segment(active: _ActiveSegment, end_ms: int) -> RawSegment:
    peak = active.sampler.pick_peak()
    start_ms = active.start_slice.pts_ms
    safe_end_ms = end_ms if end_ms > start_ms else start_ms + 1
    return RawSegment(
        start_ms=start_ms,
        end_ms=safe_end_ms,
        peak_bgr=np.ascontiguousarray(peak.bgr_roi.copy()),
        frame_index=peak.frame_index,
    )


def _estimate_frame_step_ms(recent_deltas: deque[int]) -> int:
    if not recent_deltas:
        return _DEFAULT_FRAME_STEP_MS
    return max(1, int(round(statistics.median(recent_deltas))))


def _extend_active(active: _ActiveSegment, buf: deque[_BufferedFrame], bf: _BufferedFrame) -> None:
    for b in buf:
        if active.last_confirmed_pos < b.pos <= bf.pos:
            active.sampler.append(b.pos, b.slice)
    active.last_confirmed_pos = bf.pos
    active.last_confirmed_slice = bf.slice


def segment_roi_frames(
    frames: Iterable[FrameSlice],
    config: ExtractConfig,
    on_frame_progress: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[RawSegment]:
    """Segment decoded ROI frames into subtitle RawSegments using coarse-to-fine scanning."""
    step = max(1, int(config.coarse_step))
    buf: deque[_BufferedFrame] = deque(maxlen=step + 2)
    recent_deltas: deque[int] = deque(maxlen=_MAX_DELTA_SAMPLES)

    segments: list[RawSegment] = []
    active: _ActiveSegment | None = None
    last_closed_pos = -1
    last_evaluated_pos = -1
    prev_pts_ms: int | None = None

    def evaluate(bf: _BufferedFrame) -> None:
        nonlocal active, last_closed_pos, last_evaluated_pos
        last_evaluated_pos = bf.pos
        curr_mask, curr_fg = bf.ensure_mask(config)

        if active is None:
            if curr_fg >= config.fg_ratio_min:
                active = _open_segment_from_buffer(
                    buf, bf, curr_mask, curr_fg, last_closed_pos + 1, config
                )
            return

        iou = mask_iou(active.anchor_mask, curr_mask, dilate_px=1)
        is_same_solid = curr_fg >= config.fg_ratio_min and iou >= config.split_iou_min
        is_early_fade_ramp = (
            curr_fg >= config.fg_ratio_min
            and (bf.pos - active.start_pos) <= step + 2
            and curr_fg >= active.anchor_fg * 1.4
            and _mask_containment(active.anchor_mask, curr_mask, dilate_px=1) >= 0.70
        )
        if is_same_solid or is_early_fade_ramp:
            active.fade_tail_count = 0
            if curr_fg >= active.anchor_fg:
                active.anchor_mask = curr_mask
                active.anchor_fg = curr_fg
            _extend_active(active, buf, bf)
            return

        is_fade_tail = (
            curr_fg >= config.fg_ratio_edge
            and active.fade_tail_count < _MAX_FADE_TAIL_STEPS
            and (curr_fg < config.fg_ratio_min or curr_fg < active.anchor_fg * 0.6)
            and (
                iou >= config.split_iou_min * 0.5
                or _mask_containment(curr_mask, active.anchor_mask, dilate_px=1) >= 0.70
            )
        )
        if is_fade_tail:
            active.fade_tail_count += 1
            _extend_active(active, buf, bf)
            return

        # Transition: ACTIVE -> EMPTY or ACTIVE_A -> ACTIVE_B. Refine exact boundary in ring buffer.
        has_next_cue = curr_fg >= config.fg_ratio_min
        tail = [b for b in buf if active.last_confirmed_pos < b.pos <= bf.pos]
        end_boundary = bf.slice
        for b in tail[:-1]:
            b_mask, b_fg = b.ensure_mask(config)
            matches_prev = _is_edge_match(
                b_mask, b_fg, active.anchor_mask, active.anchor_fg, config
            )
            if matches_prev and (
                not has_next_cue
                or not _belongs_to_next_cue(
                    b_mask, b_fg, active.anchor_mask, curr_mask, curr_fg, config
                )
            ):
                active.sampler.append(b.pos, b.slice)
                active.last_confirmed_pos = b.pos
                active.last_confirmed_slice = b.slice
            else:
                end_boundary = b.slice
                break

        segments.append(_finalize_segment(active, end_boundary.pts_ms))
        last_closed_pos = active.last_confirmed_pos
        active = None

        if has_next_cue:
            active = _open_segment_from_buffer(
                buf, bf, curr_mask, curr_fg, last_closed_pos + 1, config
            )

    it = iter(frames)
    pos = 0
    while True:
        if cancel_check is not None and cancel_check():
            raise ExtractionCancelled("Frame segmentation cancelled")
        try:
            frame_slice = next(it)
        except StopIteration:
            break

        if prev_pts_ms is not None and frame_slice.pts_ms > prev_pts_ms:
            recent_deltas.append(frame_slice.pts_ms - prev_pts_ms)
        prev_pts_ms = frame_slice.pts_ms

        if on_frame_progress is not None:
            on_frame_progress(frame_slice.frame_index, frame_slice.pts_ms)

        bf = _BufferedFrame(pos=pos, slice=frame_slice)
        buf.append(bf)

        if pos % step == 0:
            evaluate(bf)

        pos += 1

    if buf and last_evaluated_pos != buf[-1].pos:
        evaluate(buf[-1])

    if active is not None:
        eof_end_ms = active.last_confirmed_slice.pts_ms + _estimate_frame_step_ms(recent_deltas)
        segments.append(_finalize_segment(active, eof_end_ms))

    return segments
