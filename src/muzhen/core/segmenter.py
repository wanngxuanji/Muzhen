"""Coarse-to-fine ring-buffer hardsub frame segmenter with EMA anchor and flash bridging."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
import statistics

import cv2
import numpy as np

from muzhen.core.models import ExtractConfig, ExtractionCancelled, RawSegment
from muzhen.core.subtitle_mask import (
    build_stroke_mask,
    build_temporal_stable_mask,
    effective_dilate_px,
    fast_luma_channel,
    fg_pixel_thresholds,
    mask_iou,
    precompute_score_regions,
    score_candidate_frame,
)
from muzhen.core.video_reader import FrameSlice

_DEFAULT_FRAME_STEP_MS = 40
_MAX_CANDIDATE_FRAMES = 64
_MAX_TEMPORAL_SAMPLES = 5
_MAX_DELTA_SAMPLES = 256
_MAX_FADE_TAIL_STEPS = 2


@dataclass(slots=True)
class _BufferedFrame:
    pos: int
    slice: FrameSlice
    mask: np.ndarray | None = None
    fg_px: int | None = None
    mean_v: float | None = None

    def ensure_mask(
        self, config: ExtractConfig, frame_h: int | None
    ) -> tuple[np.ndarray, int]:
        if self.mask is None or self.fg_px is None:
            self.mask = build_stroke_mask(self.slice.bgr_roi, config, frame_h=frame_h)
            self.fg_px = int(np.count_nonzero(self.mask))
        return self.mask, self.fg_px

    def ensure_mean_v(self) -> float:
        if self.mean_v is None:
            bgr = self.slice.bgr_roi
            self.mean_v = float(cv2.mean(fast_luma_channel(bgr))[0]) if bgr.size else 0.0
        return self.mean_v


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

    def select_peak_and_mask(
        self,
        config: ExtractConfig,
        *,
        frame_h: int | None = None,
    ) -> tuple[FrameSlice, np.ndarray]:
        """Select cleanest/sharpest frame and compute temporal-stable mask."""
        n = len(self.items)
        if n == 0:
            raise ValueError("Cannot select peak frame from an empty segment")

        mid = (n - 1) / 2.0
        if n <= 5:
            sample_indices = list(range(n))
        else:
            lo = int(round(0.15 * (n - 1)))
            hi = int(round(0.85 * (n - 1)))
            if hi <= lo:
                lo, hi = 0, n - 1
            raw_idxs = np.linspace(lo, hi, num=min(_MAX_TEMPORAL_SAMPLES, hi - lo + 1))
            sample_indices = list(dict.fromkeys(int(round(v)) for v in raw_idxs))

        ordered_indices = sorted(sample_indices, key=lambda i: (abs(i - mid), i))
        sample_slices = [self.items[i][1] for i in ordered_indices]
        sample_bgrs = [s.bgr_roi for s in sample_slices]
        sample_masks = [build_stroke_mask(bgr, config, frame_h=frame_h) for bgr in sample_bgrs]

        stable_mask = build_temporal_stable_mask(
            sample_bgrs, sample_masks, config, frame_h=frame_h
        )
        regions = precompute_score_regions(
            stable_mask, sample_bgrs[0].shape[0], frame_h=frame_h
        )

        best_slice = sample_slices[0]
        best_score = score_candidate_frame(
            sample_bgrs[0],
            sample_masks[0],
            stable_mask,
            frame_h=frame_h,
            _precomputed=regions,
        )
        for s_slice, bgr, mask in zip(sample_slices[1:], sample_bgrs[1:], sample_masks[1:]):
            score = score_candidate_frame(
                bgr, mask, stable_mask, frame_h=frame_h, _precomputed=regions
            )
            if score > best_score:
                best_score = score
                best_slice = s_slice

        return best_slice, stable_mask


@dataclass(slots=True)
class _ActiveSegment:
    start_pos: int
    start_slice: FrameSlice
    anchor_ema: np.ndarray
    anchor_mask: np.ndarray
    anchor_fg_px: int
    baseline_luma: float
    last_confirmed_pos: int
    last_confirmed_slice: FrameSlice
    sampler: _CandidateSampler
    fade_tail_count: int = 0
    transient_spike_count: int = 0
    pending_end_slice: FrameSlice | None = None
    pending_closed_pos: int = -1
    gap_flash_detected: bool = False
    gap_prev_bgr: np.ndarray | None = None


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
    fg_px: int,
    ref_mask: np.ndarray,
    ref_fg_px: int,
    fg_min_px: int,
    fg_edge_px: int,
    dil_px: int,
    config: ExtractConfig,
) -> bool:
    if fg_px < fg_edge_px:
        return False
    if mask_iou(mask, ref_mask, dilate_px=dil_px) >= config.split_iou_min * 0.5:
        return True
    if fg_px < max(fg_min_px, int(round(ref_fg_px * 0.6))):
        return _mask_containment(mask, ref_mask, dilate_px=dil_px) >= 0.65
    return False


def _belongs_to_next_cue(
    mask: np.ndarray,
    fg_px: int,
    prev_anchor_mask: np.ndarray,
    next_mask: np.ndarray,
    next_fg_px: int,
    fg_min_px: int,
    fg_edge_px: int,
    dil_px: int,
    config: ExtractConfig,
) -> bool:
    if not _is_edge_match(
        mask, fg_px, next_mask, next_fg_px, fg_min_px, fg_edge_px, dil_px, config
    ):
        return False
    iou_next = mask_iou(mask, next_mask, dilate_px=dil_px)
    iou_prev = mask_iou(mask, prev_anchor_mask, dilate_px=dil_px)
    if iou_next != iou_prev:
        return iou_next > iou_prev
    return _mask_containment(mask, next_mask, dilate_px=dil_px) > _mask_containment(
        mask, prev_anchor_mask, dilate_px=dil_px
    )


def _detect_gap_disturbance(
    bgr_roi: np.ndarray,
    baseline_luma: float,
    prev_bgr: np.ndarray | None,
) -> bool:
    """Detect bright flash / glare / abrupt scene cut inside a dropout gap."""
    if bgr_roi.size == 0:
        return False
    v_chan = fast_luma_channel(bgr_roi)
    mean_v = float(cv2.mean(v_chan)[0])
    if mean_v >= baseline_luma + 22.0:
        return True
    if mean_v >= baseline_luma + 10.0 and cv2.countNonZero(v_chan >= 215) >= int(0.10 * v_chan.size):
        return True
    if prev_bgr is not None and prev_bgr.shape == bgr_roi.shape:
        mad = float(cv2.mean(cv2.absdiff(bgr_roi, prev_bgr))[0])
        if mad >= 28.0:
            return True
    return False


def _is_partial_glare_frame(
    bgr_roi: np.ndarray,
    curr_mask: np.ndarray,
    anchor_mask: np.ndarray,
    baseline_luma: float,
    dil_px: int,
) -> bool:
    """Check whether missing strokes are caused by localized bright glare rather than a cue change."""
    if _mask_containment(curr_mask, anchor_mask, dilate_px=dil_px) < 0.85:
        return False

    k = 2 * max(2, dil_px + 1) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    curr_dil = cv2.dilate((curr_mask > 0).astype(np.uint8), kernel) > 0
    missing = (anchor_mask > 0) & (~curr_dil)
    if not np.any(missing):
        return False

    v_chan = fast_luma_channel(bgr_roi)
    missing_mean_v = float(v_chan[missing].mean())
    curr_mean_v = float(cv2.mean(v_chan)[0])
    return missing_mean_v >= 180.0 or curr_mean_v >= baseline_luma + 22.0


def _open_segment_from_buffer(
    buf: deque[_BufferedFrame],
    trigger: _BufferedFrame,
    trigger_mask: np.ndarray,
    trigger_fg_px: int,
    min_pos: int,
    fg_min_px: int,
    fg_edge_px: int,
    dil_px: int,
    frame_h: int | None,
    config: ExtractConfig,
) -> _ActiveSegment:
    """Scan backward inside the ring buffer to attach fade-in / non-coarse leading frames."""
    core_mask = trigger_mask
    core_fg_px = trigger_fg_px
    matched: list[_BufferedFrame] = [trigger]

    prior_frames = [b for b in buf if min_pos <= b.pos < trigger.pos]
    for b in reversed(prior_frames):
        b_mask, b_fg_px = b.ensure_mask(config, frame_h)
        if not _is_edge_match(
            b_mask, b_fg_px, core_mask, core_fg_px, fg_min_px, fg_edge_px, dil_px, config
        ):
            break
        matched.append(b)
        if b_fg_px >= core_fg_px and b_fg_px >= fg_min_px:
            core_mask = b_mask
            core_fg_px = b_fg_px

    matched.reverse()
    sampler = _CandidateSampler()
    for b in matched:
        sampler.append(b.pos, b.slice)

    anchor_ema = (core_mask > 0).astype(np.float32)
    return _ActiveSegment(
        start_pos=matched[0].pos,
        start_slice=matched[0].slice,
        anchor_ema=anchor_ema,
        anchor_mask=core_mask,
        anchor_fg_px=core_fg_px,
        baseline_luma=trigger.ensure_mean_v(),
        last_confirmed_pos=trigger.pos,
        last_confirmed_slice=trigger.slice,
        sampler=sampler,
    )


def _finalize_segment(
    active: _ActiveSegment,
    end_ms: int,
    config: ExtractConfig,
    *,
    frame_h: int | None = None,
) -> RawSegment:
    peak, stable_mask = active.sampler.select_peak_and_mask(config, frame_h=frame_h)
    start_ms = active.start_slice.rel_ms
    safe_end_ms = end_ms if end_ms > start_ms else start_ms + 1
    return RawSegment(
        start_ms=start_ms,
        end_ms=safe_end_ms,
        peak_bgr=np.ascontiguousarray(peak.bgr_roi.copy()),
        frame_index=peak.frame_index,
        stable_mask=stable_mask,
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


def _update_anchor_ema(active: _ActiveSegment, curr_mask: np.ndarray, curr_fg_px: int) -> None:
    """Update anchor_mask smoothly via EMA while guarding against transient background spikes."""
    if curr_fg_px > int(round(active.anchor_fg_px * 1.35)):
        return
    curr_bin = (curr_mask > 0).astype(np.float32)
    active.anchor_ema = 0.75 * active.anchor_ema + 0.25 * curr_bin
    updated = (active.anchor_ema >= 0.45).astype(np.uint8) * 255
    updated_px = int(np.count_nonzero(updated))
    if updated_px > 0:
        active.anchor_mask = updated
        active.anchor_fg_px = updated_px


def segment_roi_frames(
    frames: Iterable[FrameSlice],
    config: ExtractConfig,
    on_frame_progress: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    *,
    frame_h: int | None = None,
) -> list[RawSegment]:
    """Segment decoded ROI frames into subtitle RawSegments using coarse-to-fine scanning."""
    step = max(1, int(config.coarse_step))
    buf: deque[_BufferedFrame] = deque(maxlen=max(step + 2, 6))
    recent_deltas: deque[int] = deque(maxlen=_MAX_DELTA_SAMPLES)

    dil_px = effective_dilate_px(frame_h)
    bridge_dil_px = max(dil_px, 2)
    fg_min_px: int | None = None
    fg_edge_px: int | None = None

    segments: list[RawSegment] = []
    active: _ActiveSegment | None = None
    last_closed_pos = -1
    last_evaluated_pos = -1
    prev_pts_ms: int | None = None

    def close_active_at(end_ms: int, closed_pos: int) -> None:
        nonlocal active, last_closed_pos
        assert active is not None
        segments.append(_finalize_segment(active, end_ms, config, frame_h=frame_h))
        last_closed_pos = closed_pos
        active = None

    def evaluate(bf: _BufferedFrame) -> None:
        nonlocal active, last_closed_pos, last_evaluated_pos, fg_min_px, fg_edge_px
        last_evaluated_pos = bf.pos
        if fg_min_px is None or fg_edge_px is None:
            rh, rw = bf.slice.bgr_roi.shape[:2]
            fg_min_px, fg_edge_px = fg_pixel_thresholds(rh, rw, config, frame_h=frame_h)

        curr_mask, curr_fg_px = bf.ensure_mask(config, frame_h)

        if active is None:
            if curr_fg_px >= fg_min_px:
                active = _open_segment_from_buffer(
                    buf,
                    bf,
                    curr_mask,
                    curr_fg_px,
                    last_closed_pos + 1,
                    fg_min_px,
                    fg_edge_px,
                    dil_px,
                    frame_h,
                    config,
                )
            return

        if active.pending_end_slice is not None:
            active.gap_flash_detected = active.gap_flash_detected or _detect_gap_disturbance(
                bf.slice.bgr_roi, active.baseline_luma, active.gap_prev_bgr
            )
            active.gap_prev_bgr = bf.slice.bgr_roi
            gap_ms = bf.slice.rel_ms - active.pending_end_slice.rel_ms

            if curr_fg_px >= fg_min_px:
                can_bridge = False
                if gap_ms <= config.flash_bridge_max_ms:
                    iou_b = mask_iou(active.anchor_mask, curr_mask, dilate_px=bridge_dil_px)
                    cont_b = max(
                        _mask_containment(curr_mask, active.anchor_mask, dilate_px=bridge_dil_px),
                        _mask_containment(active.anchor_mask, curr_mask, dilate_px=bridge_dil_px),
                    )
                    if gap_ms <= config.merge_gap_ms:
                        can_bridge = (iou_b >= 0.50) or (
                            active.gap_flash_detected and iou_b >= 0.38 and cont_b >= 0.78
                        )
                    elif active.gap_flash_detected:
                        can_bridge = (iou_b >= 0.48) or (iou_b >= 0.36 and cont_b >= 0.78)

                if can_bridge:
                    active.pending_end_slice = None
                    active.pending_closed_pos = -1
                    active.gap_flash_detected = False
                    active.gap_prev_bgr = None
                    active.fade_tail_count = 0
                    active.baseline_luma = 0.8 * active.baseline_luma + 0.2 * bf.ensure_mean_v()
                    _update_anchor_ema(active, curr_mask, curr_fg_px)
                    active.sampler.append(bf.pos, bf.slice)
                    active.last_confirmed_pos = bf.pos
                    active.last_confirmed_slice = bf.slice
                    return

                close_active_at(active.pending_end_slice.rel_ms, active.pending_closed_pos)
                active = _open_segment_from_buffer(
                    buf,
                    bf,
                    curr_mask,
                    curr_fg_px,
                    last_closed_pos + 1,
                    fg_min_px,
                    fg_edge_px,
                    dil_px,
                    frame_h,
                    config,
                )
                return

            if gap_ms > config.flash_bridge_max_ms:
                close_active_at(active.pending_end_slice.rel_ms, active.pending_closed_pos)
            return

        iou = mask_iou(active.anchor_mask, curr_mask, dilate_px=dil_px)
        is_same_solid = curr_fg_px >= fg_min_px and iou >= config.split_iou_min
        is_early_fade_ramp = (
            curr_fg_px >= fg_min_px
            and (bf.pos - active.start_pos) <= step + 2
            and curr_fg_px >= int(round(active.anchor_fg_px * 1.4))
            and _mask_containment(active.anchor_mask, curr_mask, dilate_px=dil_px) >= 0.70
        )
        if is_same_solid or is_early_fade_ramp:
            active.fade_tail_count = 0
            active.transient_spike_count = 0
            active.baseline_luma = 0.85 * active.baseline_luma + 0.15 * bf.ensure_mean_v()
            if is_early_fade_ramp:
                active.anchor_mask = curr_mask
                active.anchor_ema = (curr_mask > 0).astype(np.float32)
                active.anchor_fg_px = curr_fg_px
            else:
                _update_anchor_ema(active, curr_mask, curr_fg_px)
            _extend_active(active, buf, bf)
            return

        if (
            curr_fg_px > int(round(active.anchor_fg_px * 1.35))
            and active.transient_spike_count < _MAX_FADE_TAIL_STEPS
            and _mask_containment(active.anchor_mask, curr_mask, dilate_px=dil_px) >= 0.90
        ):
            active.transient_spike_count += 1
            active.fade_tail_count = 0
            active.last_confirmed_pos = bf.pos
            active.last_confirmed_slice = bf.slice
            return

        if (
            curr_fg_px >= fg_edge_px
            and curr_fg_px < active.anchor_fg_px
            and _is_partial_glare_frame(
                bf.slice.bgr_roi, curr_mask, active.anchor_mask, active.baseline_luma, dil_px
            )
        ):
            active.fade_tail_count = 0
            active.last_confirmed_pos = bf.pos
            active.last_confirmed_slice = bf.slice
            return

        is_fade_tail = (
            fg_edge_px <= curr_fg_px < fg_min_px
            and active.fade_tail_count < _MAX_FADE_TAIL_STEPS
            and (
                iou >= config.split_iou_min * 0.5
                or _mask_containment(curr_mask, active.anchor_mask, dilate_px=dil_px) >= 0.70
            )
        )
        if is_fade_tail:
            active.fade_tail_count += 1
            _extend_active(active, buf, bf)
            return

        has_next_cue = curr_fg_px >= fg_min_px
        tail = [b for b in buf if active.last_confirmed_pos < b.pos <= bf.pos]
        end_boundary = bf.slice
        for b in tail[:-1]:
            b_mask, b_fg_px = b.ensure_mask(config, frame_h)
            matches_prev = _is_edge_match(
                b_mask, b_fg_px, active.anchor_mask, active.anchor_fg_px, fg_min_px, fg_edge_px, dil_px, config
            )
            if matches_prev and (
                not has_next_cue
                or not _belongs_to_next_cue(
                    b_mask,
                    b_fg_px,
                    active.anchor_mask,
                    curr_mask,
                    curr_fg_px,
                    fg_min_px,
                    fg_edge_px,
                    dil_px,
                    config,
                )
            ):
                active.sampler.append(b.pos, b.slice)
                active.last_confirmed_pos = b.pos
                active.last_confirmed_slice = b.slice
            else:
                end_boundary = b.slice
                break

        if not has_next_cue and config.flash_bridge_max_ms > 0:
            active.pending_end_slice = end_boundary
            active.pending_closed_pos = active.last_confirmed_pos
            active.gap_flash_detected = _detect_gap_disturbance(
                bf.slice.bgr_roi, active.baseline_luma, active.last_confirmed_slice.bgr_roi
            )
            active.gap_prev_bgr = bf.slice.bgr_roi
            return

        close_active_at(end_boundary.rel_ms, active.last_confirmed_pos)

        if has_next_cue:
            active = _open_segment_from_buffer(
                buf,
                bf,
                curr_mask,
                curr_fg_px,
                last_closed_pos + 1,
                fg_min_px,
                fg_edge_px,
                dil_px,
                frame_h,
                config,
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

        if prev_pts_ms is not None and frame_slice.rel_ms > prev_pts_ms:
            recent_deltas.append(frame_slice.rel_ms - prev_pts_ms)
        prev_pts_ms = frame_slice.rel_ms

        if on_frame_progress is not None:
            on_frame_progress(frame_slice.frame_index, frame_slice.rel_ms)

        bf = _BufferedFrame(pos=pos, slice=frame_slice)
        buf.append(bf)

        if pos % step == 0:
            evaluate(bf)

        pos += 1

    if buf and last_evaluated_pos != buf[-1].pos:
        evaluate(buf[-1])

    if active is not None:
        if active.pending_end_slice is not None:
            close_active_at(active.pending_end_slice.rel_ms, active.pending_closed_pos)
        else:
            eof_end_ms = active.last_confirmed_slice.rel_ms + _estimate_frame_step_ms(recent_deltas)
            close_active_at(eof_end_ms, active.last_confirmed_pos)

    return segments
