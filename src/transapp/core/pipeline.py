"""Single-episode hardsub extraction pipeline orchestrator."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import threading
import time

import numpy as np

from transapp.core.deep_extractor import (
    detect_subtitle_spans,
    recognize_spans_with_voting,
)
from transapp.core.models import (
    ExtractConfig,
    ExtractionCancelled,
    ExtractResult,
    FrameTestResult,
    RoiRect,
)
from transapp.core.ocr_engine import OcrEngine
from transapp.core.postprocess import normalize_cjk_text, postprocess_segments
from transapp.core.subtitle_mask import build_stroke_mask
from transapp.core.video_reader import VideoReader

__all__ = [
    "ExtractionCancelled",
    "ExtractionPipeline",
]


def _check_cancelled(cancel_check: Callable[[], bool] | None) -> None:
    if cancel_check is not None and cancel_check():
        raise ExtractionCancelled("Extraction cancelled")


class ExtractionPipeline:
    """Orchestrates video ROI decoding, segmentation, OCR, and post-processing."""

    def __init__(self, ocr_engine: OcrEngine | None = None) -> None:
        self._injected_ocr_engine = ocr_engine
        self._cached_ocr_engine: OcrEngine | None = None
        self._engine_lock = threading.Lock()

    def _resolve_ocr_engine(self, config: ExtractConfig) -> OcrEngine:
        if self._injected_ocr_engine is not None:
            return self._injected_ocr_engine
        requested_provider = (config.provider or "cpu").strip().lower()
        requested_model = (config.ocr_model or "accurate").strip().lower()
        with self._engine_lock:
            if (
                self._cached_ocr_engine is None
                or self._cached_ocr_engine.provider != requested_provider
                or getattr(self._cached_ocr_engine, "ocr_model", "accurate") != requested_model
            ):
                self._cached_ocr_engine = OcrEngine(
                    provider=requested_provider,
                    ocr_model=requested_model,
                )
            return self._cached_ocr_engine

    def test_frame_bgr(
        self,
        full_bgr: np.ndarray,
        frame_index: int,
        pts_ms: int,
        roi: RoiRect,
        config: ExtractConfig,
    ) -> FrameTestResult:
        """Build stroke mask and run OCR on an already-decoded full BGR frame."""
        height, width = full_bgr.shape[:2]
        x1, y1, x2, y2 = roi.to_pixels(width, height)
        bgr_roi = np.ascontiguousarray(full_bgr[y1:y2, x1:x2])

        mask = build_stroke_mask(bgr_roi, config, frame_h=height)
        ocr = self._resolve_ocr_engine(config)
        raw_text, confidence = ocr.recognize_roi(bgr_roi)
        text = normalize_cjk_text(raw_text)

        return FrameTestResult(
            frame_index=frame_index,
            pts_ms=pts_ms,
            mask=mask,
            text=text,
            confidence=confidence,
        )

    def test_single_frame(
        self,
        video_path: Path | str,
        frame_ms: int,
        roi: RoiRect,
        config: ExtractConfig,
    ) -> FrameTestResult:
        """Decode one frame near 0-based frame_ms, build its stroke mask, and run OCR on the ROI."""
        with VideoReader(video_path) as reader:
            start_pts_ms = reader.metadata.start_pts_ms
            target_raw_ms = start_pts_ms + max(0, int(frame_ms))
            frame_idx, actual_raw_ms, full_bgr = reader.read_frame_at_ms(target_raw_ms)

        rel_ms = max(0, actual_raw_ms - start_pts_ms)
        return self.test_frame_bgr(full_bgr, frame_idx, rel_ms, roi, config)

    def run(
        self,
        video_path: Path | str,
        roi: RoiRect,
        config: ExtractConfig,
        progress_cb: Callable[[str, int, int], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> ExtractResult:
        """Run end-to-end subtitle extraction on a video file."""
        t0 = time.perf_counter()
        _check_cancelled(cancel_check)

        trim_start = max(0, int(config.trim_start_ms)) if config.trim_start_ms is not None else None
        trim_end = max(0, int(config.trim_end_ms)) if config.trim_end_ms is not None else None
        if trim_end is not None and trim_end <= (trim_start or 0):
            raise ValueError("trim_start_ms must be smaller than trim_end_ms")

        source_path = Path(video_path)
        with VideoReader(source_path) as reader:
            meta = reader.metadata
            if trim_start is not None and meta.duration_ms > 0 and trim_start >= meta.duration_ms:
                raise ValueError(
                    f"trim_start_ms ({trim_start}) must be smaller than video duration ({meta.duration_ms}ms)"
                )
            ocr = self._resolve_ocr_engine(config)
            start_pts_ms = meta.start_pts_ms
            if trim_end is not None:
                effective_end_ms = min(meta.duration_ms, trim_end) if meta.duration_ms > 0 else trim_end
            else:
                effective_end_ms = meta.duration_ms
            eff_start = trim_start or 0
            eff_end = max(1, effective_end_ms)
            eff_span = max(1, eff_end - eff_start)
            scan_progress = (
                (
                    lambda _frame_idx, pts_ms: progress_cb(
                        "scan",
                        max(0, min(max(0, pts_ms - start_pts_ms) - eff_start, eff_span)),
                        eff_span,
                    )
                )
                if progress_cb is not None
                else None
            )
            detected_spans, frame_pts_ms = detect_subtitle_spans(
                reader.iter_roi_frames(roi, start_ms=trim_start, end_ms=trim_end),
                meta.fps,
                start_pts_ms,
                ocr,
                config,
                scan_progress_cb=scan_progress,
                cancel_check=cancel_check,
            )

        _check_cancelled(cancel_check)

        ocr_progress = (
            (lambda current, total: progress_cb("ocr", current, total))
            if progress_cb is not None
            else None
        )
        ocr_segments = recognize_spans_with_voting(
            detected_spans,
            frame_pts_ms,
            ocr,
            config,
            ocr_progress_cb=ocr_progress,
            cancel_check=cancel_check,
        )

        _check_cancelled(cancel_check)

        if trim_end is None and frame_pts_ms:
            deltas = [b - a for a, b in zip(frame_pts_ms, frame_pts_ms[1:]) if b > a]
            if deltas:
                median_frame_ms = max(1, int(round(float(np.median(deltas)))))
            else:
                median_frame_ms = max(1, round(1000.0 / max(1.0, meta.fps)))
            effective_duration_ms = max(meta.duration_ms, frame_pts_ms[-1] + median_frame_ms)
        else:
            effective_duration_ms = effective_end_ms

        events = postprocess_segments(ocr_segments, config, duration_ms=effective_duration_ms)
        elapsed_sec = time.perf_counter() - t0
        if progress_cb is not None:
            progress_cb("done", len(events), len(events))

        return ExtractResult(
            video_path=source_path,
            width=meta.width,
            height=meta.height,
            duration_ms=meta.duration_ms,
            events=events,
            elapsed_sec=elapsed_sec,
            frame_pts_ms=frame_pts_ms,
        )
