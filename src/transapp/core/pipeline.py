"""Single-episode hardsub extraction pipeline orchestrator."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import threading
import time

import numpy as np

from transapp.core.models import (
    ExtractConfig,
    ExtractionCancelled,
    ExtractResult,
    FrameTestResult,
    RoiRect,
)
from transapp.core.ocr_engine import OcrEngine
from transapp.core.postprocess import normalize_cjk_text, postprocess_segments
from transapp.core.segmenter import segment_roi_frames
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
        requested_model = (config.ocr_model or "fast").strip().lower()
        with self._engine_lock:
            if (
                self._cached_ocr_engine is None
                or self._cached_ocr_engine.provider != requested_provider
                or getattr(self._cached_ocr_engine, "ocr_model", "fast") != requested_model
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

        mask = build_stroke_mask(bgr_roi, config)
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
        """Decode one frame near frame_ms, build its stroke mask, and run OCR on the ROI."""
        with VideoReader(video_path) as reader:
            frame_idx, pts_ms, full_bgr = reader.read_frame_at_ms(frame_ms)

        return self.test_frame_bgr(full_bgr, frame_idx, pts_ms, roi, config)

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

        source_path = Path(video_path)
        with VideoReader(source_path) as reader:
            meta = reader.metadata
            total_ms = max(1, meta.duration_ms)
            start_pts_ms = meta.start_pts_ms
            scan_progress = (
                (
                    lambda _frame_idx, pts_ms: progress_cb(
                        "scan",
                        max(0, min(pts_ms - start_pts_ms, total_ms)),
                        total_ms,
                    )
                )
                if progress_cb is not None
                else None
            )
            raw_segments = segment_roi_frames(
                reader.iter_roi_frames(roi),
                config,
                on_frame_progress=scan_progress,
                cancel_check=cancel_check,
            )

        _check_cancelled(cancel_check)

        # Skip OCR for segments shorter than min_duration_ms; postprocess drops them unconditionally.
        eligible_segments = [
            seg
            for seg in raw_segments
            if (seg.end_ms - seg.start_ms) >= config.min_duration_ms
        ]

        ocr = self._resolve_ocr_engine(config)
        ocr_progress = (
            (lambda current, total: progress_cb("ocr", current, total))
            if progress_cb is not None
            else None
        )
        ocr_segments = ocr.process_segments(
            eligible_segments,
            progress_cb=ocr_progress,
            cancel_check=cancel_check,
        )

        _check_cancelled(cancel_check)

        events = postprocess_segments(ocr_segments, config)
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
        )
