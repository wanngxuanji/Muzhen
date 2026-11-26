"""Core domain dataclasses for TransApp."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class RoiRect:
    """Normalized region of interest with coordinates in [0.0, 1.0]."""

    x: float
    y: float
    w: float
    h: float

    def clamped(self) -> RoiRect:
        """Return a copy clamped so the rectangle lies inside [0.0, 1.0]^2."""
        x = max(0.0, min(1.0, float(self.x)))
        y = max(0.0, min(1.0, float(self.y)))
        w = max(0.0, min(1.0 - x, float(self.w)))
        h = max(0.0, min(1.0 - y, float(self.h)))
        return RoiRect(x=x, y=y, w=w, h=h)

    def to_pixels(self, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        """Convert normalized ROI to clipped pixel bounds (x1, y1, x2, y2) with size >= 1."""
        if frame_w < 1 or frame_h < 1:
            raise ValueError("frame_w and frame_h must be >= 1")
        r = self.clamped()
        x1 = max(0, min(frame_w - 1, round(r.x * frame_w)))
        y1 = max(0, min(frame_h - 1, round(r.y * frame_h)))
        x2 = max(x1 + 1, min(frame_w, round((r.x + r.w) * frame_w)))
        y2 = max(y1 + 1, min(frame_h, round((r.y + r.h) * frame_h)))
        return (x1, y1, x2, y2)


@dataclass(slots=True)
class ExtractConfig:
    """Parameters controlling subtitle mask detection, segmentation, and OCR."""

    white_v_min: int = 185
    white_s_max: int = 65
    outline_v_max: int = 85
    min_blob_px: int = 12
    coarse_step: int = 2
    fg_ratio_min: float = 0.004
    fg_ratio_edge: float = 0.002
    split_iou_min: float = 0.55
    min_duration_ms: int = 120
    merge_gap_ms: int = 160
    merge_similarity: float = 0.85
    conf_threshold: float = 0.85
    provider: str = "cpu"


@dataclass(slots=True)
class RawSegment:
    """Detected subtitle time span before OCR."""

    start_ms: int
    end_ms: int
    peak_bgr: np.ndarray
    frame_index: int


@dataclass(slots=True)
class OcrSegment:
    """OCR result for a raw subtitle segment."""

    start_ms: int
    end_ms: int
    text: str
    confidence: float
    thumbnail_png: bytes


@dataclass(slots=True)
class SubtitleEvent:
    """Finalized subtitle event ready for review and export."""

    index: int
    start_ms: int
    end_ms: int
    text: str
    confidence: float
    flagged: bool
    thumbnail_png: bytes


@dataclass(slots=True)
class ExtractResult:
    """Summary and extracted subtitle events for a processed video."""

    video_path: Path
    width: int
    height: int
    duration_ms: int
    events: list[SubtitleEvent]
    elapsed_sec: float


@dataclass(slots=True)
class FrameTestResult:
    """Single-frame diagnostic result for ROI/threshold tuning."""

    frame_index: int
    pts_ms: int
    mask: np.ndarray
    text: str
    confidence: float
