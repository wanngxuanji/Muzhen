"""Core data models, ROI presets, and extraction pipeline."""

from transapp.core.models import (
    ExtractConfig,
    ExtractResult,
    FrameTestResult,
    OcrSegment,
    RawSegment,
    RoiRect,
    SubtitleEvent,
)
from transapp.core.presets import PresetStore, RoiPreset
from transapp.core.subtitle_mask import (
    build_stroke_mask,
    frame_sharpness,
    mask_fg_ratio,
    mask_iou,
)
from transapp.core.video_reader import FrameSlice, VideoMetadata, VideoReader

__all__ = [
    "ExtractConfig",
    "ExtractResult",
    "FrameSlice",
    "FrameTestResult",
    "OcrSegment",
    "PresetStore",
    "RawSegment",
    "RoiPreset",
    "RoiRect",
    "SubtitleEvent",
    "VideoMetadata",
    "VideoReader",
    "build_stroke_mask",
    "frame_sharpness",
    "mask_fg_ratio",
    "mask_iou",
]
