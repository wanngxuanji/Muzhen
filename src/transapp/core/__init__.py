"""Core data models, ROI presets, and extraction pipeline."""

from transapp.core.models import (
    ExtractConfig,
    ExtractionCancelled,
    ExtractResult,
    FrameTestResult,
    OcrSegment,
    RawSegment,
    RoiRect,
    SubtitleEvent,
)
from transapp.core.ocr_engine import OcrEngine
from transapp.core.pipeline import ExtractionPipeline
from transapp.core.postprocess import (
    normalize_cjk_text,
    postprocess_segments,
    text_similarity,
)
from transapp.core.presets import PresetStore, RoiPreset
from transapp.core.segmenter import segment_roi_frames
from transapp.core.subtitle_mask import (
    build_stroke_mask,
    frame_sharpness,
    mask_fg_ratio,
    mask_iou,
)
from transapp.core.video_reader import FrameSlice, VideoMetadata, VideoReader

__all__ = [
    "ExtractConfig",
    "ExtractionCancelled",
    "ExtractionPipeline",
    "ExtractResult",
    "FrameSlice",
    "FrameTestResult",
    "OcrEngine",
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
    "normalize_cjk_text",
    "postprocess_segments",
    "segment_roi_frames",
    "text_similarity",
]
