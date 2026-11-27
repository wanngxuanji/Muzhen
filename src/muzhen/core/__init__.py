"""Core data models, ROI presets, and extraction pipeline."""

from muzhen.core.models import (
    ExtractConfig,
    ExtractionCancelled,
    ExtractResult,
    FrameTestResult,
    OcrSegment,
    RawSegment,
    RoiRect,
    SubtitleEvent,
)
from muzhen.core.ocr_engine import OcrEngine
from muzhen.core.pipeline import ExtractionPipeline
from muzhen.core.postprocess import (
    normalize_cjk_text,
    postprocess_segments,
    text_similarity,
)
from muzhen.core.presets import PresetStore, RoiPreset
from muzhen.core.segmenter import segment_roi_frames
from muzhen.core.subtitle_mask import (
    build_stroke_mask,
    frame_sharpness,
    mask_fg_ratio,
    mask_iou,
)
from muzhen.core.video_reader import FrameSlice, VideoMetadata, VideoReader

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
