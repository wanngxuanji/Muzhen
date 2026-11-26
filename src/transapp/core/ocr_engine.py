"""RapidOCR Chinese text recognition wrapper and PNG thumbnail encoder."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import cv2
import numpy as np

from transapp.core.models import ExtractionCancelled, OcrSegment, RawSegment

logger = logging.getLogger(__name__)

_MIN_OCR_HEIGHT = 48
_BORDER_PAD_PX = 10
_SUPPORTED_PROVIDERS = ("cpu", "directml")


@dataclass(frozen=True, slots=True)
class _OcrBox:
    cx: float
    cy: float
    height: float
    text: str
    score: float


def _prepare_roi_for_ocr(bgr_roi: np.ndarray) -> np.ndarray:
    """Upscale and pad small ROIs (<48px tall) so text detection/recognition remain accurate."""
    h, w = bgr_roi.shape[:2]
    if h >= _MIN_OCR_HEIGHT:
        return bgr_roi

    scale = _MIN_OCR_HEIGHT / float(h)
    new_w = max(1, round(w * scale))
    new_h = max(_MIN_OCR_HEIGHT, round(h * scale))
    resized = cv2.resize(bgr_roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(
        resized,
        _BORDER_PAD_PX,
        _BORDER_PAD_PX,
        _BORDER_PAD_PX,
        _BORDER_PAD_PX,
        cv2.BORDER_REPLICATE,
    )


def _group_and_format_boxes(raw_output: Any) -> tuple[str, float]:
    """Group OCR boxes into horizontal lines, sort top-to-bottom/left-to-right, and compute min score."""
    if raw_output is None:
        return ("", 0.0)

    items = raw_output[0] if isinstance(raw_output, tuple) and len(raw_output) == 2 else raw_output
    if not items:
        return ("", 0.0)

    boxes: list[_OcrBox] = []
    for item in items:
        if not item or len(item) < 3:
            continue
        coords, raw_text, raw_score = item[0], item[1], item[2]
        text = str(raw_text).strip()
        if not text or not coords:
            continue
        xs = [float(pt[0]) for pt in coords]
        ys = [float(pt[1]) for pt in coords]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        height = max(1.0, max(ys) - min(ys))
        boxes.append(_OcrBox(cx=cx, cy=cy, height=height, text=text, score=float(raw_score)))

    if not boxes:
        return ("", 0.0)

    median_height = float(np.median([box.height for box in boxes]))
    line_tol = 0.5 * median_height

    lines: list[list[_OcrBox]] = []
    line_centers: list[float] = []
    for box in sorted(boxes, key=lambda b: b.cy):
        if lines and abs(box.cy - line_centers[-1]) <= line_tol:
            current = lines[-1]
            current.append(box)
            line_centers[-1] = sum(b.cy for b in current) / len(current)
        else:
            lines.append([box])
            line_centers.append(box.cy)

    ordered_lines = [
        " ".join(b.text for b in sorted(line, key=lambda b: b.cx))
        for _, line in sorted(zip(line_centers, lines), key=lambda pair: pair[0])
    ]
    confidence = max(0.0, min(1.0, float(min(box.score for box in boxes))))
    return ("\n".join(ordered_lines), confidence)


class OcrEngine:
    """Lazy RapidOCR wrapper with DirectML-to-CPU fallback and PNG thumbnail encoding."""

    def __init__(
        self,
        provider: str = "cpu",
        ocr_callable: Callable[[np.ndarray], Any] | None = None,
    ) -> None:
        normalized = (provider or "cpu").strip().lower()
        if normalized not in _SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported OCR provider: {provider!r}")
        self.provider = normalized
        self._ocr_callable = ocr_callable
        self._engine: Callable[[np.ndarray], Any] | None = ocr_callable

    @staticmethod
    def available_providers() -> list[str]:
        """Return supported TransApp OCR providers available in the current ONNX Runtime."""
        try:
            import onnxruntime

            ort_providers = set(onnxruntime.get_available_providers())
        except Exception:
            return []

        providers: list[str] = []
        if "CPUExecutionProvider" in ort_providers:
            providers.append("cpu")
        if "DmlExecutionProvider" in ort_providers:
            providers.append("directml")
        return providers

    def _get_engine(self) -> Callable[[np.ndarray], Any]:
        if self._engine is not None:
            return self._engine

        import rapidocr_onnxruntime

        if self.provider == "directml":
            if "directml" not in self.available_providers():
                logger.warning(
                    "DirectML execution provider is unavailable; falling back to CPU OCR."
                )
                self._engine = rapidocr_onnxruntime.RapidOCR()
                return self._engine
            try:
                self._engine = rapidocr_onnxruntime.RapidOCR(
                    det_use_dml=True,
                    cls_use_dml=True,
                    rec_use_dml=True,
                )
                return self._engine
            except Exception as exc:
                logger.warning(
                    "Failed to initialize RapidOCR with DirectML (%s); falling back to CPU OCR.",
                    exc,
                )

        self._engine = rapidocr_onnxruntime.RapidOCR()
        return self._engine

    def recognize_roi(self, bgr_roi: np.ndarray) -> tuple[str, float]:
        """Run OCR on a cropped BGR ROI and return (text, confidence in [0.0, 1.0])."""
        if bgr_roi is None or not isinstance(bgr_roi, np.ndarray) or bgr_roi.size == 0 or bgr_roi.ndim < 2:
            return ("", 0.0)

        prepared = _prepare_roi_for_ocr(bgr_roi)
        engine = self._get_engine()
        raw_output = engine(prepared)
        return _group_and_format_boxes(raw_output)

    def encode_thumbnail_png(self, bgr_roi: np.ndarray) -> bytes:
        """Encode a BGR ROI image as in-memory PNG bytes."""
        if (
            bgr_roi is None
            or not isinstance(bgr_roi, np.ndarray)
            or bgr_roi.size == 0
            or bgr_roi.ndim not in (2, 3)
        ):
            raise ValueError("bgr_roi must be a non-empty 2D or 3D numpy array")

        ok, buffer = cv2.imencode(".png", bgr_roi)
        if not ok:
            raise ValueError("Failed to encode ROI thumbnail as PNG")
        return buffer.tobytes()

    def process_segments(
        self,
        segments: list[RawSegment],
        progress_cb: Callable[[int, int], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> list[OcrSegment]:
        """Recognize text and encode PNG thumbnails for each detected raw segment."""
        total = len(segments)
        results: list[OcrSegment] = []

        for idx, segment in enumerate(segments):
            if cancel_check is not None and cancel_check():
                raise ExtractionCancelled("OCR processing cancelled")

            text, confidence = self.recognize_roi(segment.peak_bgr)
            thumbnail_png = self.encode_thumbnail_png(segment.peak_bgr)
            results.append(
                OcrSegment(
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    text=text,
                    confidence=confidence,
                    thumbnail_png=thumbnail_png,
                )
            )
            if progress_cb is not None:
                progress_cb(idx + 1, total)

        return results
