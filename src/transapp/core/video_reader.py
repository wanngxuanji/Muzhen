"""PTS-accurate PyAV video reader with streaming ROI cropping."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import av.error
import numpy as np

from transapp.core.models import RoiRect


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Basic container and video stream properties."""

    width: int
    height: int
    duration_ms: int
    fps: float
    estimated_frames: int


@dataclass(slots=True)
class FrameSlice:
    """Single decoded ROI frame with monotonic presentation timestamp."""

    frame_index: int
    pts_ms: int
    bgr_roi: np.ndarray


class VideoReader:
    """Streaming PyAV reader yielding ROI crops and supporting millisecond seeks."""

    def __init__(self, video_path: Path | str) -> None:
        self._path = Path(video_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Video file not found: {self._path}")
        if not self._path.is_file():
            raise ValueError(f"Video path is not a file: {self._path}")

        try:
            container = av.open(str(self._path))
        except (av.error.FFmpegError, OSError) as exc:
            raise ValueError(f"Unable to open video file {self._path}: {exc}") from exc

        if not container.streams.video:
            container.close()
            raise ValueError(f"No video stream found in {self._path}")

        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        self._container: av.container.InputContainer | None = container
        self._stream: av.video.stream.VideoStream | None = stream
        self._start_pts_ms: int = self._compute_start_pts_ms(stream)
        self._metadata: VideoMetadata = self._build_metadata(container, stream)

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def metadata(self) -> VideoMetadata:
        """Return metadata for the opened video stream."""
        return self._metadata

    def close(self) -> None:
        """Close the underlying PyAV container (idempotent)."""
        if self._container is not None:
            try:
                self._container.close()
            finally:
                self._container = None
                self._stream = None

    def iter_roi_frames(self, roi: RoiRect) -> Iterator[FrameSlice]:
        """Stream decoded frames from the beginning, yielding only the cropped ROI."""
        container, stream = self._ensure_open()
        x1, y1, x2, y2 = roi.to_pixels(self._metadata.width, self._metadata.height)

        self._seek_to_start(container, stream)

        frame_index = 0
        prev_pts_ms: int | None = None
        try:
            for frame in container.decode(stream):
                pts_ms = self._frame_pts_ms(frame, stream, frame_index, prev_pts_ms)
                prev_pts_ms = pts_ms
                full_bgr = frame.to_ndarray(format="bgr24")
                # Copy only the ROI slice so the full-frame buffer can be freed immediately.
                bgr_roi = np.ascontiguousarray(full_bgr[y1:y2, x1:x2])
                yield FrameSlice(frame_index=frame_index, pts_ms=pts_ms, bgr_roi=bgr_roi)
                frame_index += 1
        except av.error.EOFError:
            return

    def read_frame_at_ms(self, target_ms: int) -> tuple[int, int, np.ndarray]:
        """Seek near target_ms and return (frame_index, pts_ms, full_bgr_frame)."""
        container, stream = self._ensure_open()
        clamped_ms = max(0, int(target_ms))

        result = self._decode_near_ms(container, stream, clamped_ms, seek_ms=clamped_ms)
        if result is None:
            # Seeking past EOF or mid-stream can yield zero frames; fall back near tail then start.
            tail_ms = max(0, self._start_pts_ms + self._metadata.duration_ms - 5000)
            if 0 < tail_ms < clamped_ms:
                result = self._decode_near_ms(container, stream, clamped_ms, seek_ms=tail_ms)
        if result is None:
            result = self._decode_near_ms(container, stream, clamped_ms, seek_ms=None)
        if result is None:
            raise ValueError(f"No decodable video frames found in {self._path}")
        return result

    def _ensure_open(self) -> tuple[av.container.InputContainer, av.video.stream.VideoStream]:
        if self._container is None or self._stream is None:
            raise ValueError("VideoReader is closed")
        return self._container, self._stream

    @staticmethod
    def _compute_start_pts_ms(stream: av.video.stream.VideoStream) -> int:
        if stream.start_time is not None and stream.time_base is not None:
            return max(0, int(round(stream.start_time * float(stream.time_base) * 1000.0)))
        return 0

    def _build_metadata(
        self,
        container: av.container.InputContainer,
        stream: av.video.stream.VideoStream,
    ) -> VideoMetadata:
        width = int(stream.codec_context.width or stream.width or 0)
        height = int(stream.codec_context.height or stream.height or 0)

        rate = stream.average_rate or stream.guessed_rate or stream.base_rate
        fps = float(rate) if rate is not None and float(rate) > 0.0 else 25.0

        duration_ms = 0
        if stream.duration is not None and stream.duration > 0 and stream.time_base is not None:
            duration_ms = max(0, int(round(stream.duration * float(stream.time_base) * 1000.0)))
        if duration_ms <= 0 and container.duration is not None and container.duration > 0:
            duration_ms = max(0, int(round(container.duration * 1000.0 / av.time_base)))

        need_dims = width <= 0 or height <= 0
        need_duration = duration_ms <= 0
        if need_dims or need_duration:
            first_dims, scanned_dur = self._probe_stream(
                container, stream, fps, probe_duration=need_duration
            )
            if need_dims:
                width, height = first_dims
            if need_duration:
                duration_ms = scanned_dur

        if width <= 0 or height <= 0:
            container.close()
            raise ValueError(f"Invalid video dimensions ({width}x{height}) in {self._path}")

        if stream.frames and stream.frames > 0:
            estimated_frames = int(stream.frames)
        elif duration_ms > 0:
            estimated_frames = max(1, int(round(duration_ms * fps / 1000.0)))
        else:
            estimated_frames = 0

        return VideoMetadata(
            width=width,
            height=height,
            duration_ms=duration_ms,
            fps=fps,
            estimated_frames=estimated_frames,
        )

    def _probe_stream(
        self,
        container: av.container.InputContainer,
        stream: av.video.stream.VideoStream,
        fps: float,
        *,
        probe_duration: bool = True,
    ) -> tuple[tuple[int, int], int]:
        """Fallback probe when container headers lack dimensions or duration."""
        self._seek_to_start(container, stream)
        width = 0
        height = 0
        first_pts: int | None = None
        last_pts: int | None = None
        count = 0
        try:
            for frame in container.decode(stream):
                if width == 0 or height == 0:
                    width = int(frame.width)
                    height = int(frame.height)
                if not probe_duration:
                    break
                pts_ms = self._raw_pts_ms(frame, stream, count, fps)
                if first_pts is None:
                    first_pts = pts_ms
                last_pts = pts_ms
                count += 1
        except av.error.FFmpegError:
            pass
        finally:
            self._seek_to_start(container, stream)

        if count == 0:
            return (width, height), 0
        span = (last_pts - first_pts) if (first_pts is not None and last_pts is not None) else 0
        step_ms = int(round(1000.0 / fps))
        return (width, height), max(1, span + step_ms)

    @staticmethod
    def _seek_to_start(
        container: av.container.InputContainer,
        stream: av.video.stream.VideoStream,
    ) -> None:
        try:
            container.seek(0, backward=True, any_frame=False)
        except av.error.FFmpegError:
            try:
                container.seek(0, backward=True, stream=stream)
            except av.error.FFmpegError:
                pass

    def _raw_pts_ms(
        self,
        frame: av.VideoFrame,
        stream: av.video.stream.VideoStream,
        frame_index: int,
        fps: float,
    ) -> int:
        if frame.pts is not None:
            tb = float(frame.time_base) if frame.time_base else float(stream.time_base)
            return max(0, int(round(frame.pts * tb * 1000.0)))
        if frame.dts is not None:
            tb = float(frame.time_base) if frame.time_base else float(stream.time_base)
            return max(0, int(round(frame.dts * tb * 1000.0)))
        return max(0, int(round(frame_index * 1000.0 / fps)))

    def _frame_pts_ms(
        self,
        frame: av.VideoFrame,
        stream: av.video.stream.VideoStream,
        frame_index: int,
        prev_pts_ms: int | None,
    ) -> int:
        pts_ms = self._raw_pts_ms(frame, stream, frame_index, self._metadata.fps)
        if prev_pts_ms is not None and pts_ms <= prev_pts_ms:
            pts_ms = prev_pts_ms + 1
        return pts_ms

    def _decode_near_ms(
        self,
        container: av.container.InputContainer,
        stream: av.video.stream.VideoStream,
        target_ms: int,
        *,
        seek_ms: int | None,
    ) -> tuple[int, int, np.ndarray] | None:
        if seek_ms is None or seek_ms <= 0:
            self._seek_to_start(container, stream)
        else:
            tb = float(stream.time_base) if stream.time_base else 1.0 / 1000.0
            seek_pts = int(round((seek_ms / 1000.0) / tb))
            try:
                container.seek(seek_pts, backward=True, any_frame=False, stream=stream)
            except av.error.FFmpegError:
                return None

        prev_entry: tuple[int, int, np.ndarray] | None = None
        base_index: int | None = None
        local_idx = 0
        prev_pts_ms: int | None = None

        try:
            for frame in container.decode(stream):
                pts_ms = self._frame_pts_ms(frame, stream, local_idx, prev_pts_ms)
                prev_pts_ms = pts_ms

                if base_index is None:
                    delta_ms = max(0, pts_ms - self._start_pts_ms)
                    base_index = 0 if delta_ms <= int(round(500.0 / self._metadata.fps)) else int(
                        round(delta_ms * self._metadata.fps / 1000.0)
                    )

                frame_index = base_index + local_idx
                local_idx += 1
                bgr = frame.to_ndarray(format="bgr24")
                curr_entry = (frame_index, pts_ms, bgr)

                if pts_ms >= target_ms:
                    if prev_entry is None:
                        return curr_entry
                    if abs(pts_ms - target_ms) <= abs(target_ms - prev_entry[1]):
                        return curr_entry
                    return prev_entry

                prev_entry = curr_entry
        except av.error.FFmpegError:
            pass

        return prev_entry
