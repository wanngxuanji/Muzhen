"""PTS-accurate PyAV video reader with streaming ROI cropping."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import av.error
import numpy as np

from muzhen.core.models import RoiRect

_SEEK_BACKWARD_MARGIN_MS = 500
_PREV_FRAME_SEEK_BACKOFFS_MS: tuple[int | None, ...] = (500, 2000, 5000, None)


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Basic container and video stream properties."""

    width: int
    height: int
    duration_ms: int
    fps: float
    estimated_frames: int
    start_pts_ms: int = 0


@dataclass(slots=True)
class FrameSlice:
    """Single decoded ROI frame with monotonic presentation timestamp."""

    frame_index: int
    pts_ms: int
    bgr_roi: np.ndarray
    rel_ms: int = -1

    def __post_init__(self) -> None:
        if self.rel_ms < 0:
            self.rel_ms = max(0, int(self.pts_ms))


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
        try:
            fallback_start_pts_ms = self._compute_start_pts_ms(stream)
            self._start_pts_ms: int = fallback_start_pts_ms
            self._first_pts_ticks: int | None = stream.start_time
            self._fps_frac: Fraction = self._extract_fps_fraction(stream)
            self._metadata: VideoMetadata = self._build_metadata(
                container, stream, fallback_start_pts_ms
            )
        except Exception:
            self.close()
            raise

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

    def iter_roi_frames(
        self,
        roi: RoiRect,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> Iterator[FrameSlice]:
        """Stream decoded frames (optionally bounded by 0-based rel_ms window), yielding ROI crops."""
        container, stream = self._ensure_open()
        x1, y1, x2, y2 = roi.to_pixels(self._metadata.width, self._metadata.height)

        norm_start_ms = max(0, int(start_ms)) if start_ms is not None else None
        norm_end_ms = int(end_ms) if end_ms is not None else None
        if norm_start_ms is not None and norm_end_ms is not None and norm_start_ms > norm_end_ms:
            return
        if norm_end_ms is not None and norm_end_ms < 0:
            return

        did_seek = False
        seek_raw_ms = self._start_pts_ms
        if norm_start_ms is not None and norm_start_ms > 0:
            target_raw_ms = self._start_pts_ms + norm_start_ms
            seek_raw_ms = max(
                self._metadata.start_pts_ms, target_raw_ms - _SEEK_BACKWARD_MARGIN_MS
            )
            if seek_raw_ms <= self._start_pts_ms:
                self._seek_to_start(
                    container,
                    stream,
                    start_pts_ms=self._start_pts_ms,
                    first_pts_ticks=self._first_pts_ticks,
                )
            else:
                tb = self._to_time_base_fraction(stream.time_base)
                seek_ticks = math.floor(Fraction(seek_raw_ms, 1000) / tb)
                try:
                    container.seek(seek_ticks, backward=True, any_frame=False, stream=stream)
                    did_seek = True
                except av.error.FFmpegError:
                    self._seek_to_start(
                        container,
                        stream,
                        start_pts_ms=self._start_pts_ms,
                        first_pts_ticks=self._first_pts_ticks,
                    )
        else:
            self._seek_to_start(
                container,
                stream,
                start_pts_ms=self._start_pts_ms,
                first_pts_ticks=self._first_pts_ticks,
            )

        anchor_pts_ms = seek_raw_ms if did_seek else self._start_pts_ms
        anchor_decoded_index = 0
        has_real_pts = False
        frame_index = 0
        decoded_index = 0
        prev_pts_ms: int | None = None
        try:
            for frame in container.decode(stream):
                pts_ms = self._frame_pts_ms(
                    frame,
                    stream,
                    decoded_index,
                    prev_pts_ms,
                    anchor_pts_ms=anchor_pts_ms,
                    anchor_index=anchor_decoded_index,
                    has_real_pts=has_real_pts,
                )
                if frame.pts is not None:
                    anchor_pts_ms = pts_ms
                    anchor_decoded_index = decoded_index
                    has_real_pts = True
                prev_pts_ms = pts_ms
                decoded_index += 1

                rel_ms = max(0, pts_ms - self._start_pts_ms)
                if norm_start_ms is not None and rel_ms < norm_start_ms:
                    continue
                if norm_end_ms is not None and rel_ms > norm_end_ms:
                    break

                full_bgr = frame.to_ndarray(format="bgr24")
                bgr_roi = np.ascontiguousarray(full_bgr[y1:y2, x1:x2])
                yield FrameSlice(
                    frame_index=frame_index,
                    pts_ms=pts_ms,
                    bgr_roi=bgr_roi,
                    rel_ms=rel_ms,
                )
                frame_index += 1
        except av.error.EOFError:
            return

    def read_frame_at_ms(self, target_ms: int) -> tuple[int, int, np.ndarray]:
        """Seek near target_ms and return (frame_index, pts_ms, full_bgr_frame)."""
        container, stream = self._ensure_open()
        min_target_ms = min(0, self._start_pts_ms)
        clamped_ms = max(min_target_ms, int(target_ms))

        result = self._decode_near_ms(container, stream, clamped_ms, seek_ms=clamped_ms)
        if result is None:
            tail_ms = max(self._start_pts_ms, self._start_pts_ms + self._metadata.duration_ms - 5000)
            if self._start_pts_ms < tail_ms < clamped_ms:
                result = self._decode_near_ms(container, stream, clamped_ms, seek_ms=tail_ms)
        if result is None:
            result = self._decode_near_ms(container, stream, clamped_ms, seek_ms=None)
        if result is None:
            raise ValueError(f"No decodable video frames found in {self._path}")
        return result

    def read_prev_frame_before_ms(self, current_pts_ms: int) -> tuple[np.ndarray, int]:
        """Return (bgr_frame, pts_ms) for the latest frame strictly before current_pts_ms, or frame 0."""
        container, stream = self._ensure_open()
        target_ms = int(current_pts_ms)
        if target_ms <= self._start_pts_ms:
            _, first_pts_ms, first_bgr = self.read_frame_at_ms(self._start_pts_ms)
            return (first_bgr, first_pts_ms)

        tb = self._to_time_base_fraction(stream.time_base)
        for offset_ms in _PREV_FRAME_SEEK_BACKOFFS_MS:
            did_seek = False
            at_start = False
            if offset_ms is None:
                seek_raw_ms = self._start_pts_ms
                at_start = True
                self._seek_to_start(
                    container,
                    stream,
                    start_pts_ms=self._start_pts_ms,
                    first_pts_ticks=self._first_pts_ticks,
                )
            else:
                seek_raw_ms = max(self._metadata.start_pts_ms, target_ms - offset_ms)
                if seek_raw_ms <= self._start_pts_ms:
                    at_start = True
                    self._seek_to_start(
                        container,
                        stream,
                        start_pts_ms=self._start_pts_ms,
                        first_pts_ticks=self._first_pts_ticks,
                    )
                else:
                    seek_ticks = math.floor(Fraction(seek_raw_ms, 1000) / tb)
                    try:
                        container.seek(seek_ticks, backward=True, any_frame=False, stream=stream)
                        did_seek = True
                    except av.error.FFmpegError:
                        continue

            anchor_pts_ms = seek_raw_ms if did_seek else self._start_pts_ms
            anchor_idx = 0
            has_real_pts = False
            local_idx = 0
            prev_pts_ms: int | None = None
            best_before: tuple[int, av.VideoFrame] | None = None
            first_seen: tuple[int, av.VideoFrame] | None = None

            try:
                for frame in container.decode(stream):
                    pts_ms = self._frame_pts_ms(
                        frame,
                        stream,
                        local_idx,
                        prev_pts_ms,
                        anchor_pts_ms=anchor_pts_ms,
                        anchor_index=anchor_idx,
                        has_real_pts=has_real_pts,
                    )
                    if frame.pts is not None:
                        anchor_pts_ms = pts_ms
                        anchor_idx = local_idx
                        has_real_pts = True
                    prev_pts_ms = pts_ms
                    local_idx += 1

                    if first_seen is None:
                        first_seen = (pts_ms, frame)

                    if pts_ms < target_ms:
                        best_before = (pts_ms, frame)
                    else:
                        break
            except av.error.FFmpegError:
                pass

            if best_before is not None:
                chosen_pts, chosen_frame = best_before
                return (chosen_frame.to_ndarray(format="bgr24"), chosen_pts)

            if at_start:
                if first_seen is not None:
                    first_pts, first_frame = first_seen
                    return (first_frame.to_ndarray(format="bgr24"), first_pts)
                break

        raise ValueError(f"No decodable video frames found in {self._path}")

    def iter_full_frames_from_ms(
        self, start_raw_ms: int
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        """Stream sequential full BGR frames starting at or after ``start_raw_ms`` (no per-frame seek)."""
        container, stream = self._ensure_open()
        min_target_ms = min(0, self._start_pts_ms)
        clamped_ms = max(min_target_ms, int(start_raw_ms))
        seek_raw_ms = max(self._metadata.start_pts_ms, clamped_ms - _SEEK_BACKWARD_MARGIN_MS)

        did_seek = False
        if seek_raw_ms <= self._start_pts_ms:
            self._seek_to_start(
                container,
                stream,
                start_pts_ms=self._start_pts_ms,
                first_pts_ticks=self._first_pts_ticks,
            )
        else:
            tb = self._to_time_base_fraction(stream.time_base)
            seek_ticks = math.floor(Fraction(seek_raw_ms, 1000) / tb)
            try:
                container.seek(seek_ticks, backward=True, any_frame=False, stream=stream)
                did_seek = True
            except av.error.FFmpegError:
                self._seek_to_start(
                    container,
                    stream,
                    start_pts_ms=self._start_pts_ms,
                    first_pts_ticks=self._first_pts_ticks,
                )

        anchor_pts_ms = seek_raw_ms if did_seek else self._start_pts_ms
        anchor_decoded_index = 0
        has_real_pts = False
        decoded_index = 0
        prev_pts_ms: int | None = None
        try:
            for frame in container.decode(stream):
                pts_ms = self._frame_pts_ms(
                    frame,
                    stream,
                    decoded_index,
                    prev_pts_ms,
                    anchor_pts_ms=anchor_pts_ms,
                    anchor_index=anchor_decoded_index,
                    has_real_pts=has_real_pts,
                )
                if frame.pts is not None:
                    anchor_pts_ms = pts_ms
                    anchor_decoded_index = decoded_index
                    has_real_pts = True
                prev_pts_ms = pts_ms
                decoded_index += 1

                if pts_ms < clamped_ms:
                    continue

                rel_ms = max(0, pts_ms - self._start_pts_ms)
                est_idx = int(round(Fraction(rel_ms) * self._fps_frac / 1000))
                full_bgr = frame.to_ndarray(format="bgr24")
                yield (est_idx, pts_ms, full_bgr)
        except av.error.EOFError:
            return

    def _ensure_open(self) -> tuple[av.container.InputContainer, av.video.stream.VideoStream]:
        if self._container is None or self._stream is None:
            raise ValueError("VideoReader is closed")
        return self._container, self._stream

    @staticmethod
    def _to_time_base_fraction(time_base: Fraction | float | None) -> Fraction:
        if isinstance(time_base, Fraction) and time_base > 0:
            return time_base
        if time_base:
            frac = Fraction(time_base).limit_denominator(1_000_000)
            if frac > 0:
                return frac
        return Fraction(1, 1000)

    @staticmethod
    def _ticks_to_ms(ticks: int, time_base: Fraction | float | None) -> int:
        tb = VideoReader._to_time_base_fraction(time_base)
        return int(round(Fraction(ticks) * tb * 1000))

    @staticmethod
    def _extract_fps_fraction(stream: av.video.stream.VideoStream) -> Fraction:
        rate = stream.average_rate or stream.guessed_rate or stream.base_rate
        if isinstance(rate, Fraction) and rate > 0:
            return rate
        if rate is not None and float(rate) > 0.0:
            return Fraction(float(rate)).limit_denominator(100_000)
        return Fraction(25, 1)

    @staticmethod
    def _compute_start_pts_ms(stream: av.video.stream.VideoStream) -> int:
        if stream.start_time is not None and stream.time_base:
            return VideoReader._ticks_to_ms(stream.start_time, stream.time_base)
        return 0

    def _build_metadata(
        self,
        container: av.container.InputContainer,
        stream: av.video.stream.VideoStream,
        fallback_start_pts_ms: int,
    ) -> VideoMetadata:
        width = int(stream.codec_context.width or stream.width or 0)
        height = int(stream.codec_context.height or stream.height or 0)
        fps = float(self._fps_frac)

        first_dims, first_pts_ticks, first_pts_ms = self._probe_first_frame(container, stream)
        origin_known = first_pts_ms is not None
        if first_pts_ms is not None:
            self._start_pts_ms = first_pts_ms
            self._first_pts_ticks = first_pts_ticks
        else:
            self._start_pts_ms = fallback_start_pts_ms
            self._first_pts_ticks = stream.start_time

        if width <= 0 or height <= 0:
            width, height = first_dims

        duration_ms = 0
        if stream.duration is not None and stream.duration > 0 and stream.time_base is not None:
            duration_ms = max(0, self._ticks_to_ms(stream.duration, stream.time_base))
        if duration_ms <= 0 and container.duration is not None and container.duration > 0:
            duration_ms = max(0, int(round(Fraction(container.duration * 1000, av.time_base))))

        if duration_ms <= 0:
            duration_ms = self._probe_duration_via_demux(
                container, stream, self._fps_frac, origin_known=origin_known
            )

        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid video dimensions ({width}x{height}) in {self._path}")

        if stream.frames and stream.frames > 0:
            estimated_frames = int(stream.frames)
        elif duration_ms > 0:
            estimated_frames = max(1, int(round(Fraction(duration_ms) * self._fps_frac / 1000)))
        else:
            estimated_frames = 0

        return VideoMetadata(
            width=width,
            height=height,
            duration_ms=duration_ms,
            fps=fps,
            estimated_frames=estimated_frames,
            start_pts_ms=self._start_pts_ms,
        )

    def _probe_first_frame(
        self,
        container: av.container.InputContainer,
        stream: av.video.stream.VideoStream,
    ) -> tuple[tuple[int, int], int | None, int | None]:
        """Decode only frame 0 to anchor signed origin_pts_ms and recover missing dimensions."""
        width = 0
        height = 0
        first_pts_ticks: int | None = None
        first_pts_ms: int | None = None
        try:
            for frame in container.decode(stream):
                width = int(frame.width)
                height = int(frame.height)
                if frame.pts is not None:
                    first_pts_ticks = int(frame.pts)
                    first_pts_ms = self._ticks_to_ms(frame.pts, frame.time_base or stream.time_base)
                break
        except av.error.FFmpegError:
            pass
        finally:
            self._seek_to_start(
                container,
                stream,
                start_pts_ms=first_pts_ms if first_pts_ms is not None else self._start_pts_ms,
                first_pts_ticks=first_pts_ticks if first_pts_ticks is not None else stream.start_time,
            )
        return (width, height), first_pts_ticks, first_pts_ms

    def _probe_duration_via_demux(
        self,
        container: av.container.InputContainer,
        stream: av.video.stream.VideoStream,
        fps_frac: Fraction,
        *,
        origin_known: bool = True,
    ) -> int:
        """Estimate stream duration via packet demuxing without decoding every frame."""
        self._seek_to_start(
            container,
            stream,
            start_pts_ms=self._start_pts_ms,
            first_pts_ticks=self._first_pts_ticks,
        )
        step_ms = max(1, int(round(Fraction(1000) / fps_frac)))
        min_pts_ms: int | None = None
        max_end_ms: int | None = None
        try:
            for packet in container.demux(stream):
                if packet.pts is None:
                    continue
                tb = packet.time_base or stream.time_base
                pts_ms = self._ticks_to_ms(packet.pts, tb)
                if packet.duration is not None and packet.duration > 0:
                    end_ms = self._ticks_to_ms(packet.pts + packet.duration, tb)
                else:
                    end_ms = pts_ms + step_ms
                if min_pts_ms is None or pts_ms < min_pts_ms:
                    min_pts_ms = pts_ms
                if max_end_ms is None or end_ms > max_end_ms:
                    max_end_ms = end_ms
        except av.error.FFmpegError:
            pass
        finally:
            self._seek_to_start(
                container,
                stream,
                start_pts_ms=self._start_pts_ms,
                first_pts_ticks=self._first_pts_ticks,
            )

        if min_pts_ms is not None and max_end_ms is not None:
            origin = self._start_pts_ms if origin_known else min_pts_ms
            return max(1, max_end_ms - origin)

        return self._probe_duration_via_decode(container, stream, fps_frac)

    def _probe_duration_via_decode(
        self,
        container: av.container.InputContainer,
        stream: av.video.stream.VideoStream,
        fps_frac: Fraction,
    ) -> int:
        """Last-resort duration probe when container headers and packets both lack PTS."""
        self._seek_to_start(
            container,
            stream,
            start_pts_ms=self._start_pts_ms,
            first_pts_ticks=self._first_pts_ticks,
        )
        first_pts: int | None = None
        last_pts: int | None = None
        prev_pts: int | None = None
        anchor_pts_ms = self._start_pts_ms
        anchor_index = 0
        has_real_pts = False
        count = 0
        try:
            for frame in container.decode(stream):
                pts_ms = self._frame_pts_ms(
                    frame,
                    stream,
                    count,
                    prev_pts,
                    anchor_pts_ms=anchor_pts_ms,
                    anchor_index=anchor_index,
                    has_real_pts=has_real_pts,
                )
                if frame.pts is not None:
                    anchor_pts_ms = pts_ms
                    anchor_index = count
                    has_real_pts = True
                prev_pts = pts_ms
                if first_pts is None:
                    first_pts = pts_ms
                last_pts = pts_ms
                count += 1
        except av.error.FFmpegError:
            pass
        finally:
            self._seek_to_start(
                container,
                stream,
                start_pts_ms=self._start_pts_ms,
                first_pts_ticks=self._first_pts_ticks,
            )

        if count == 0:
            return 0
        span = (last_pts - first_pts) if (first_pts is not None and last_pts is not None) else 0
        step_ms = max(1, int(round(Fraction(1000) / fps_frac)))
        return max(1, span + step_ms)

    @staticmethod
    def _seek_to_start(
        container: av.container.InputContainer,
        stream: av.video.stream.VideoStream,
        *,
        start_pts_ms: int = 0,
        first_pts_ticks: int | None = None,
    ) -> None:
        target_us = min(0, int(start_pts_ms) * 1000)
        try:
            container.seek(target_us, backward=True, any_frame=False)
        except av.error.FFmpegError:
            target_ticks = min(0, int(first_pts_ticks)) if first_pts_ticks is not None else 0
            try:
                container.seek(target_ticks, backward=True, stream=stream)
            except av.error.FFmpegError:
                if target_us < 0:
                    try:
                        container.seek(0, backward=True, any_frame=False)
                    except av.error.FFmpegError:
                        pass

    def _frame_pts_ms(
        self,
        frame: av.VideoFrame,
        stream: av.video.stream.VideoStream,
        frame_index: int,
        prev_pts_ms: int | None,
        *,
        anchor_pts_ms: int | None = None,
        anchor_index: int = 0,
        has_real_pts: bool = True,
    ) -> int:
        if frame.pts is not None:
            pts_ms = self._ticks_to_ms(frame.pts, frame.time_base or stream.time_base)
            if has_real_pts and prev_pts_ms is not None and pts_ms <= prev_pts_ms:
                pts_ms = prev_pts_ms + 1
            return pts_ms

        base_pts = self._start_pts_ms if anchor_pts_ms is None else anchor_pts_ms
        delta_frames = frame_index - anchor_index
        pts_ms = base_pts + int(round(Fraction(delta_frames * 1000) / self._fps_frac))
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
        did_seek = False
        seek_raw_ms = self._start_pts_ms
        if seek_ms is None:
            self._seek_to_start(
                container,
                stream,
                start_pts_ms=self._start_pts_ms,
                first_pts_ticks=self._first_pts_ticks,
            )
        else:
            seek_raw_ms = max(
                self._metadata.start_pts_ms, int(seek_ms) - _SEEK_BACKWARD_MARGIN_MS
            )
            if seek_raw_ms <= self._start_pts_ms:
                self._seek_to_start(
                    container,
                    stream,
                    start_pts_ms=self._start_pts_ms,
                    first_pts_ticks=self._first_pts_ticks,
                )
            else:
                tb = self._to_time_base_fraction(stream.time_base)
                seek_pts = math.floor(Fraction(seek_raw_ms, 1000) / tb)
                try:
                    container.seek(seek_pts, backward=True, any_frame=False, stream=stream)
                    did_seek = True
                except av.error.FFmpegError:
                    return None

        prev_entry: tuple[int, int, av.VideoFrame] | None = None
        chosen_entry: tuple[int, int, av.VideoFrame] | None = None
        base_index: int | None = None
        anchor_pts_ms = seek_raw_ms if did_seek else self._start_pts_ms
        anchor_idx = 0
        has_real_pts = False
        local_idx = 0
        prev_pts_ms: int | None = None

        try:
            for frame in container.decode(stream):
                pts_ms = self._frame_pts_ms(
                    frame,
                    stream,
                    local_idx,
                    prev_pts_ms,
                    anchor_pts_ms=anchor_pts_ms,
                    anchor_index=anchor_idx,
                    has_real_pts=has_real_pts,
                )
                if frame.pts is not None:
                    anchor_pts_ms = pts_ms
                    anchor_idx = local_idx
                    has_real_pts = True
                prev_pts_ms = pts_ms

                if base_index is None:
                    delta_ms = max(0, pts_ms - self._start_pts_ms)
                    half_frame_ms = int(round(Fraction(500) / self._fps_frac))
                    base_index = (
                        0
                        if delta_ms <= half_frame_ms
                        else int(round(Fraction(delta_ms) * self._fps_frac / 1000))
                    )

                frame_index = base_index + local_idx
                local_idx += 1
                curr_entry = (frame_index, pts_ms, frame)

                if pts_ms >= target_ms:
                    if prev_entry is None:
                        chosen_entry = curr_entry
                    elif abs(pts_ms - target_ms) <= abs(target_ms - prev_entry[1]):
                        chosen_entry = curr_entry
                    else:
                        chosen_entry = prev_entry
                    break

                prev_entry = curr_entry
        except av.error.FFmpegError:
            pass

        if chosen_entry is None:
            chosen_entry = prev_entry
        if chosen_entry is None:
            return None

        chosen_idx, chosen_pts_ms, chosen_frame = chosen_entry
        return (chosen_idx, chosen_pts_ms, chosen_frame.to_ndarray(format="bgr24"))
