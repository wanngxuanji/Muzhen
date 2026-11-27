"""Synchronized PyAV + QtMultimedia PCM audio preview player."""

from __future__ import annotations

from collections import deque
from fractions import Fraction
import math
from pathlib import Path
import threading
import time

import av
import av.error
from PySide6.QtCore import QIODevice, QObject, QTimer, Slot
from PySide6.QtMultimedia import QAudioDevice, QAudioFormat, QAudioSink, QMediaDevices

_SEEK_BACKWARD_MARGIN_MS = 500
_FEED_INTERVAL_MS = 15
_MAX_QUEUE_SECONDS = 1.5
_BUFFER_SECONDS = 0.25
_MAX_PREPEND_SILENCE_MS = 500
_PREBUFFER_WAIT_SEC = 0.08


def _select_audio_format(
    device: QAudioDevice,
) -> tuple[QAudioFormat, str, str, int, int, int] | None:
    """Choose a device-supported QAudioFormat and matching PyAV resampler parameters."""
    candidates: list[tuple[int, int, QAudioFormat.SampleFormat, str, str, int]] = [
        (48000, 2, QAudioFormat.SampleFormat.Int16, "s16", "stereo", 2),
        (44100, 2, QAudioFormat.SampleFormat.Int16, "s16", "stereo", 2),
        (48000, 2, QAudioFormat.SampleFormat.Float, "flt", "stereo", 4),
        (44100, 2, QAudioFormat.SampleFormat.Float, "flt", "stereo", 4),
        (48000, 1, QAudioFormat.SampleFormat.Int16, "s16", "mono", 2),
        (44100, 1, QAudioFormat.SampleFormat.Int16, "s16", "mono", 2),
    ]
    for rate, channels, qt_fmt, av_fmt, av_layout, bps in candidates:
        fmt = QAudioFormat()
        fmt.setSampleRate(rate)
        fmt.setChannelCount(channels)
        fmt.setSampleFormat(qt_fmt)
        if device.isFormatSupported(fmt):
            return (fmt, av_fmt, av_layout, rate, channels, bps)

    pref = device.preferredFormat()
    if not pref.isValid():
        return None
    rate = int(pref.sampleRate())
    channels = int(pref.channelCount())
    qt_fmt = pref.sampleFormat()
    if rate <= 0 or channels not in (1, 2):
        return None
    if qt_fmt == QAudioFormat.SampleFormat.Int16:
        av_fmt, bps = "s16", 2
    elif qt_fmt == QAudioFormat.SampleFormat.Float:
        av_fmt, bps = "flt", 4
    else:
        return None
    av_layout = "stereo" if channels == 2 else "mono"
    return (pref, av_fmt, av_layout, rate, channels, bps)


class AudioPreviewPlayer(QObject):
    """Decodes container audio with PyAV and plays it through QAudioSink with A/V clock."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._muted: bool = False
        self._active: bool = False
        self._sink: QAudioSink | None = None
        self._io_device: QIODevice | None = None
        self._feed_timer = QTimer(self)
        self._feed_timer.setInterval(_FEED_INTERVAL_MS)
        self._feed_timer.timeout.connect(self._feed_audio)

        self._cond = threading.Condition()
        self._queue: deque[bytes] = deque()
        self._queued_bytes: int = 0
        self._max_queue_bytes: int = 0
        self._pending_chunk: bytes = b""
        self._stop_event = threading.Event()
        self._first_chunk_event = threading.Event()
        self._decoder_thread: threading.Thread | None = None
        self._decoder_eof: bool = False

        self._audio_origin_ms: int = 0
        self._wall_start: float = 0.0
        self._eof_drain_pts_ms: int | None = None
        self._eof_drain_wall: float = 0.0
        self._bytes_per_sec: int = 192000

    @property
    def is_active(self) -> bool:
        """Return True while audio playback is actively streaming."""
        return self._active

    def is_muted(self) -> bool:
        """Return True when audio output is muted."""
        return self._muted

    def set_muted(self, muted: bool) -> None:
        """Mute or unmute live audio output without interrupting playback."""
        self._muted = bool(muted)
        if self._sink is not None:
            self._sink.setVolume(0.0 if self._muted else 1.0)

    def start(self, video_path: Path | str, start_raw_ms: int) -> bool:
        """Start streaming audio from ``start_raw_ms``; return False if unavailable."""
        self.stop()

        device = QMediaDevices.defaultAudioOutput()
        if device.isNull():
            return False

        selected = _select_audio_format(device)
        if selected is None:
            return False
        qt_fmt, av_fmt, av_layout, sample_rate, channels, bytes_per_sample = selected

        path = Path(video_path)
        try:
            container = av.open(str(path))
        except (av.error.FFmpegError, OSError):
            return False

        if not container.streams.audio:
            container.close()
            return False

        audio_stream = container.streams.audio[0]
        bytes_per_frame = channels * bytes_per_sample
        bytes_per_sec = sample_rate * bytes_per_frame
        self._bytes_per_sec = bytes_per_sec
        self._max_queue_bytes = max(bytes_per_frame * 1024, int(bytes_per_sec * _MAX_QUEUE_SECONDS))

        self._stop_event = threading.Event()
        self._first_chunk_event = threading.Event()
        with self._cond:
            self._queue.clear()
            self._queued_bytes = 0
            self._decoder_eof = False
        self._pending_chunk = b""
        self._audio_origin_ms = int(start_raw_ms)
        self._eof_drain_pts_ms = None
        self._eof_drain_wall = 0.0

        thread = threading.Thread(
            target=self._decode_worker,
            args=(
                container,
                audio_stream,
                int(start_raw_ms),
                av_fmt,
                av_layout,
                sample_rate,
                bytes_per_frame,
                self._stop_event,
                self._first_chunk_event,
            ),
            name="muzhen-audio-decode",
            daemon=True,
        )
        self._decoder_thread = thread
        try:
            thread.start()
        except Exception:
            container.close()
            self._decoder_thread = None
            return False

        self._first_chunk_event.wait(timeout=_PREBUFFER_WAIT_SEC)

        with self._cond:
            empty_and_done = self._decoder_eof and self._queued_bytes == 0
        if empty_and_done:
            self.stop()
            return False

        try:
            sink = QAudioSink(device, qt_fmt, self)
            sink.setBufferSize(max(bytes_per_frame * 512, int(bytes_per_sec * _BUFFER_SECONDS)))
            sink.setVolume(0.0 if self._muted else 1.0)
            io_dev = sink.start()
        except Exception:
            self.stop()
            return False

        if io_dev is None:
            sink.stop()
            sink.deleteLater()
            self.stop()
            return False

        self._sink = sink
        self._io_device = io_dev
        self._active = True
        self._wall_start = time.monotonic()
        self._feed_audio()
        self._feed_timer.start()
        return True

    def stop(self) -> None:
        """Stop audio output, join decoder thread, and release container handle."""
        if self._feed_timer.isActive():
            self._feed_timer.stop()

        self._active = False
        self._stop_event.set()
        with self._cond:
            self._cond.notify_all()

        sink = self._sink
        self._sink = None
        self._io_device = None
        if sink is not None:
            try:
                sink.stop()
            except RuntimeError:
                pass
            sink.deleteLater()

        thread = self._decoder_thread
        self._decoder_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.5)

        with self._cond:
            self._queue.clear()
            self._queued_bytes = 0
            self._decoder_eof = False
        self._pending_chunk = b""
        self._eof_drain_pts_ms = None

    def playback_pts_ms(self) -> int | None:
        """Return master audio playback timestamp in raw ms, or None when inactive."""
        if not self._active or self._sink is None:
            return None
        if self._eof_drain_pts_ms is not None:
            elapsed_ms = int((time.monotonic() - self._eof_drain_wall) * 1000.0)
            return self._eof_drain_pts_ms + max(0, elapsed_ms)
        processed_us = int(self._sink.processedUSecs())
        if processed_us > 0:
            return self._audio_origin_ms + (processed_us // 1000)
        elapsed_ms = int((time.monotonic() - self._wall_start) * 1000.0)
        return self._audio_origin_ms + max(0, elapsed_ms)

    @Slot()
    def _feed_audio(self) -> None:
        if not self._active or self._sink is None or self._io_device is None:
            return
        free = int(self._sink.bytesFree())
        if free <= 0:
            return

        to_write = bytearray()
        if self._pending_chunk:
            take = min(free, len(self._pending_chunk))
            to_write.extend(self._pending_chunk[:take])
            self._pending_chunk = self._pending_chunk[take:]
            free -= take

        if free > 0:
            with self._cond:
                while free > 0 and self._queue:
                    chunk = self._queue.popleft()
                    self._queued_bytes = max(0, self._queued_bytes - len(chunk))
                    if len(chunk) <= free:
                        to_write.extend(chunk)
                        free -= len(chunk)
                    else:
                        to_write.extend(chunk[:free])
                        self._pending_chunk = chunk[free:]
                        free = 0
                if self._queued_bytes < self._max_queue_bytes:
                    self._cond.notify_all()
                drained = self._decoder_eof and not self._queue and not self._pending_chunk
        else:
            drained = False

        if to_write:
            written = int(self._io_device.write(bytes(to_write)))
            if 0 <= written < len(to_write):
                self._pending_chunk = bytes(to_write[written:]) + self._pending_chunk
        elif drained and self._eof_drain_pts_ms is None:
            if int(self._sink.bytesFree()) >= int(self._sink.bufferSize()):
                processed_us = int(self._sink.processedUSecs())
                self._eof_drain_pts_ms = self._audio_origin_ms + (processed_us // 1000)
                self._eof_drain_wall = time.monotonic()

    def _enqueue_pcm(
        self,
        pcm: bytes,
        stop_event: threading.Event,
        first_chunk_event: threading.Event,
    ) -> bool:
        if not pcm:
            return True
        with self._cond:
            while (
                not stop_event.is_set()
                and self._queued_bytes >= self._max_queue_bytes
            ):
                self._cond.wait(timeout=0.05)
            if stop_event.is_set():
                return False
            self._queue.append(pcm)
            self._queued_bytes += len(pcm)
            first_chunk_event.set()
            return True

    def _decode_worker(
        self,
        container: av.container.InputContainer,
        audio_stream: av.audio.stream.AudioStream,
        start_raw_ms: int,
        av_fmt: str,
        av_layout: str,
        sample_rate: int,
        bytes_per_frame: int,
        stop_event: threading.Event,
        first_chunk_event: threading.Event,
    ) -> None:
        try:
            tb = (
                Fraction(audio_stream.time_base)
                if audio_stream.time_base is not None
                else Fraction(1, max(1, sample_rate))
            )
            seek_ms = max(0, start_raw_ms - _SEEK_BACKWARD_MARGIN_MS)
            if seek_ms > 0:
                seek_ticks = math.floor(Fraction(seek_ms, 1000) / tb)
                try:
                    container.seek(seek_ticks, backward=True, any_frame=False, stream=audio_stream)
                except av.error.FFmpegError:
                    try:
                        container.seek(0, backward=True, any_frame=False, stream=audio_stream)
                    except av.error.FFmpegError:
                        pass

            resampler = av.AudioResampler(
                format=av_fmt,
                layout=av_layout,
                rate=sample_rate,
            )
            have_origin = False
            cursor_ms: float = float(seek_ms)

            def _handle_resampled(
                resampled_frames: list[av.AudioFrame],
                frame_pts_ms: float,
            ) -> bool:
                nonlocal have_origin
                local_cursor = frame_pts_ms
                for rf in resampled_frames:
                    if stop_event.is_set():
                        return False
                    samples = int(rf.samples)
                    if samples <= 0:
                        continue
                    dur_ms = (samples * 1000.0) / float(sample_rate)
                    end_ms = local_cursor + dur_ms
                    if not have_origin and end_ms <= float(start_raw_ms):
                        local_cursor = end_ms
                        continue
                    pcm = rf.to_ndarray().tobytes()
                    if not have_origin:
                        if local_cursor < float(start_raw_ms):
                            skip_ms = float(start_raw_ms) - local_cursor
                            skip_samples = min(
                                samples,
                                max(0, int(round(skip_ms * sample_rate / 1000.0))),
                            )
                            pcm = pcm[skip_samples * bytes_per_frame :]
                            self._audio_origin_ms = start_raw_ms
                        else:
                            gap_ms = int(round(local_cursor - float(start_raw_ms)))
                            if 0 < gap_ms <= _MAX_PREPEND_SILENCE_MS:
                                pad_samples = int(round(gap_ms * sample_rate / 1000.0))
                                pcm = bytes(pad_samples * bytes_per_frame) + pcm
                                self._audio_origin_ms = start_raw_ms
                            else:
                                self._audio_origin_ms = int(round(local_cursor))
                        have_origin = True
                    local_cursor = end_ms
                    if pcm and not self._enqueue_pcm(pcm, stop_event, first_chunk_event):
                        return False
                return True

            for frame in container.decode(audio_stream):
                if stop_event.is_set():
                    break
                if frame.pts is not None:
                    frame_pts_ms = float(frame.pts * tb) * 1000.0
                else:
                    frame_pts_ms = cursor_ms
                in_rate = int(frame.sample_rate) if frame.sample_rate else sample_rate
                in_dur_ms = (int(frame.samples) * 1000.0) / float(max(1, in_rate))
                cursor_ms = frame_pts_ms + in_dur_ms

                resampled = resampler.resample(frame)
                if not _handle_resampled(resampled, frame_pts_ms):
                    break

            if not stop_event.is_set():
                flushed = resampler.resample(None)
                if flushed:
                    _handle_resampled(flushed, cursor_ms)
        except Exception:
            pass
        finally:
            try:
                container.close()
            except Exception:
                pass
            with self._cond:
                self._decoder_eof = True
                self._cond.notify_all()
            first_chunk_event.set()
