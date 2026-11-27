from __future__ import annotations

import math
from pathlib import Path

import av
import numpy as np
from PySide6.QtMultimedia import QAudioDevice, QAudioFormat

from muzhen.core.audio_player import AudioPreviewPlayer


def _make_av_mp4(path: Path, *, with_audio: bool = True, duration_sec: float = 1.2) -> Path:
    fps = 25
    sample_rate = 48000
    container = av.open(str(path), mode="w")
    try:
        vstream = container.add_stream("mpeg4", rate=fps)
        vstream.width = 160
        vstream.height = 90
        vstream.pix_fmt = "yuv420p"

        astream = None
        if with_audio:
            astream = container.add_stream("aac", rate=sample_rate)
            astream.layout = "stereo"

        total_frames = int(round(fps * duration_sec))
        for i in range(total_frames):
            img = np.zeros((90, 160, 3), dtype=np.uint8)
            img[:, :, 1] = (i * 7) % 255
            vf = av.VideoFrame.from_ndarray(img, format="bgr24")
            for pkt in vstream.encode(vf):
                container.mux(pkt)

        if astream is not None:
            chunk = 1024
            total_samples = int(round(sample_rate * duration_sec))
            t_idx = 0
            while t_idx < total_samples:
                n = min(chunk, total_samples - t_idx)
                t = (np.arange(t_idx, t_idx + n, dtype=np.float32)) / float(sample_rate)
                wave = (0.25 * np.sin(2.0 * math.pi * 440.0 * t)).astype(np.float32)
                stereo = np.stack([wave, wave], axis=0)
                af = av.AudioFrame.from_ndarray(stereo, format="fltp", layout="stereo")
                af.sample_rate = sample_rate
                af.pts = t_idx
                for pkt in astream.encode(af):
                    container.mux(pkt)
                t_idx += n

        for pkt in vstream.encode():
            container.mux(pkt)
        if astream is not None:
            for pkt in astream.encode():
                container.mux(pkt)
    finally:
        container.close()
    return path


class _FakeAudioDevice:
    def isNull(self) -> bool:
        return False

    def isFormatSupported(self, fmt: QAudioFormat) -> bool:
        return (
            fmt.sampleRate() == 48000
            and fmt.channelCount() == 2
            and fmt.sampleFormat() == QAudioFormat.SampleFormat.Int16
        )

    def preferredFormat(self) -> QAudioFormat:
        fmt = QAudioFormat()
        fmt.setSampleRate(48000)
        fmt.setChannelCount(2)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        return fmt


class _FakeIODevice:
    def __init__(self, sink: _FakeAudioSink) -> None:
        self._sink = sink

    def write(self, data: bytes) -> int:
        self._sink.written_chunks.append(bytes(data))
        self._sink.total_written += len(data)
        return len(data)


class _FakeAudioSink:
    def __init__(self, device, fmt: QAudioFormat, parent=None) -> None:
        self.device = device
        self.fmt = fmt
        self.parent = parent
        self._buffer_size = 48000
        self.volume = 1.0
        self.stopped = False
        self.deleted = False
        self.processed_us = 0
        self.total_written = 0
        self.written_chunks: list[bytes] = []
        self._io = _FakeIODevice(self)

    def setBufferSize(self, size: int) -> None:
        self._buffer_size = int(size)

    def bufferSize(self) -> int:
        return self._buffer_size

    def bytesFree(self) -> int:
        return self._buffer_size

    def setVolume(self, vol: float) -> None:
        self.volume = float(vol)

    def start(self) -> _FakeIODevice:
        return self._io

    def stop(self) -> None:
        self.stopped = True

    def deleteLater(self) -> None:
        self.deleted = True

    def processedUSecs(self) -> int:
        return self.processed_us


def test_audio_player_handles_missing_device_and_silent_video(qtbot, tmp_path: Path, monkeypatch):
    import muzhen.core.audio_player as ap_mod

    silent_mp4 = _make_av_mp4(tmp_path / "silent.mp4", with_audio=False, duration_sec=0.5)
    sound_mp4 = _make_av_mp4(tmp_path / "sound.mp4", with_audio=True, duration_sec=0.6)

    player = AudioPreviewPlayer()

    monkeypatch.setattr(ap_mod.QMediaDevices, "defaultAudioOutput", lambda: QAudioDevice())
    assert player.start(sound_mp4, 0) is False
    assert player.is_active is False
    assert player.playback_pts_ms() is None

    monkeypatch.setattr(ap_mod.QMediaDevices, "defaultAudioOutput", lambda: _FakeAudioDevice())
    assert player.start(silent_mp4, 0) is False
    assert player.is_active is False
    assert player.playback_pts_ms() is None
    player.stop()


def test_audio_player_streams_pcm_tracks_clock_mute_and_releases_file(
    qtbot, tmp_path: Path, monkeypatch
):
    import muzhen.core.audio_player as ap_mod

    sound_mp4 = _make_av_mp4(tmp_path / "episode_audio.mp4", with_audio=True, duration_sec=1.2)
    created_sinks: list[_FakeAudioSink] = []

    def _make_sink(device, fmt, parent=None):
        sink = _FakeAudioSink(device, fmt, parent)
        created_sinks.append(sink)
        return sink

    monkeypatch.setattr(ap_mod.QMediaDevices, "defaultAudioOutput", lambda: _FakeAudioDevice())
    monkeypatch.setattr(ap_mod, "QAudioSink", _make_sink)

    player = AudioPreviewPlayer()
    assert player.is_muted() is False
    assert player.start(sound_mp4, 400) is True
    assert player.is_active is True
    assert len(created_sinks) == 1
    sink = created_sinks[0]

    qtbot.waitUntil(lambda: sink.total_written > 0, timeout=3000)
    assert sink.volume == 1.0

    sink.processed_us = 250_000
    pts = player.playback_pts_ms()
    assert pts is not None
    assert 600 <= pts <= 700

    player.set_muted(True)
    assert player.is_muted() is True
    assert sink.volume == 0.0

    player.set_muted(False)
    assert player.is_muted() is False
    assert sink.volume == 1.0

    player.stop()
    assert player.is_active is False
    assert player.playback_pts_ms() is None
    assert sink.stopped is True
    assert sink.deleted is True

    sound_mp4.unlink()
    assert not sound_mp4.exists()
