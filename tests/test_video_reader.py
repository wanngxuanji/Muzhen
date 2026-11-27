from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import av
import numpy as np
import pytest

from transapp.core.models import ExtractResult, RoiRect
from transapp.core.video_reader import FrameSlice, VideoMetadata, VideoReader


def _make_synthetic_vfr_mp4(
    path: Path,
    *,
    width: int = 320,
    height: int = 180,
    pts_ms_list: tuple[int, ...] = (500, 540, 620, 700, 860, 910),
) -> None:
    """Encode a synthetic VFR MP4 with non-zero start PTS using PyAV."""
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=1000)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)

        for idx, pts_ms in enumerate(pts_ms_list):
            img = np.zeros((height, width, 3), dtype=np.uint8)
            img[:, :] = ((30 + idx * 35) % 256, (60 + idx * 25) % 256, (220 - idx * 30) % 256)
            y1 = int(height * 0.8)
            y2 = int(height * 0.95)
            img[y1:y2, :] = ((240 - idx * 20) % 256, (240 - idx * 15) % 256, (240 - idx * 10) % 256)

            frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            frame.pts = pts_ms
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)


def test_video_reader_metadata_and_vfr_roi_streaming_unicode_path(tmp_path: Path):
    video_path = tmp_path / "测试_—_片段 01.mp4"
    expected_pts = (500, 540, 620, 700, 860, 910)
    _make_synthetic_vfr_mp4(video_path, width=320, height=180, pts_ms_list=expected_pts)

    roi = RoiRect(0.1, 0.8, 0.8, 0.15)
    x1, y1, x2, y2 = roi.to_pixels(320, 180)
    expected_roi_shape = (y2 - y1, x2 - x1, 3)

    with VideoReader(video_path) as reader:
        meta = reader.metadata
        assert isinstance(meta, VideoMetadata)
        assert meta.width == 320
        assert meta.height == 180
        assert meta.duration_ms > 0
        assert meta.fps > 0.0
        assert meta.estimated_frames >= 1

        slices = list(reader.iter_roi_frames(roi))

    assert len(slices) == 6
    for idx, sl in enumerate(slices):
        assert isinstance(sl, FrameSlice)
        assert sl.frame_index == idx
        assert isinstance(sl.pts_ms, int)
        assert sl.bgr_roi.dtype == np.uint8
        assert sl.bgr_roi.shape == expected_roi_shape

    pts_values = [sl.pts_ms for sl in slices]
    assert pts_values == list(expected_pts)
    for a, b in pairwise(pts_values):
        assert b > a


def test_video_reader_read_frame_at_ms_and_clamping(tmp_path: Path):
    video_path = tmp_path / "测试_—_片段 01.mp4"
    expected_pts = (500, 540, 620, 700, 860, 910)
    _make_synthetic_vfr_mp4(video_path, width=320, height=180, pts_ms_list=expected_pts)

    with VideoReader(str(video_path)) as reader:
        idx_neg, pts_neg, frame_neg = reader.read_frame_at_ms(-500)
        assert idx_neg == 0
        assert pts_neg >= 500
        assert frame_neg.shape == (180, 320, 3)
        assert frame_neg.dtype == np.uint8

        idx_mid, pts_mid, frame_mid = reader.read_frame_at_ms(700)
        assert 0 <= idx_mid < 6
        assert abs(pts_mid - 700) <= 50
        assert frame_mid.shape == (180, 320, 3)

        idx_end, pts_end, frame_end = reader.read_frame_at_ms(999_999)
        assert idx_end == 5
        assert pts_end >= 900
        assert frame_end.shape == (180, 320, 3)

        slices = list(reader.iter_roi_frames(RoiRect(0.1, 0.8, 0.8, 0.15)))
        assert len(slices) == 6
        assert [s.frame_index for s in slices] == [0, 1, 2, 3, 4, 5]


def test_video_reader_error_handling_and_close(tmp_path: Path):
    missing = tmp_path / "does_not_exist.mp4"
    with pytest.raises(FileNotFoundError):
        VideoReader(missing)

    corrupt = tmp_path / "corrupt_测试.mp4"
    corrupt.write_bytes(b"not a valid mp4 container payload")
    with pytest.raises(ValueError):
        VideoReader(corrupt)

    audio_only = tmp_path / "audio_only.mp4"
    with av.open(str(audio_only), mode="w") as container:
        astream = container.add_stream("aac", rate=44100)
        aframe = av.AudioFrame(format="fltp", layout="stereo", samples=1024)
        aframe.sample_rate = 44100
        for plane in aframe.planes:
            plane.update(np.zeros(plane.buffer_size, dtype=np.uint8))
        for pkt in astream.encode(aframe):
            container.mux(pkt)
        for pkt in astream.encode():
            container.mux(pkt)

    with pytest.raises(ValueError):
        VideoReader(audio_only)

    valid = tmp_path / "valid.mp4"
    _make_synthetic_vfr_mp4(valid, pts_ms_list=(100, 200))
    reader = VideoReader(valid)
    reader.close()
    reader.close()
    with pytest.raises(ValueError):
        reader.read_frame_at_ms(0)


def test_video_reader_monotonic_pts_fallback_on_missing_or_duplicate_ticks(tmp_path: Path):
    class _FakeFrame:
        def __init__(self, pts, dts):
            self.pts = pts
            self.dts = dts
            self.time_base = Fraction(1, 1000)

    class _FakeStream:
        time_base = Fraction(1, 1000)

    valid = tmp_path / "fallback.mp4"
    _make_synthetic_vfr_mp4(valid, pts_ms_list=(100, 140))

    with VideoReader(valid) as reader:
        stream = _FakeStream()
        p0 = reader._frame_pts_ms(_FakeFrame(pts=None, dts=200), stream, 0, None)
        p1 = reader._frame_pts_ms(_FakeFrame(pts=p0, dts=200), stream, 1, p0)
        p2 = reader._frame_pts_ms(_FakeFrame(pts=None, dts=None), stream, 2, p1)
        assert p0 == reader.metadata.start_pts_ms
        assert p1 == p0 + 1
        assert p2 > p1


def test_video_reader_duration_fallback_and_probe_stops_after_first_frame(tmp_path: Path, monkeypatch):
    video_path = tmp_path / "probe_check.mp4"
    _make_synthetic_vfr_mp4(video_path, width=320, height=180, pts_ms_list=(100, 140, 180, 220, 260, 300))

    real_open = av.open

    class _StreamProxy:
        def __init__(self, real_stream, *, zero_dims: bool):
            self._real = real_stream
            self._zero_dims = zero_dims
            self.duration = 0
            self.time_base = real_stream.time_base
            self.start_time = real_stream.start_time
            self.average_rate = real_stream.average_rate
            self.guessed_rate = real_stream.guessed_rate
            self.base_rate = real_stream.base_rate
            self.frames = real_stream.frames
            self.thread_type = "AUTO"
            self.width = 0 if zero_dims else real_stream.width
            self.height = 0 if zero_dims else real_stream.height

            class _CodecProxy:
                width = 0 if zero_dims else real_stream.codec_context.width
                height = 0 if zero_dims else real_stream.codec_context.height

            self.codec_context = _CodecProxy()

    class _ContainerProxy:
        def __init__(self, real_container, *, zero_dims: bool):
            self._real = real_container
            self._stream_proxy = _StreamProxy(real_container.streams.video[0], zero_dims=zero_dims)
            self.duration = 2_500_000
            self.decoded_count = 0

            class _Streams:
                def __init__(self, video_stream):
                    self.video = [video_stream]

            self.streams = _Streams(self._stream_proxy)

        def seek(self, *args, **kwargs):
            return self._real.seek(*args, **kwargs)

        def decode(self, stream):
            target = stream._real if isinstance(stream, _StreamProxy) else stream
            for frame in self._real.decode(target):
                self.decoded_count += 1
                yield frame

        def close(self):
            self._real.close()

    proxies: list[_ContainerProxy] = []

    def _open_case1(path, *args, **kwargs):
        proxy = _ContainerProxy(real_open(path, *args, **kwargs), zero_dims=False)
        proxies.append(proxy)
        return proxy

    monkeypatch.setattr(av, "open", _open_case1)
    with VideoReader(video_path) as reader:
        assert reader.metadata.duration_ms == 2500
        assert proxies[0].decoded_count == 1

    proxies.clear()

    def _open_case2(path, *args, **kwargs):
        proxy = _ContainerProxy(real_open(path, *args, **kwargs), zero_dims=True)
        proxies.append(proxy)
        return proxy

    monkeypatch.setattr(av, "open", _open_case2)
    with VideoReader(video_path) as reader:
        assert reader.metadata.width == 320
        assert reader.metadata.height == 180
        assert reader.metadata.duration_ms == 2500
        assert proxies[0].decoded_count == 1


def test_video_reader_start_pts_ms_falsy_timebase_and_init_error_closes_container(
    tmp_path: Path, monkeypatch
):
    legacy_meta = VideoMetadata(width=320, height=180, duration_ms=100, fps=25.0, estimated_frames=3)
    assert legacy_meta.start_pts_ms == 0

    video_path = tmp_path / "offset_pts.mp4"
    _make_synthetic_vfr_mp4(
        video_path,
        width=320,
        height=180,
        pts_ms_list=(5000, 5040, 5080, 5120),
    )
    with VideoReader(video_path) as reader:
        meta = reader.metadata
        assert meta.start_pts_ms == 5000
        assert meta.start_pts_ms >= meta.duration_ms

        class _FakeFrame:
            pts = 1234
            dts = None
            time_base = None

        class _FakeStream:
            time_base = None

        assert reader._frame_pts_ms(_FakeFrame(), _FakeStream(), 0, None) == 1234
        assert not hasattr(reader, "_raw_pts_ms")

    real_open = av.open
    close_calls = {"count": 0}

    class _TrackingContainer:
        def __init__(self, inner):
            self._inner = inner
            self.streams = inner.streams
            self.duration = inner.duration

        def close(self):
            close_calls["count"] += 1
            self._inner.close()

    monkeypatch.setattr(av, "open", lambda path, *a, **kw: _TrackingContainer(real_open(path, *a, **kw)))
    monkeypatch.setattr(
        VideoReader,
        "_compute_start_pts_ms",
        staticmethod(lambda _stream: (_ for _ in ()).throw(RuntimeError("boom"))),
    )
    with pytest.raises(RuntimeError, match="boom"):
        VideoReader(video_path)
    assert close_calls["count"] == 1


def test_video_reader_signed_origin_pts_negative_edit_list_and_missing_stream_start_time(
    tmp_path: Path, monkeypatch
):
    """Negative edit-list PTS[0] and stream.start_time=None must both anchor rel_ms=0 with 0ms drift."""
    res = ExtractResult(
        video_path=tmp_path / "dummy.mp4",
        width=320,
        height=180,
        duration_ms=1000,
        events=[],
        elapsed_sec=0.1,
    )
    assert res.frame_pts_ms == []

    legacy_slice = FrameSlice(1, 40, np.zeros((4, 4, 3), dtype=np.uint8))
    assert legacy_slice.rel_ms == 40

    dummy_file = tmp_path / "synthetic_origin.mp4"
    _make_synthetic_vfr_mp4(dummy_file, width=320, height=180, pts_ms_list=(0, 40))

    class _FakeFrame:
        def __init__(self, pts, dts, tb, width=320, height=180):
            self.pts = pts
            self.dts = dts
            self.time_base = tb
            self.width = width
            self.height = height

        def to_ndarray(self, format="bgr24"):
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    class _FakeStream:
        def __init__(self, start_time, time_base, rate):
            self.start_time = start_time
            self.time_base = time_base
            self.average_rate = rate
            self.guessed_rate = rate
            self.base_rate = rate
            self.duration = 4040
            self.frames = 101
            self.width = 320
            self.height = 180
            self.thread_type = "AUTO"

            class _Codec:
                width = 320
                height = 180

            self.codec_context = _Codec()

    class _FakeContainer:
        def __init__(self, stream, frames):
            self._frames = frames
            self.duration = 4_040_000
            self.seek_calls: list[tuple[tuple, dict]] = []

            class _Streams:
                def __init__(self, vs):
                    self.video = [vs]

            self.streams = _Streams(stream)

        def seek(self, *args, **kwargs):
            self.seek_calls.append((args, kwargs))

        def decode(self, _stream):
            yield from self._frames

        def close(self):
            pass

    roi = RoiRect(0.1, 0.8, 0.8, 0.15)

    tb_ms = Fraction(1, 1000)
    neg_frames = [_FakeFrame(pts=-83 + i * 40, dts=-123 + i * 40, tb=tb_ms) for i in range(101)]
    neg_stream = _FakeStream(start_time=0, time_base=tb_ms, rate=Fraction(25, 1))
    neg_container = _FakeContainer(neg_stream, neg_frames)

    monkeypatch.setattr(av, "open", lambda *_a, **_kw: neg_container)
    with VideoReader(dummy_file) as reader:
        assert reader.metadata.start_pts_ms == -83
        slices = list(reader.iter_roi_frames(roi))
        assert len(slices) == 101
        assert slices[0].pts_ms == -83
        assert slices[0].rel_ms == 0
        assert slices[100].pts_ms == -83 + 4000
        assert slices[100].rel_ms == 4000
        assert any(call_args and call_args[0] < 0 for call_args, _ in neg_container.seek_calls)

        p0 = reader._frame_pts_ms(_FakeFrame(pts=None, dts=9999, tb=tb_ms), neg_stream, 0, None)
        p1 = reader._frame_pts_ms(_FakeFrame(pts=None, dts=5000, tb=tb_ms), neg_stream, 1, p0)
        assert p0 == -83
        assert p1 == -83 + 40

        idx_neg0, pts_neg0, _ = reader.read_frame_at_ms(-83)
        idx_neg1, pts_neg1, _ = reader.read_frame_at_ms(-43)
        assert idx_neg0 == 0
        assert pts_neg0 == -83
        assert idx_neg1 == 1
        assert pts_neg1 == -43

    shifted_frames = [_FakeFrame(pts=12000 + i * 40, dts=12000 + i * 40, tb=tb_ms) for i in range(101)]
    shifted_stream = _FakeStream(start_time=None, time_base=tb_ms, rate=Fraction(25, 1))
    shifted_container = _FakeContainer(shifted_stream, shifted_frames)

    monkeypatch.setattr(av, "open", lambda *_a, **_kw: shifted_container)
    with VideoReader(dummy_file) as reader:
        assert reader.metadata.start_pts_ms == 12000
        slices = list(reader.iter_roi_frames(roi))
        assert len(slices) == 101
        assert slices[0].pts_ms == 12000
        assert slices[0].rel_ms == 0
        assert slices[100].pts_ms == 16000
        assert slices[100].rel_ms == 4000


def test_video_reader_ntsc_23976_rational_pts_zero_drift(tmp_path: Path):
    """Exact rational Fraction conversion on 23.976fps (24000/1001) must have 0ms drift over 24+ min."""
    dummy_file = tmp_path / "ntsc_check.mp4"
    _make_synthetic_vfr_mp4(dummy_file, pts_ms_list=(0, 42))

    class _FakeFrame:
        def __init__(self, pts, tb):
            self.pts = pts
            self.dts = None
            self.time_base = tb

    class _FakeStream:
        def __init__(self, tb):
            self.time_base = tb

    tb_frame = Fraction(1001, 24000)
    tb_fine = Fraction(1, 24000)
    stream_frame = _FakeStream(tb_frame)
    stream_fine = _FakeStream(tb_fine)

    with VideoReader(dummy_file) as reader:
        prev_pts: int | None = None
        for k in range(1000):
            expected_ms = int(round(Fraction(k * 1001 * 1000, 24000)))
            got_frame = reader._frame_pts_ms(_FakeFrame(k, tb_frame), stream_frame, k, prev_pts)
            got_fine = reader._frame_pts_ms(_FakeFrame(k * 1001, tb_fine), stream_fine, k, None)
            assert got_frame == expected_ms
            assert got_fine == expected_ms
            prev_pts = got_frame

        k_24m = 34_560
        expected_24m = int(round(Fraction(k_24m * 1001 * 1000, 24000)))
        assert expected_24m == 1_441_440
        assert reader._frame_pts_ms(_FakeFrame(k_24m, tb_frame), stream_frame, k_24m, None) == expected_24m
        assert reader._frame_pts_ms(_FakeFrame(k_24m * 1001, tb_fine), stream_fine, k_24m, None) == expected_24m


def test_video_reader_packet_demux_duration_probe_without_full_decode(tmp_path: Path, monkeypatch):
    """Missing container/stream duration must estimate duration via packet demux without full decode."""
    video_path = tmp_path / "no_duration.mp4"
    pts_list = tuple(200 + i * 40 for i in range(20))
    _make_synthetic_vfr_mp4(video_path, width=320, height=180, pts_ms_list=pts_list)

    real_open = av.open

    class _StreamProxy:
        def __init__(self, real_stream):
            self._real = real_stream
            self.duration = None
            self.time_base = real_stream.time_base
            self.start_time = real_stream.start_time
            self.average_rate = real_stream.average_rate
            self.guessed_rate = real_stream.guessed_rate
            self.base_rate = real_stream.base_rate
            self.frames = 0
            self.thread_type = "AUTO"
            self.width = real_stream.width
            self.height = real_stream.height
            self.codec_context = real_stream.codec_context

    class _ContainerProxy:
        def __init__(self, real_container):
            self._real = real_container
            self._stream_proxy = _StreamProxy(real_container.streams.video[0])
            self.duration = None
            self.decoded_count = 0
            self.demuxed_packets = 0

            class _Streams:
                def __init__(self, vs):
                    self.video = [vs]

            self.streams = _Streams(self._stream_proxy)

        def seek(self, *args, **kwargs):
            if "stream" in kwargs and isinstance(kwargs["stream"], _StreamProxy):
                kwargs = {**kwargs, "stream": kwargs["stream"]._real}
            return self._real.seek(*args, **kwargs)

        def demux(self, stream):
            target = stream._real if isinstance(stream, _StreamProxy) else stream
            for pkt in self._real.demux(target):
                if pkt.pts is not None or pkt.dts is not None:
                    self.demuxed_packets += 1
                yield pkt

        def decode(self, stream):
            target = stream._real if isinstance(stream, _StreamProxy) else stream
            for frame in self._real.decode(target):
                self.decoded_count += 1
                yield frame

        def close(self):
            self._real.close()

    proxies: list[_ContainerProxy] = []

    def _open_proxy(path, *args, **kwargs):
        proxy = _ContainerProxy(real_open(path, *args, **kwargs))
        proxies.append(proxy)
        return proxy

    monkeypatch.setattr(av, "open", _open_proxy)
    with VideoReader(video_path) as reader:
        meta = reader.metadata
        assert proxies[0].decoded_count == 1
        assert proxies[0].demuxed_packets >= 20
        assert 760 <= meta.duration_ms <= 840
        assert meta.start_pts_ms == 200


def test_video_reader_iter_roi_frames_trim_window_preserves_absolute_rel_ms(tmp_path: Path):
    """Trimmed iter_roi_frames(start_ms, end_ms) seeks cleanly and preserves absolute pts_ms & rel_ms."""
    video_path = tmp_path / "trim_window.mp4"
    width, height = 320, 180
    total_frames = 40
    origin_ms = 480
    pts_list = [origin_ms + i * 40 for i in range(total_frames)]

    with av.open(str(video_path), mode="w") as container:
        stream = container.add_stream("libx264", rate=25)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.options = {"g": "5", "keyint_min": "5"}

        for idx, pts_ms in enumerate(pts_list):
            img = np.zeros((height, width, 3), dtype=np.uint8)
            val = min(250, 15 + idx * 5)
            img[:, :] = (40, val, 60)
            frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            frame.pts = pts_ms
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    roi = RoiRect(0.1, 0.8, 0.8, 0.15)
    with VideoReader(video_path) as reader:
        assert reader.metadata.start_pts_ms == origin_ms

        window = list(reader.iter_roi_frames(roi, start_ms=400, end_ms=1000))
        assert len(window) == 16
        assert [sl.frame_index for sl in window] == list(range(16))
        assert [sl.rel_ms for sl in window] == [400 + i * 40 for i in range(16)]
        assert [sl.pts_ms for sl in window] == [origin_ms + 400 + i * 40 for i in range(16)]
        first_green = float(window[0].bgr_roi[:, :, 1].mean())
        assert abs(first_green - 65.0) < 10.0

        head = list(reader.iter_roi_frames(roi, end_ms=200))
        assert [sl.rel_ms for sl in head] == [0, 40, 80, 120, 160, 200]

        tail = list(reader.iter_roi_frames(roi, start_ms=1400))
        assert [sl.rel_ms for sl in tail] == [1400, 1440, 1480, 1520, 1560]

        assert list(reader.iter_roi_frames(roi, start_ms=900, end_ms=400)) == []


def test_video_reader_missing_pts_after_seek_ntsc_zero_drift_and_demux_origin(
    tmp_path: Path, monkeypatch
):
    """Regression: post-seek missing PTS uses rational anchor (0ms NTSC drift), pre-roll real PTS
    does not clamp against synthetic guess, and demux duration does not inflate when origin is unknown."""
    dummy_file = tmp_path / "synthetic_seek_pts.mp4"
    _make_synthetic_vfr_mp4(dummy_file, width=320, height=180, pts_ms_list=(0, 40))

    class _FakeFrame:
        def __init__(self, pts, dts, tb, width=320, height=180):
            self.pts = pts
            self.dts = dts
            self.time_base = tb
            self.width = width
            self.height = height

        def to_ndarray(self, format="bgr24"):
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    class _FakePacket:
        def __init__(self, pts, duration, tb):
            self.pts = pts
            self.dts = pts
            self.duration = duration
            self.time_base = tb

    class _FakeStream:
        def __init__(self, *, start_time, time_base, rate, duration=10_000):
            self.start_time = start_time
            self.time_base = time_base
            self.average_rate = rate
            self.guessed_rate = rate
            self.base_rate = rate
            self.duration = duration
            self.frames = 0
            self.width = 320
            self.height = 180
            self.thread_type = "AUTO"

            class _Codec:
                width = 320
                height = 180

            self.codec_context = _Codec()

    class _FakeContainer:
        def __init__(self, stream, init_frames, seek_frames=None, packets=None, duration=10_000_000):
            self._init_frames = init_frames
            self._seek_frames = seek_frames if seek_frames is not None else init_frames
            self._packets = packets or []
            self.duration = duration
            self._after_seek = False

            class _Streams:
                def __init__(self, vs):
                    self.video = [vs]

            self.streams = _Streams(stream)

        def seek(self, offset, *args, **kwargs):
            self._after_seek = offset > 0

        def decode(self, _stream):
            yield from (self._seek_frames if self._after_seek else self._init_frames)

        def demux(self, _stream):
            yield from self._packets

        def close(self):
            pass

    roi = RoiRect(0.1, 0.8, 0.8, 0.15)
    ntsc_rate = Fraction(24000, 1001)
    tb_ms = Fraction(1, 1000)

    init_ntsc = [_FakeFrame(pts=0, dts=0, tb=tb_ms)]
    missing_ntsc = [_FakeFrame(pts=None, dts=9999 - i, tb=tb_ms) for i in range(101)]
    stream_ntsc = _FakeStream(start_time=0, time_base=tb_ms, rate=ntsc_rate)
    cont_ntsc = _FakeContainer(stream_ntsc, init_ntsc, seek_frames=missing_ntsc)

    monkeypatch.setattr(av, "open", lambda *_a, **_kw: cont_ntsc)
    with VideoReader(dummy_file) as reader:
        slices = list(reader.iter_roi_frames(roi, start_ms=1000, end_ms=6000))
        assert len(slices) == 89
        expected_first = 500 + int(round(Fraction(12 * 1000) / ntsc_rate))
        expected_last = 500 + int(round(Fraction(100 * 1000) / ntsc_rate))
        assert expected_first == 1000
        assert expected_last == 4671
        assert slices[0].pts_ms == expected_first
        assert slices[0].rel_ms == expected_first
        assert slices[-1].pts_ms == expected_last
        assert slices[-1].rel_ms == expected_last

        _idx, got_pts, _bgr = reader.read_frame_at_ms(1200)
        assert got_pts == 700 + int(round(Fraction(12 * 1000) / ntsc_rate))

    init_preroll = [_FakeFrame(pts=500, dts=500, tb=tb_ms)]
    seek_preroll = [
        _FakeFrame(pts=None, dts=1380, tb=tb_ms),
        _FakeFrame(pts=1420, dts=1420, tb=tb_ms),
        _FakeFrame(pts=None, dts=1460, tb=tb_ms),
        _FakeFrame(pts=1500, dts=1500, tb=tb_ms),
        _FakeFrame(pts=1540, dts=1540, tb=tb_ms),
    ]
    stream_preroll = _FakeStream(start_time=500, time_base=tb_ms, rate=Fraction(25, 1))
    cont_preroll = _FakeContainer(stream_preroll, init_preroll, seek_frames=seek_preroll)

    monkeypatch.setattr(av, "open", lambda *_a, **_kw: cont_preroll)
    with VideoReader(dummy_file) as reader:
        slices = list(reader.iter_roi_frames(roi, start_ms=1000, end_ms=1200))
        assert [sl.rel_ms for sl in slices] == [1000, 1040]
        assert [sl.pts_ms for sl in slices] == [1500, 1540]

    init_unknown = [_FakeFrame(pts=None, dts=None, tb=tb_ms)]
    packets_shifted = [_FakePacket(pts=12000 + i * 40, duration=40, tb=tb_ms) for i in range(25)]
    stream_unknown = _FakeStream(start_time=None, time_base=tb_ms, rate=Fraction(25, 1), duration=None)
    cont_unknown = _FakeContainer(
        stream_unknown, init_unknown, packets=packets_shifted, duration=None
    )

    monkeypatch.setattr(av, "open", lambda *_a, **_kw: cont_unknown)
    with VideoReader(dummy_file) as reader:
        assert reader.metadata.start_pts_ms == 0
        assert reader.metadata.duration_ms == 1000


def test_read_prev_frame_before_ms_does_not_deadlock_on_vfr_gap(tmp_path: Path, monkeypatch):
    """read_prev_frame_before_ms returns latest frame strictly < current_pts_ms across VFR gaps."""
    video_path = tmp_path / "vfr_gap.mp4"
    pts_list = (0, 1000, 1084, 1126)
    _make_synthetic_vfr_mp4(video_path, width=320, height=180, pts_ms_list=pts_list)

    with VideoReader(video_path) as reader:
        bgr_1000, pts_1000 = reader.read_prev_frame_before_ms(1084)
        assert pts_1000 == 1000
        assert bgr_1000.shape == (180, 320, 3)
        assert bgr_1000.dtype == np.uint8

        bgr_0, pts_0 = reader.read_prev_frame_before_ms(1000)
        assert pts_0 == 0
        assert bgr_0.shape == (180, 320, 3)

        assert abs(float(bgr_1000[:40, :40, 0].mean()) - float(bgr_0[:40, :40, 0].mean())) > 15.0

        bgr_start, pts_start = reader.read_prev_frame_before_ms(0)
        assert pts_start == 0
        assert bgr_start.shape == (180, 320, 3)

    tb_ms = Fraction(1, 1000)

    class _FakeFrame:
        def __init__(self, pts: int):
            self.pts = pts
            self.dts = pts
            self.time_base = tb_ms
            self.width = 64
            self.height = 64
            self.convert_calls = 0

        def to_ndarray(self, format: str = "bgr24") -> np.ndarray:
            self.convert_calls += 1
            return np.full((64, 64, 3), self.pts % 250, dtype=np.uint8)

    class _FakeStream:
        start_time = 0
        time_base = tb_ms
        average_rate = Fraction(24, 1)
        guessed_rate = Fraction(24, 1)
        base_rate = Fraction(24, 1)
        duration = 10_000
        frames = 4
        width = 64
        height = 64
        thread_type = "AUTO"

        class _Codec:
            width = 64
            height = 64

        codec_context = _Codec()

    frames = [_FakeFrame(0), _FakeFrame(4000), _FakeFrame(7000), _FakeFrame(7042)]

    class _BackoffContainer:
        def __init__(self) -> None:
            self.duration = 10_000_000
            self._seek_ms = 0
            self.seek_targets: list[int] = []

            class _Streams:
                video = [_FakeStream()]

            self.streams = _Streams()

        def seek(self, offset, *args, **kwargs):
            pos_ms = int(offset // 1000) if "stream" not in kwargs else int(offset)
            self._seek_ms = pos_ms
            self.seek_targets.append(pos_ms)

        def decode(self, _stream):
            if self._seek_ms <= 0:
                yield from frames
            elif self._seek_ms <= 4000:
                yield from frames[1:]
            else:
                yield from frames[2:]

        def close(self) -> None:
            pass

    cont = _BackoffContainer()
    monkeypatch.setattr(av, "open", lambda *_a, **_kw: cont)
    with VideoReader(video_path) as reader:
        cont.seek_targets.clear()
        bgr, pts = reader.read_prev_frame_before_ms(7000)
        assert pts == 4000
        assert int(bgr[0, 0, 0]) == 4000 % 250
        assert cont.seek_targets == [6500, 5000, 2000]
        assert sum(f.convert_calls for f in frames) == 1


def test_iter_full_frames_from_ms_handles_negative_start_pts_vfr_and_past_eof(
    tmp_path: Path, monkeypatch
):
    """iter_full_frames_from_ms streams mid-clip VFR, negative start_pts_ms, and yields empty past EOF."""
    video_path = tmp_path / "vfr_full_stream.mp4"
    pts_list = (500, 540, 620, 700, 860, 910)
    _make_synthetic_vfr_mp4(video_path, width=320, height=180, pts_ms_list=pts_list)

    with VideoReader(video_path) as reader:
        mid = list(reader.iter_full_frames_from_ms(650))
        assert [pts for _, pts, _ in mid] == [700, 860, 910]
        assert all(bgr.shape == (180, 320, 3) and bgr.dtype == np.uint8 for _, _, bgr in mid)

        past_eof = list(reader.iter_full_frames_from_ms(5000))
        assert past_eof == []

    tb_ms = Fraction(1, 1000)

    class _FakeFrame:
        def __init__(self, pts: int):
            self.pts = pts
            self.dts = pts
            self.time_base = tb_ms
            self.width = 64
            self.height = 64

        def to_ndarray(self, format: str = "bgr24") -> np.ndarray:
            return np.zeros((64, 64, 3), dtype=np.uint8)

    class _FakeStream:
        start_time = 0
        time_base = tb_ms
        average_rate = Fraction(25, 1)
        guessed_rate = Fraction(25, 1)
        base_rate = Fraction(25, 1)
        duration = 2000
        frames = 40
        width = 64
        height = 64
        thread_type = "AUTO"

        class _Codec:
            width = 64
            height = 64

        codec_context = _Codec()

    neg_frames = [_FakeFrame(-83 + i * 40) for i in range(40)]

    class _NegContainer:
        def __init__(self) -> None:
            self.duration = 2_000_000
            self.seek_calls: list[tuple[int, bool]] = []

            class _Streams:
                video = [_FakeStream()]

            self.streams = _Streams()

        def seek(self, offset, *args, **kwargs):
            self.seek_calls.append((int(offset), "stream" in kwargs))

        def decode(self, _stream):
            yield from neg_frames

        def close(self) -> None:
            pass

    cont = _NegContainer()
    monkeypatch.setattr(av, "open", lambda *_a, **_kw: cont)
    with VideoReader(video_path) as reader:
        assert reader.metadata.start_pts_ms == -83

        from_neg = list(reader.iter_full_frames_from_ms(-43))
        assert from_neg[0][1] == -43
        assert from_neg[-1][1] == neg_frames[-1].pts

        cont.seek_calls.clear()
        from_mid = list(reader.iter_full_frames_from_ms(1000))
        assert from_mid[0][1] == 1037
        stream_seeks = [off for off, has_stream in cont.seek_calls if has_stream]
        assert stream_seeks and stream_seeks[0] <= 500

        empty = list(reader.iter_full_frames_from_ms(9999))
        assert empty == []


def test_seek_applies_backward_safety_margin_and_defers_bgr_conversion(
    tmp_path: Path, monkeypatch
):
    """_decode_near_ms & iter_roi_frames seek with >=500ms backward margin and convert only chosen frame."""
    dummy_path = tmp_path / "lazy_bgr.mp4"
    real_pts = tuple(i * 100 for i in range(15))
    _make_synthetic_vfr_mp4(dummy_path, width=160, height=90, pts_ms_list=real_pts)
    with VideoReader(dummy_path) as reader:
        _, pts_500, bgr_500 = reader.read_frame_at_ms(520)
        _, pts_600, bgr_600 = reader.read_frame_at_ms(600)
        assert pts_500 == 500
        assert pts_600 == 600
        assert abs(float(bgr_500[:20, :20, 0].mean()) - float(bgr_600[:20, :20, 0].mean())) > 15.0

    tb_ms = Fraction(1, 1000)

    class _CountingFrame:
        def __init__(self, pts: int, counter: list[int]):
            self.pts = pts
            self.dts = pts
            self.time_base = tb_ms
            self.width = 64
            self.height = 64
            self._counter = counter

        def to_ndarray(self, format: str = "bgr24") -> np.ndarray:
            self._counter.append(self.pts)
            return np.full((64, 64, 3), self.pts % 250, dtype=np.uint8)

    class _FakeStream:
        start_time = 200
        time_base = tb_ms
        average_rate = Fraction(10, 1)
        guessed_rate = Fraction(10, 1)
        base_rate = Fraction(10, 1)
        duration = 2400
        frames = 24
        width = 64
        height = 64
        thread_type = "AUTO"

        class _Codec:
            width = 64
            height = 64

        codec_context = _Codec()

    converted_pts: list[int] = []
    fake_frames = [_CountingFrame(200 + i * 100, converted_pts) for i in range(24)]

    class _SpyContainer:
        def __init__(self) -> None:
            self.duration = 2_400_000
            self.stream_seeks: list[int] = []
            self.start_seeks: list[int] = []

            class _Streams:
                video = [_FakeStream()]

            self.streams = _Streams()

        def seek(self, offset, *args, **kwargs):
            if "stream" in kwargs:
                self.stream_seeks.append(int(offset))
            else:
                self.start_seeks.append(int(offset))

        def decode(self, _stream):
            yield from fake_frames

        def close(self) -> None:
            pass

    cont = _SpyContainer()
    monkeypatch.setattr(av, "open", lambda *_a, **_kw: cont)
    with VideoReader(dummy_path) as reader:
        assert reader.metadata.start_pts_ms == 200

        cont.stream_seeks.clear()
        converted_pts.clear()
        _, pts, bgr = reader.read_frame_at_ms(1500)
        assert pts == 1500
        assert int(bgr[0, 0, 0]) == 1500 % 250
        assert cont.stream_seeks == [1000]
        assert converted_pts == [1500]

        cont.stream_seeks.clear()
        converted_pts.clear()
        _, pts, bgr = reader.read_frame_at_ms(1430)
        assert pts == 1400
        assert int(bgr[0, 0, 0]) == 1400 % 250
        assert cont.stream_seeks == [930]
        assert converted_pts == [1400]

        cont.stream_seeks.clear()
        cont.start_seeks.clear()
        converted_pts.clear()
        _, pts, _ = reader.read_frame_at_ms(400)
        assert pts == 400
        assert cont.stream_seeks == []
        assert len(cont.start_seeks) >= 1
        assert converted_pts == [400]

        cont.stream_seeks.clear()
        slices = list(reader.iter_roi_frames(RoiRect(0.0, 0.0, 1.0, 1.0), start_ms=1300, end_ms=1400))
        assert [sl.pts_ms for sl in slices] == [1500, 1600]
        assert cont.stream_seeks == [1000]


def test_pts_less_frames_before_trim_start_are_not_anchored_prematurely(
    tmp_path: Path, monkeypatch
):
    """Post-seek frames with pts=None before target_raw_ms anchor at seek_raw_ms and are skipped."""
    dummy_path = tmp_path / "pts_less_preroll.mp4"
    _make_synthetic_vfr_mp4(dummy_path, width=64, height=64, pts_ms_list=(0, 40))

    tb_ms = Fraction(1, 1000)

    class _FakeFrame:
        def __init__(self, pts: int | None, tag: int):
            self.pts = pts
            self.dts = None
            self.time_base = tb_ms
            self.width = 64
            self.height = 64
            self.tag = tag

        def to_ndarray(self, format: str = "bgr24") -> np.ndarray:
            return np.full((64, 64, 3), self.tag, dtype=np.uint8)

    class _FakeStream:
        start_time = 500
        time_base = tb_ms
        average_rate = Fraction(25, 1)
        guessed_rate = Fraction(25, 1)
        base_rate = Fraction(25, 1)
        duration = 3000
        frames = 75
        width = 64
        height = 64
        thread_type = "AUTO"

        class _Codec:
            width = 64
            height = 64

        codec_context = _Codec()

    init_frames = [_FakeFrame(pts=500, tag=10)]
    post_seek_frames = [
        _FakeFrame(pts=None, tag=20),
        _FakeFrame(pts=None, tag=30),
        _FakeFrame(pts=1420, tag=40),
        _FakeFrame(pts=None, tag=50),
        _FakeFrame(pts=1500, tag=60),
        _FakeFrame(pts=1540, tag=70),
    ]

    class _FakeContainer:
        def __init__(self) -> None:
            self.duration = 3_000_000
            self._after_seek = False

            class _Streams:
                video = [_FakeStream()]

            self.streams = _Streams()

        def seek(self, offset, *args, **kwargs):
            self._after_seek = "stream" in kwargs and int(offset) > 500

        def decode(self, _stream):
            yield from (post_seek_frames if self._after_seek else init_frames)

        def close(self) -> None:
            pass

    cont = _FakeContainer()
    monkeypatch.setattr(av, "open", lambda *_a, **_kw: cont)
    roi = RoiRect(0.0, 0.0, 1.0, 1.0)
    with VideoReader(dummy_path) as reader:
        slices = list(reader.iter_roi_frames(roi, start_ms=1000, end_ms=1100))
        assert [sl.pts_ms for sl in slices] == [1500, 1540]
        assert [sl.rel_ms for sl in slices] == [1000, 1040]
        assert [int(sl.bgr_roi[0, 0, 0]) for sl in slices] == [60, 70]

        full_frames = list(reader.iter_full_frames_from_ms(1500))
        assert [pts for _, pts, _ in full_frames] == [1500, 1540]
        assert [int(bgr[0, 0, 0]) for _, _, bgr in full_frames] == [60, 70]
