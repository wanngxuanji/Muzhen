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
            # Distinct color per frame (BGR) so frame identity can be verified.
            img[:, :] = ((30 + idx * 35) % 256, (60 + idx * 25) % 256, (220 - idx * 30) % 256)
            # Bright band inside bottom ROI region (y in [0.8, 0.95]).
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

        # Streaming after seeking should still yield all 6 frames cleanly from start.
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
    reader.close()  # idempotent
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
        # Missing PTS must NOT fall back to B-frame decode-order DTS; synthesizes monotonically.
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
            self.duration = 2_500_000  # 2500 ms in av.time_base
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

    # Case 1: stream.duration == 0 falls back to container.duration, decoding only frame 0 for origin.
    proxies: list[_ContainerProxy] = []

    def _open_case1(path, *args, **kwargs):
        proxy = _ContainerProxy(real_open(path, *args, **kwargs), zero_dims=False)
        proxies.append(proxy)
        return proxy

    monkeypatch.setattr(av, "open", _open_case1)
    with VideoReader(video_path) as reader:
        assert reader.metadata.duration_ms == 2500
        assert proxies[0].decoded_count == 1

    # Case 2: missing dimensions with duration_ms > 0 probes ONLY the first frame.
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
    # Backward-compatible default on VideoMetadata
    legacy_meta = VideoMetadata(width=320, height=180, duration_ms=100, fps=25.0, estimated_frames=3)
    assert legacy_meta.start_pts_ms == 0

    # Non-zero start_pts_ms >= duration_ms
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

        # Falsy frame/stream time_base falls back to 1/1000 instead of raising TypeError
        class _FakeFrame:
            pts = 1234
            dts = None
            time_base = None

        class _FakeStream:
            time_base = None

        assert reader._raw_pts_ms(_FakeFrame(), _FakeStream(), 0, meta.fps) == 1234

    # Init error during _compute_start_pts_ms / _build_metadata must close container
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
    # ExtractResult must default frame_pts_ms to an empty list for backward compatibility.
    res = ExtractResult(
        video_path=tmp_path / "dummy.mp4",
        width=320,
        height=180,
        duration_ms=1000,
        events=[],
        elapsed_sec=0.1,
    )
    assert res.frame_pts_ms == []

    # FrameSlice 3-arg compatibility computes rel_ms from pts_ms when omitted.
    legacy_slice = FrameSlice(1, 40, np.zeros((4, 4, 3), dtype=np.uint8))
    assert legacy_slice.rel_ms == 40

    # Create a real tiny file just to pass Path existence checks; container is proxied.
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

    # Case A: Negative edit-list PTS (-83ms) while stream.start_time is 0.
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
        # Ensure seek to start for negative origin uses a negative offset (not clamped to 0).
        assert any(call_args and call_args[0] < 0 for call_args, _ in neg_container.seek_calls)

        # Missing frame.pts must NOT fall back to B-frame decode-order frame.dts.
        p0 = reader._frame_pts_ms(_FakeFrame(pts=None, dts=9999, tb=tb_ms), neg_stream, 0, None)
        p1 = reader._frame_pts_ms(_FakeFrame(pts=None, dts=5000, tb=tb_ms), neg_stream, 1, p0)
        assert p0 == -83
        assert p1 == -83 + 40

        # Negative edit-list targets must seek cleanly without clamping to 0.
        idx_neg0, pts_neg0, _ = reader.read_frame_at_ms(-83)
        idx_neg1, pts_neg1, _ = reader.read_frame_at_ms(-43)
        assert idx_neg0 == 0
        assert pts_neg0 == -83
        assert idx_neg1 == 1
        assert pts_neg1 == -43

    # Case B: stream.start_time is None while first decoded frame is at 12000ms.
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

    # Test both timebase = 1001/24000 (ticks 0..N) and timebase = 1/24000 (ticks 0, 1001, 2002...).
    tb_frame = Fraction(1001, 24000)
    tb_fine = Fraction(1, 24000)
    stream_frame = _FakeStream(tb_frame)
    stream_fine = _FakeStream(tb_fine)

    with VideoReader(dummy_file) as reader:
        # 1,000 consecutive ticks must match exact rational ms with zero drift.
        prev_pts: int | None = None
        for k in range(1000):
            expected_ms = int(round(Fraction(k * 1001 * 1000, 24000)))
            got_frame = reader._frame_pts_ms(_FakeFrame(k, tb_frame), stream_frame, k, prev_pts)
            got_fine = reader._raw_pts_ms(_FakeFrame(k * 1001, tb_fine), stream_fine, k, 24000 / 1001)
            assert got_frame == expected_ms
            assert got_fine == expected_ms
            prev_pts = got_frame

        # Verify 24+ minutes (34,560 frames = exactly 1,441,440ms on 24000/1001).
        k_24m = 34_560
        expected_24m = int(round(Fraction(k_24m * 1001 * 1000, 24000)))
        assert expected_24m == 1_441_440
        assert reader._raw_pts_ms(_FakeFrame(k_24m, tb_frame), stream_frame, k_24m, 24000 / 1001) == expected_24m
        assert reader._raw_pts_ms(_FakeFrame(k_24m * 1001, tb_fine), stream_fine, k_24m, 24000 / 1001) == expected_24m


def test_video_reader_packet_demux_duration_probe_without_full_decode(tmp_path: Path, monkeypatch):
    """Missing container/stream duration must estimate duration via packet demux without full decode."""
    video_path = tmp_path / "no_duration.mp4"
    pts_list = tuple(200 + i * 40 for i in range(20))  # 200..960ms -> duration ~800ms
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
        # Only frame 0 should be decoded during __init__, never all 20 frames.
        assert proxies[0].decoded_count == 1
        assert proxies[0].demuxed_packets >= 20
        assert 760 <= meta.duration_ms <= 840
        assert meta.start_pts_ms == 200


def test_video_reader_iter_roi_frames_trim_window_preserves_absolute_rel_ms(tmp_path: Path):
    """Trimmed iter_roi_frames(start_ms, end_ms) seeks cleanly and preserves absolute pts_ms & rel_ms."""
    video_path = tmp_path / "trim_window.mp4"
    width, height = 320, 180
    total_frames = 40
    origin_ms = 480  # exact multiple of 40ms for 25fps codec timebase
    pts_list = [origin_ms + i * 40 for i in range(total_frames)]  # rel_ms: 0, 40, ..., 1560

    with av.open(str(video_path), mode="w") as container:
        stream = container.add_stream("libx264", rate=25)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 1000)
        stream.options = {"g": "5", "keyint_min": "5"}

        for idx, pts_ms in enumerate(pts_list):
            img = np.zeros((height, width, 3), dtype=np.uint8)
            # Encode frame index into green channel brightness inside ROI to verify exact frame identity.
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

        # Middle window: rel_ms in [400, 1000] -> original frames 10..25 (16 frames).
        window = list(reader.iter_roi_frames(roi, start_ms=400, end_ms=1000))
        assert len(window) == 16
        assert [sl.frame_index for sl in window] == list(range(16))
        assert [sl.rel_ms for sl in window] == [400 + i * 40 for i in range(16)]
        assert [sl.pts_ms for sl in window] == [origin_ms + 400 + i * 40 for i in range(16)]
        # Verify frame content corresponds to original frame 10 (green ~ 15 + 10*5 = 65), not frame 0 (15).
        first_green = float(window[0].bgr_roi[:, :, 1].mean())
        assert abs(first_green - 65.0) < 10.0

        # End-only trim: rel_ms in [0, 200] -> frames 0..5 (6 frames).
        head = list(reader.iter_roi_frames(roi, end_ms=200))
        assert [sl.rel_ms for sl in head] == [0, 40, 80, 120, 160, 200]

        # Start-only trim: rel_ms in [1400, 1560] -> frames 35..39 (5 frames).
        tail = list(reader.iter_roi_frames(roi, start_ms=1400))
        assert [sl.rel_ms for sl in tail] == [1400, 1440, 1480, 1520, 1560]

        # Inverted window yields empty.
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

    # Subtest A: Post-seek consecutive missing-PTS frames on 24000/1001 keep 0ms drift and anchor at target_raw_ms.
    init_ntsc = [_FakeFrame(pts=0, dts=0, tb=tb_ms)]
    missing_ntsc = [_FakeFrame(pts=None, dts=9999 - i, tb=tb_ms) for i in range(101)]
    stream_ntsc = _FakeStream(start_time=0, time_base=tb_ms, rate=ntsc_rate)
    cont_ntsc = _FakeContainer(stream_ntsc, init_ntsc, seek_frames=missing_ntsc)

    monkeypatch.setattr(av, "open", lambda *_a, **_kw: cont_ntsc)
    with VideoReader(dummy_file) as reader:
        slices = list(reader.iter_roi_frames(roi, start_ms=1000, end_ms=6000))
        assert len(slices) == 101
        assert slices[0].pts_ms == 1000
        assert slices[0].rel_ms == 1000
        expected_100 = 1000 + int(round(Fraction(100 * 1000) / ntsc_rate))
        assert expected_100 == 5171
        assert slices[100].pts_ms == expected_100
        assert slices[100].rel_ms == expected_100

        # _decode_near_ms (via read_frame_at_ms) also anchors post-seek missing PTS at seek_ms (not origin 0).
        _idx, got_pts, _bgr = reader.read_frame_at_ms(1200)
        assert got_pts == 1200

    # Subtest B: Real pre-roll frame.pts < target_raw_ms arriving after a missing-PTS frame must NOT clamp forward.
    init_preroll = [_FakeFrame(pts=500, dts=500, tb=tb_ms)]
    seek_preroll = [
        _FakeFrame(pts=None, dts=800, tb=tb_ms),  # synthesized as target_raw_ms=900 (guess)
        _FakeFrame(pts=820, dts=820, tb=tb_ms),   # real pre-roll < 900 -> rel_ms=320, must be skipped
        _FakeFrame(pts=None, dts=860, tb=tb_ms),  # synthesized from 820 -> 860 (rel_ms=360), skipped
        _FakeFrame(pts=900, dts=900, tb=tb_ms),   # rel_ms=400 -> yielded
        _FakeFrame(pts=940, dts=940, tb=tb_ms),   # rel_ms=440 -> yielded
    ]
    stream_preroll = _FakeStream(start_time=500, time_base=tb_ms, rate=Fraction(25, 1))
    cont_preroll = _FakeContainer(stream_preroll, init_preroll, seek_frames=seek_preroll)

    monkeypatch.setattr(av, "open", lambda *_a, **_kw: cont_preroll)
    with VideoReader(dummy_file) as reader:
        slices = list(reader.iter_roi_frames(roi, start_ms=400, end_ms=600))
        # Frame 0 (unanchored guess=900) + frames 3 & 4 (real 900 & 940); pre-roll 820 & 860 skipped.
        assert [sl.rel_ms for sl in slices] == [400, 400, 440]
        assert [sl.pts_ms for sl in slices] == [900, 900, 940]

    # Subtest C: Demux duration probe with unknown origin (frame 0 pts=None, stream.start_time=None)
    # must anchor at min_pts_ms (12000), NOT inflate duration to 13000 via fallback 0.
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
