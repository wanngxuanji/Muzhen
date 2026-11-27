from __future__ import annotations

from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import av
import numpy as np
import pytest

from transapp.core.models import RoiRect
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
            img[:, :] = (30 + idx * 35, 60 + idx * 25, 220 - idx * 30)
            # Bright band inside bottom ROI region (y in [0.8, 0.95]).
            y1 = int(height * 0.8)
            y2 = int(height * 0.95)
            img[y1:y2, :] = (240 - idx * 20, 240 - idx * 15, 240 - idx * 10)

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
        # Missing PTS falls back to DTS, duplicate/regressing ticks advance monotonically.
        p0 = reader._frame_pts_ms(_FakeFrame(pts=None, dts=200), stream, 0, None)
        p1 = reader._frame_pts_ms(_FakeFrame(pts=200, dts=200), stream, 1, p0)
        p2 = reader._frame_pts_ms(_FakeFrame(pts=None, dts=None), stream, 2, p1)
        assert p0 == 200
        assert p1 == 201
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

    # Case 1: stream.duration == 0 falls back to container.duration without probing frames.
    proxies: list[_ContainerProxy] = []

    def _open_case1(path, *args, **kwargs):
        proxy = _ContainerProxy(real_open(path, *args, **kwargs), zero_dims=False)
        proxies.append(proxy)
        return proxy

    monkeypatch.setattr(av, "open", _open_case1)
    with VideoReader(video_path) as reader:
        assert reader.metadata.duration_ms == 2500
        assert proxies[0].decoded_count == 0

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
