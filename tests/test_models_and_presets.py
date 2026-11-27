from pathlib import Path

import numpy as np

from transapp.core.models import (
    ExtractConfig,
    ExtractResult,
    FrameTestResult,
    OcrSegment,
    RawSegment,
    RoiRect,
    SubtitleEvent,
)
from transapp.core.presets import PresetStore, RoiPreset


def test_roi_rect_clamps_and_converts_across_resolutions():
    roi = RoiRect(x=-0.1, y=0.8, w=0.95, h=0.3).clamped()
    assert 0.0 <= roi.x <= 1.0 and 0.0 <= roi.y <= 1.0
    assert roi.x + roi.w <= 1.0 and roi.y + roi.h <= 1.0
    x1, y1, x2, y2 = roi.to_pixels(1920, 1080)
    assert (x1, y1, x2, y2) == (0, 864, 1824, 1080)
    x1_hd, y1_hd, x2_hd, y2_hd = roi.to_pixels(1280, 720)
    assert x2_hd > x1_hd and y2_hd == 720


def test_roi_rect_to_pixels_enforces_minimum_size_and_bounds():
    roi = RoiRect(x=1.5, y=1.5, w=0.0, h=-0.2)
    x1, y1, x2, y2 = roi.to_pixels(1920, 1080)
    assert 0 <= x1 < x2 <= 1920
    assert 0 <= y1 < y2 <= 1080
    assert (x2 - x1) >= 1 and (y2 - y1) >= 1


def test_core_dataclasses_and_defaults():
    cfg = ExtractConfig()
    assert cfg.white_v_min == 185
    assert cfg.white_s_max == 65
    assert cfg.outline_v_max == 85
    assert cfg.min_blob_px == 12
    assert cfg.coarse_step == 2
    assert cfg.fg_ratio_min == 0.004
    assert cfg.fg_ratio_edge == 0.002
    assert cfg.split_iou_min == 0.55
    assert cfg.min_duration_ms == 120
    assert cfg.merge_gap_ms == 160
    assert cfg.merge_similarity == 0.85
    assert cfg.conf_threshold == 0.85
    assert cfg.provider == "cpu"

    img = np.zeros((10, 20, 3), dtype=np.uint8)
    mask = np.zeros((10, 20), dtype=np.uint8)
    raw = RawSegment(start_ms=100, end_ms=500, peak_bgr=img, frame_index=5)
    ocr = OcrSegment(start_ms=100, end_ms=500, text="你好", confidence=0.92, thumbnail_png=b"png")
    event = SubtitleEvent(
        index=1,
        start_ms=100,
        end_ms=500,
        text="你好",
        confidence=0.92,
        flagged=False,
        thumbnail_png=b"png",
    )
    result = ExtractResult(
        video_path=Path("demo.mp4"),
        width=1920,
        height=1080,
        duration_ms=5000,
        events=[event],
        elapsed_sec=1.25,
    )
    frame_res = FrameTestResult(frame_index=5, pts_ms=200, mask=mask, text="你好", confidence=0.92)
    assert raw.frame_index == 5
    assert ocr.text == "你好"
    assert result.events == [event]
    assert frame_res.pts_ms == 200


def test_preset_store_roundtrip_outside_video_dir(tmp_path: Path):
    store = PresetStore(config_dir=tmp_path)
    preset = RoiPreset(name="Donghua_Bottom", roi=RoiRect(0.1, 0.8, 0.8, 0.14), conf_threshold=0.82)
    store.save(preset)
    assert store.list_names() == ["Donghua_Bottom"]
    loaded = store.load("Donghua_Bottom")
    assert loaded == preset
    assert store.delete("Donghua_Bottom") is True
    assert store.list_names() == []


def test_preset_store_utf8_update_and_missing_cases(tmp_path: Path, monkeypatch):
    store = PresetStore(config_dir=tmp_path)
    assert store.list_names() == []
    assert store.load("missing") is None
    assert store.delete("missing") is False

    preset_cjk = RoiPreset(
        name="国漫底部",
        roi=RoiRect(0.1, 0.82, 0.8, 0.12),
        ass_template_path="templates/default.ass",
        conf_threshold=0.9,
    )
    store.save(preset_cjk)
    raw_json = (tmp_path / "presets.json").read_text(encoding="utf-8")
    assert "国漫底部" in raw_json

    updated = RoiPreset(
        name="国漫底部",
        roi=RoiRect(0.15, 0.8, 0.7, 0.15),
        ass_template_path=None,
        conf_threshold=0.88,
    )
    store.save(updated)
    assert store.load("国漫底部") == updated
    assert store.list_names() == ["国漫底部"]

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppDataLocal"))
    default_store = PresetStore()
    assert default_store.config_dir == tmp_path / "AppDataLocal" / "TransApp"

    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    fallback_store = PresetStore()
    assert fallback_store.config_dir == Path.home() / ".transapp"


def test_preset_store_corrupt_json_malformed_entries_and_atomic_write(tmp_path: Path, monkeypatch):
    import os
    import transapp.core.presets as presets_mod

    store = PresetStore(config_dir=tmp_path)

    # 1. Corrupt/truncated JSON returns empty cleanly
    store.presets_path.write_text('{"bad": ', encoding="utf-8")
    assert store.list_names() == []
    assert store.load("bad") is None

    # 2. Invalid UTF-8 bytes or non-object JSON root return empty cleanly
    store.presets_path.write_bytes(b"\xff\xfe\xfd")
    assert store.list_names() == []
    assert store.load("any") is None

    store.presets_path.write_text("[1, 2, 3]", encoding="utf-8")
    assert store.list_names() == []

    # 3. Malformed entries return None on load() without crashing
    import json

    malformed_payload = {
        "not_a_dict": "oops",
        "missing_name": {"roi": {"x": 0.1, "y": 0.8, "w": 0.8, "h": 0.1}},
        "bad_roi_type": {"name": "bad_roi_type", "roi": "oops"},
        "bad_coord_val": {"name": "bad_coord_val", "roi": {"x": "bad", "y": 0.8, "w": 0.8, "h": 0.1}},
        "bool_coord": {"name": "bool_coord", "roi": {"x": True, "y": 0.8, "w": 0.8, "h": 0.1}},
        "bad_conf": {
            "name": "bad_conf",
            "roi": {"x": 0.1, "y": 0.8, "w": 0.8, "h": 0.1},
            "conf_threshold": "not_float",
        },
        "out_of_range_conf": {
            "name": "out_of_range_conf",
            "roi": {"x": 0.1, "y": 0.8, "w": 0.8, "h": 0.1},
            "conf_threshold": 1.5,
        },
        "valid_one": {
            "name": "valid_one",
            "roi": {"x": 0.1, "y": 0.8, "w": 0.8, "h": 0.1},
            "conf_threshold": 0.85,
        },
    }
    store.presets_path.write_text(json.dumps(malformed_payload), encoding="utf-8")
    assert "valid_one" in store.list_names()
    for bad_key in (
        "not_a_dict",
        "missing_name",
        "bad_roi_type",
        "bad_coord_val",
        "bool_coord",
        "bad_conf",
        "out_of_range_conf",
    ):
        assert store.load(bad_key) is None
    loaded_valid = store.load("valid_one")
    assert loaded_valid == RoiPreset(
        name="valid_one",
        roi=RoiRect(0.1, 0.8, 0.8, 0.1),
        conf_threshold=0.85,
    )

    # 4. Atomic write via os.replace preserves previous file if replace fails
    replace_calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def spy_replace(src, dst):
        replace_calls.append((Path(src), Path(dst)))
        return real_replace(src, dst)

    monkeypatch.setattr(presets_mod.os, "replace", spy_replace)
    store.save(RoiPreset(name="atomic", roi=RoiRect(0.2, 0.7, 0.6, 0.2)))
    assert len(replace_calls) == 1
    assert replace_calls[0][0].parent == store.config_dir
    assert replace_calls[0][1] == store.presets_path
    assert [p for p in store.config_dir.glob("*.tmp")] == []

    before_bytes = store.presets_path.read_bytes()
    monkeypatch.setattr(
        presets_mod.os,
        "replace",
        lambda _s, _d: (_ for _ in ()).throw(OSError("disk full")),
    )
    import pytest

    with pytest.raises(OSError, match="disk full"):
        store.save(RoiPreset(name="failed_atomic", roi=RoiRect(0.1, 0.1, 0.2, 0.2)))
    assert store.presets_path.read_bytes() == before_bytes
    assert [p for p in store.config_dir.glob("*.tmp")] == []


def test_roi_rect_clamped_handles_nan_and_inf():
    import math

    clamped = RoiRect(float("nan"), float("inf"), float("-inf"), 0.2).clamped()
    for val in (clamped.x, clamped.y, clamped.w, clamped.h):
        assert math.isfinite(val)
    assert 0.0 <= clamped.x <= 1.0
    assert 0.0 <= clamped.y <= 1.0
    assert clamped.w >= 0.01
    assert clamped.h >= 0.01
    assert clamped.x + clamped.w <= 1.0
    assert clamped.y + clamped.h <= 1.0


def test_preset_store_save_rejects_blank_name_and_non_finite_roi(tmp_path: Path):
    import pytest

    store = PresetStore(config_dir=tmp_path)

    with pytest.raises(ValueError):
        store.save(RoiPreset(name="   ", roi=RoiRect(0.1, 0.8, 0.8, 0.12)))

    with pytest.raises(ValueError):
        store.save(RoiPreset(name="", roi=RoiRect(0.1, 0.8, 0.8, 0.12)))

    for bad_roi in (
        RoiRect(float("nan"), 0.8, 0.8, 0.12),
        RoiRect(0.1, float("inf"), 0.8, 0.12),
        RoiRect(0.1, 0.8, float("-inf"), 0.12),
        RoiRect(0.1, 0.8, 0.8, float("nan")),
    ):
        with pytest.raises(ValueError):
            store.save(RoiPreset(name="bad_roi", roi=bad_roi))

    assert not store.presets_path.exists()
    assert store.list_names() == []
