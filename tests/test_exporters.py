from __future__ import annotations

from pathlib import Path

from transapp.core.models import SubtitleEvent
from transapp.exporters import (
    export_ass as pkg_export_ass,
    export_srt as pkg_export_srt,
    format_ass_time as pkg_format_ass_time,
    format_srt_time as pkg_format_srt_time,
)
from transapp.exporters.ass_exporter import export_ass, format_ass_time
from transapp.exporters.srt_exporter import export_srt, format_srt_time


def _event(
    index: int,
    start_ms: int,
    end_ms: int,
    text: str,
    confidence: float = 0.95,
    flagged: bool = False,
) -> SubtitleEvent:
    return SubtitleEvent(
        index=index,
        start_ms=start_ms,
        end_ms=end_ms,
        text=text,
        confidence=confidence,
        flagged=flagged,
        thumbnail_png=b"png",
    )


def test_exporters_package_reexports():
    assert pkg_export_ass is export_ass
    assert pkg_export_srt is export_srt
    assert pkg_format_ass_time is format_ass_time
    assert pkg_format_srt_time is format_srt_time


def test_format_ass_time_rounding_carry_and_clamping():
    assert format_ass_time(-100) == "0:00:00.00"
    assert format_ass_time(0) == "0:00:00.00"
    assert format_ass_time(3723456) == "1:02:03.46"
    # Centisecond carry must roll into seconds/minutes/hours, never emit .100
    assert format_ass_time(9999) == "0:00:10.00"
    assert format_ass_time(59999) == "0:01:00.00"
    assert format_ass_time(3599999) == "1:00:00.00"


def test_format_srt_time_and_clamping():
    assert format_srt_time(-50) == "00:00:00,000"
    assert format_srt_time(0) == "00:00:00,000"
    assert format_srt_time(3723456) == "01:02:03,456"


def test_export_ass_default_layout_actor_flag_and_newlines(tmp_path: Path):
    out_path = tmp_path / "nested" / "测试_—_导出.ass"
    events = [
        _event(1, 1000, 2500, "第一行\n第二行", confidence=0.96, flagged=False),
        _event(2, 3000, 3002, "可疑文本\r\n换行", confidence=0.74, flagged=True),
        _event(3, 5000, 4900, "倒挂时间", confidence=0.90, flagged=False),
    ]

    result = export_ass(events, out_path, video_width=1920, video_height=1080)
    assert result == out_path
    raw_bytes = out_path.read_bytes()
    assert not raw_bytes.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw_bytes

    content = raw_bytes.decode("utf-8")
    assert "[Script Info]" in content
    assert "ScriptType: v4.00+" in content
    assert "ScaledBorderAndShadow: yes" in content
    assert "PlayResX: 1920" in content
    assert "PlayResY: 1080" in content
    assert "[V4+ Styles]" in content
    assert "Style: Default," in content
    assert "[Events]" in content
    assert "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text" in content

    dialogue_lines = [line for line in content.splitlines() if line.startswith("Dialogue:")]
    assert len(dialogue_lines) == 3

    fields_0 = dialogue_lines[0].removeprefix("Dialogue: ").split(",", 9)
    assert fields_0[3] == "Default"
    assert fields_0[4] == ""
    assert fields_0[9] == r"第一行\N第二行"

    fields_1 = dialogue_lines[1].removeprefix("Dialogue: ").split(",", 9)
    assert fields_1[1] == "0:00:03.00"
    assert fields_1[2] == "0:00:03.01"  # clamped >= start_cs + 1
    assert fields_1[3] == "Default"
    assert fields_1[4] == "CHECK_0.74"
    assert fields_1[9] == r"可疑文本\N换行"
    assert "CHECK_" not in fields_1[9]

    fields_2 = dialogue_lines[2].removeprefix("Dialogue: ").split(",", 9)
    assert fields_2[1] == "0:00:05.00"
    assert fields_2[2] == "0:00:05.01"


def test_export_ass_preserves_bom_crlf_template_and_uses_first_style(tmp_path: Path):
    template_path = tmp_path / "template.ass"
    template_text = (
        "[Script Info]\r\n"
        "Title: Custom Template\r\n"
        "ScriptType: v4.00+\r\n"
        "PlayResX: 1280\r\n"
        "PlayResY: 720\r\n"
        "\r\n"
        "[Aegisub Project Garbage]\r\n"
        "Last Style Storage: Default\r\n"
        "\r\n"
        "[V4+ Styles]\r\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\r\n"
        "Style: DonghuaStyle,Source Han Sans SC,52,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
        "0,0,0,0,100,100,0,0,1,2.5,1,2,20,20,30,1\r\n"
        "Style: SecondaryStyle,Arial,40,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,"
        "0,0,0,0,100,100,0,0,1,2,1,2,20,20,30,1\r\n"
        "\r\n"
        "[Events]\r\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\r\n"
        "Dialogue: 0,0:00:00.00,0:00:01.00,DonghuaStyle,,0,0,0,,OLD LINE SHOULD BE REPLACED\r\n"
    )
    template_path.write_bytes(b"\xef\xbb\xbf" + template_text.encode("utf-8"))

    out_path = tmp_path / "out.ass"
    events = [_event(1, 500, 1800, "你好，世界", confidence=0.68, flagged=True)]

    export_ass(events, out_path, video_width=1920, video_height=1080, template_path=template_path)

    raw_bytes = out_path.read_bytes()
    assert not raw_bytes.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw_bytes

    content = raw_bytes.decode("utf-8")
    assert "Title: Custom Template" in content
    assert "[Aegisub Project Garbage]" in content
    assert "Style: DonghuaStyle,Source Han Sans SC,52" in content
    assert "OLD LINE SHOULD BE REPLACED" not in content

    dialogue_lines = [line for line in content.splitlines() if line.startswith("Dialogue:")]
    assert len(dialogue_lines) == 1
    fields = dialogue_lines[0].removeprefix("Dialogue: ").split(",", 9)
    assert fields[3] == "DonghuaStyle"
    assert fields[4] == "CHECK_0.68"
    assert fields[9] == "你好，世界"


def test_export_ass_template_without_events_or_styles_falls_back_gracefully(tmp_path: Path):
    template_path = tmp_path / "minimal_template.ass"
    template_path.write_text(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n",
        encoding="utf-8",
    )

    out_path = tmp_path / "minimal_out.ass"
    export_ass([_event(1, 0, 1000, "测试")], out_path, video_width=1920, video_height=1080, template_path=template_path)

    content = out_path.read_text(encoding="utf-8")
    assert "[Events]" in content
    assert "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text" in content
    assert "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,测试" in content


def test_export_srt_clean_utf8_unicode_path_sequential_and_no_check_markers(tmp_path: Path):
    out_path = tmp_path / "sub" / "测试_—_导出.srt"
    events = [
        _event(10, 1000, 2500, "第一句\r\n第二行", confidence=0.74, flagged=True),
        _event(25, 3000, 3000, "同一毫秒", confidence=0.98, flagged=False),
    ]

    result = export_srt(events, out_path)
    assert result == out_path
    raw_bytes = out_path.read_bytes()
    assert not raw_bytes.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw_bytes

    content = raw_bytes.decode("utf-8")
    assert "CHECK_" not in content
    assert r"\N" not in content
    expected = (
        "1\n"
        "00:00:01,000 --> 00:00:02,500\n"
        "第一句\n"
        "第二行\n"
        "\n"
        "2\n"
        "00:00:03,000 --> 00:00:03,001\n"
        "同一毫秒\n"
    )
    assert content == expected


def test_export_srt_empty_events(tmp_path: Path):
    out_path = tmp_path / "empty.srt"
    export_srt([], out_path)
    assert out_path.read_bytes() == b""
