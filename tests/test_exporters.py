from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from transapp.core.models import SubtitleEvent
from transapp.exporters import (
    export_ass as pkg_export_ass,
    export_srt as pkg_export_srt,
    format_ass_time as pkg_format_ass_time,
    format_srt_time as pkg_format_srt_time,
    list_ass_template_styles as pkg_list_ass_template_styles,
    prepare_export_cues as pkg_prepare_export_cues,
    snap_ass_centiseconds as pkg_snap_ass_centiseconds,
)
from transapp.exporters.ass_exporter import export_ass, format_ass_time, list_ass_template_styles
from transapp.exporters.sanitize import prepare_export_cues, snap_ass_centiseconds, write_text_atomic
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
    assert pkg_list_ass_template_styles is list_ass_template_styles
    assert pkg_prepare_export_cues is prepare_export_cues
    assert pkg_snap_ass_centiseconds is snap_ass_centiseconds


def test_format_ass_time_rounding_carry_and_clamping():
    assert format_ass_time(-100) == "0:00:00.00"
    assert format_ass_time(0) == "0:00:00.00"
    assert format_ass_time(3723456) == "1:02:03.46"
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
    assert fields_1[2] == "0:00:03.01"
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


def test_export_srt_collapses_blank_lines_and_skips_empty_cues(tmp_path: Path):
    out_path = tmp_path / "sanitized.srt"
    events = [
        _event(1, 1000, 2000, "  第一行 \n\n  第二行  \r\n\r\n"),
        _event(2, 2100, 2500, "   \n\t \r\n  "),
        _event(3, 2600, 3000, ""),
        _event(4, 3100, 4000, "\n  第三句  \n"),
    ]
    export_srt(events, out_path)
    content = out_path.read_text(encoding="utf-8")
    expected = (
        "1\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "第一行\n"
        "第二行\n"
        "\n"
        "2\n"
        "00:00:03,100 --> 00:00:04,000\n"
        "第三句\n"
    )
    assert content == expected


def _parse_ass_ts_cs(ts: str) -> int:
    hms, cs_str = ts.split(".")
    h_str, m_str, s_str = hms.split(":")
    return ((int(h_str) * 3600 + int(m_str) * 60 + int(s_str)) * 100) + int(cs_str)


def _parse_srt_ts_ms(ts: str) -> int:
    hms, ms_str = ts.split(",")
    h_str, m_str, s_str = hms.split(":")
    return ((int(h_str) * 3600 + int(m_str) * 60 + int(s_str)) * 1000) + int(ms_str)


def _build_cfr_pts(fps: Fraction, count: int) -> list[int]:
    return [round(Fraction(idx) * 1000 / fps) for idx in range(count)]


def _build_vfr_pts(count: int) -> list[int]:
    rates = [Fraction(12, 1), Fraction(24, 1), Fraction(60, 1)]
    pts: list[int] = [0]
    t = Fraction(0, 1)
    for idx in range(1, count):
        fps = rates[(idx // 15) % len(rates)]
        t += Fraction(1000, 1) / fps
        pts.append(round(t))
    return pts


def test_ass_midpoint_snapping_100_percent_libass_visibility_oracle(tmp_path: Path):
    assert snap_ass_centiseconds(5, 25, None) == (1, 3)
    assert snap_ass_centiseconds(15, 35, [0]) == (2, 4)

    timelines: list[tuple[str, list[int]]] = [
        ("23.976", _build_cfr_pts(Fraction(24000, 1001), 1600)),
        ("25", _build_cfr_pts(Fraction(25, 1), 1600)),
        ("29.97", _build_cfr_pts(Fraction(30000, 1001), 1600)),
        ("59.94", _build_cfr_pts(Fraction(60000, 1001), 1600)),
        ("vfr", _build_vfr_pts(1600)),
    ]

    total_cues_checked = 0
    for label, pts in timelines:
        events: list[SubtitleEvent] = []
        frame_ranges: list[tuple[int, int]] = []
        cursor = 0
        for cue_idx in range(200):
            if cue_idx == 199:
                f_start = len(pts) - 5
                f_end = len(pts)
                start_ms = pts[f_start]
                end_ms = pts[-1] + max(16, pts[-1] - pts[-2])
            else:
                gap = 0 if (cue_idx % 3 == 0) else (1 if cue_idx % 3 == 1 else 2)
                if cue_idx == 0:
                    gap = 0
                dur_frames = 1 if (cue_idx % 11 == 0) else (2 + (cue_idx % 6))
                f_start = cursor + gap
                f_end = f_start + dur_frames
                cursor = f_end
                start_ms = pts[f_start]
                end_ms = pts[f_end]
            events.append(_event(cue_idx + 1, start_ms, end_ms, f"{label}_cue_{cue_idx}"))
            frame_ranges.append((f_start, f_end))

        out_path = tmp_path / f"oracle_{label}.ass"
        export_ass(events, out_path, width=1920, height=1080, frame_pts_ms=pts)

        dialogues = [
            line for line in out_path.read_text(encoding="utf-8").splitlines() if line.startswith("Dialogue:")
        ]
        assert len(dialogues) == 200

        prev_end_cs = -1
        for cue_idx, (line, (f_start, f_end)) in enumerate(zip(dialogues, frame_ranges)):
            fields = line.removeprefix("Dialogue: ").split(",", 9)
            start_cs = _parse_ass_ts_cs(fields[1])
            end_cs = _parse_ass_ts_cs(fields[2])
            assert start_cs < end_cs, f"{label} cue {cue_idx}: zero/negative duration"
            assert prev_end_cs <= start_cs, f"{label} cue {cue_idx}: overlap {prev_end_cs} > {start_cs}"
            prev_end_cs = end_cs

            check_lo = max(0, f_start - 3)
            check_hi = min(len(pts), f_end + 3)
            for f in range(check_lo, check_hi):
                visible = (start_cs * 10) <= pts[f] < (end_cs * 10)
                expected = f_start <= f < f_end
                assert visible == expected, (
                    f"{label} cue {cue_idx} frame {f} (pts={pts[f]}): "
                    f"expected={expected} got={visible} ({start_cs * 10}..{end_cs * 10})"
                )
            total_cues_checked += 1

    assert total_cues_checked == 1000


def test_ass_escapes_braces_and_literal_backslash_n_and_preserves_fonts_section(tmp_path: Path):
    template_missing_res = tmp_path / "tpl_missing_res.ass"
    template_missing_res.write_text(
        "[Script Info]\n"
        "Title: Missing Res\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 0\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: PrimaryStyle,Microsoft YaHei,54,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        "-1,0,0,0,100,100,0,0,1,2,1,2,20,20,30,1\n"
        "Style: AltStyle,Arial,44,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        "0,0,0,0,100,100,0,0,1,2,1,2,20,20,30,1\n"
        "Style: PrimaryStyle,DuplicateIgnored,40,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        "0,0,0,0,100,100,0,0,1,2,1,2,20,20,30,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:00.00,0:00:01.00,PrimaryStyle,,0,0,0,,REPLACE ME\n"
        "\n"
        "[Fonts]\n"
        "fontname: custom_0.ttf\n"
        "!!!!encodedfontpayload!!!!\n"
        "\n"
        "[Graphics]\n"
        "filename: logo.png\n"
        "!!!!encodedgraphicpayload!!!!\n",
        encoding="utf-8",
    )

    assert list_ass_template_styles(template_missing_res) == ["PrimaryStyle", "AltStyle"]

    out_path = tmp_path / "escaped.ass"
    events = [
        _event(1, 1000, 2500, r"{note} literal\Nbreak\hspace\nend" + "\n第二行{tag}"),
    ]
    export_ass(
        events,
        out_path,
        width=1920,
        height=1080,
        template_path=template_missing_res,
        style_name="AltStyle",
    )

    content = out_path.read_text(encoding="utf-8")
    assert "PlayResX: 1920" in content
    assert "PlayResY: 1080" in content
    assert "PlayResX: 0" not in content
    assert "REPLACE ME" not in content

    events_pos = content.index("[Events]")
    dialogue_pos = content.index("Dialogue:")
    fonts_pos = content.index("[Fonts]")
    graphics_pos = content.index("[Graphics]")
    assert events_pos < dialogue_pos < fonts_pos < graphics_pos
    assert "!!!!encodedfontpayload!!!!" in content
    assert "!!!!encodedgraphicpayload!!!!" in content

    dialogue_line = [line for line in content.splitlines() if line.startswith("Dialogue:")][0]
    fields = dialogue_line.removeprefix("Dialogue: ").split(",", 9)
    assert fields[3] == "AltStyle"
    text_field = fields[9]
    assert "{" not in text_field and "}" not in text_field
    assert "｛note｝" in text_field
    assert "｛tag｝" in text_field
    assert r"\N第二行｛tag｝" in text_field
    assert "\\\u200bNbreak" in text_field
    assert "\\\u200bhspace" in text_field
    assert "\\\u200bnend" in text_field

    template_valid_res = tmp_path / "tpl_valid_res.ass"
    template_valid_res.write_text(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1280\nPlayResY: 720\n",
        encoding="utf-8",
    )
    out_valid = tmp_path / "valid_res.ass"
    export_ass(
        [_event(1, 0, 1000, "测试")],
        out_valid,
        width=1920,
        height=1080,
        template_path=template_valid_res,
    )
    valid_content = out_valid.read_text(encoding="utf-8")
    assert "PlayResX: 1280" in valid_content
    assert "PlayResY: 720" in valid_content
    assert "PlayResX: 1920" not in valid_content
    assert "[V4+ Styles]" in valid_content
    assert "Style: Default," in valid_content


def test_ass_and_srt_empty_cue_parity_chronological_sort_and_overlap_clamp(tmp_path: Path):
    events = [
        _event(5, 3000, 4000, "  重叠后句  "),
        _event(1, 1000, 1004, "碰撞前"),
        _event(2, 1004, 2000, "碰撞后"),
        _event(3, 1500, 1800, "   \r\n\t  "),
        _event(4, 2000, 3500, "重叠前句"),
        _event(6, 5000, 6000, ""),
        _event(7, 4500, 5000, "同起点A"),
        _event(7, 4500, 5000, "同起点B"),
    ]
    original_snapshot = [(ev.index, ev.start_ms, ev.end_ms, ev.text) for ev in events]

    prepared = prepare_export_cues(events)
    assert [(ev.index, ev.start_ms, ev.end_ms, ev.text) for ev in events] == original_snapshot
    assert [ev.text for ev in prepared] == [
        "碰撞前",
        "碰撞后",
        "重叠前句",
        "重叠后句",
        "同起点A",
        "同起点B",
    ]
    assert all(p is not o for p in prepared for o in events)

    ass_path = tmp_path / "parity.ass"
    srt_path = tmp_path / "parity.srt"
    export_ass(events, ass_path, width=1920, height=1080)
    export_srt(events, srt_path)

    assert [(ev.index, ev.start_ms, ev.end_ms, ev.text) for ev in events] == original_snapshot

    ass_dialogues = [
        line for line in ass_path.read_text(encoding="utf-8").splitlines() if line.startswith("Dialogue:")
    ]
    srt_blocks = [
        block for block in srt_path.read_text(encoding="utf-8").strip().split("\n\n") if block.strip()
    ]
    assert len(ass_dialogues) == len(srt_blocks) == 6

    ass_prev_end = -1
    ass_texts: list[str] = []
    for line in ass_dialogues:
        fields = line.removeprefix("Dialogue: ").split(",", 9)
        s_cs = _parse_ass_ts_cs(fields[1])
        e_cs = _parse_ass_ts_cs(fields[2])
        assert s_cs < e_cs
        assert ass_prev_end <= s_cs
        ass_prev_end = e_cs
        ass_texts.append(fields[9])
    assert ass_texts == ["碰撞前", "碰撞后", "重叠前句", "重叠后句", "同起点A", "同起点B"]

    srt_prev_end = -1
    srt_texts: list[str] = []
    for idx, block in enumerate(srt_blocks, start=1):
        lines = block.splitlines()
        assert lines[0] == str(idx)
        start_s, end_s = lines[1].split(" --> ")
        s_ms = _parse_srt_ts_ms(start_s)
        e_ms = _parse_srt_ts_ms(end_s)
        assert s_ms < e_ms
        assert srt_prev_end <= s_ms
        srt_prev_end = e_ms
        srt_texts.append("\n".join(lines[2:]))
    assert srt_texts == ["碰撞前", "碰撞后", "重叠前句", "重叠后句", "同起点A", "同起点B"]


def test_ass_snapping_edge_cases_trimmed_eof_short_and_vfr_fallback(tmp_path: Path):
    trimmed_pts = [90007, 90049, 90090, 90132, 90174]
    s0, e0 = snap_ass_centiseconds(90007, 90090, trimmed_pts)
    assert s0 == 9000
    assert [s0 * 10 <= p < e0 * 10 for p in trimmed_pts] == [True, True, False, False, False]

    s_eof, e_eof = snap_ass_centiseconds(90132, 90216, trimmed_pts)
    assert [s_eof * 10 <= p < e_eof * 10 for p in trimmed_pts] == [False, False, False, True, True]

    short_pts = [0, 42, 83]
    c1_s, c1_e = snap_ass_centiseconds(42, 84, short_pts)
    c2_s, c2_e = snap_ass_centiseconds(84, 200, short_pts)
    assert c1_e == c2_s == 9
    assert [c1_s * 10 <= p < c1_e * 10 for p in short_pts] == [False, True, True]
    assert [c2_s * 10 <= p < c2_e * 10 for p in short_pts] == [False, False, False]

    t_s, t_e = snap_ass_centiseconds(42, 44, short_pts)
    assert [t_s * 10 <= p < t_e * 10 for p in short_pts] == [False, True, False]

    vfr_tight = [0, 12, 18, 50]
    v_s, v_e = snap_ass_centiseconds(18, 50, vfr_tight)
    assert v_s == 2
    assert v_e > v_s

    cascade_events = [
        _event(1, 1000, 1004, r"C:\path\to\file"),
        _event(2, 1004, 1008, "中段"),
        _event(3, 1008, 2000, "尾段"),
    ]
    ass_path = tmp_path / "cascade.ass"
    srt_path = tmp_path / "cascade.srt"
    export_ass(cascade_events, ass_path, width=1920, height=1080)
    export_srt(cascade_events, srt_path)

    ass_lines = [
        line for line in ass_path.read_text(encoding="utf-8").splitlines() if line.startswith("Dialogue:")
    ]
    assert len(ass_lines) == 3
    ass_spans = []
    for line in ass_lines:
        fields = line.removeprefix("Dialogue: ").split(",", 9)
        ass_spans.append((_parse_ass_ts_cs(fields[1]), _parse_ass_ts_cs(fields[2]), fields[9]))
    assert ass_spans[0] == (100, 101, r"C:\path\to\file")
    assert "\u200b" not in ass_spans[0][2]
    assert ass_spans[1] == (101, 102, "中段")
    assert ass_spans[2] == (102, 200, "尾段")

    srt_blocks = [b for b in srt_path.read_text(encoding="utf-8").strip().split("\n\n") if b.strip()]
    assert len(srt_blocks) == 3
    srt_prev_end = -1
    for block in srt_blocks:
        s_str, e_str = block.splitlines()[1].split(" --> ")
        s_ms, e_ms = _parse_srt_ts_ms(s_str), _parse_srt_ts_ms(e_str)
        assert s_ms < e_ms
        assert srt_prev_end <= s_ms
        srt_prev_end = e_ms


def test_snap_ass_centiseconds_ntsc_1ms_overshoot_stays_on_nearest_frame():
    pts_23976 = [417, 459, 500, 542]
    exact_s, exact_e = snap_ass_centiseconds(500, 542, pts_23976)
    over_s, over_e = snap_ass_centiseconds(501, 542, pts_23976)
    assert over_s == exact_s
    assert over_e == exact_e
    assert 459 < over_s * 10 <= 500
    assert [over_s * 10 <= p < over_e * 10 for p in pts_23976] == [False, False, True, False]

    pts_2997_head = [1034, 1068, 1101]
    exact_s0, exact_e0 = snap_ass_centiseconds(1034, 1101, pts_2997_head)
    over_s0, over_e0 = snap_ass_centiseconds(1035, 1101, pts_2997_head)
    assert over_s0 == exact_s0 == 103
    assert over_e0 == exact_e0
    assert [over_s0 * 10 <= p < over_e0 * 10 for p in pts_2997_head] == [True, True, False]

    pts_2997_mid = [1001, 1034, 1068, 1101]
    exact_sm, exact_em = snap_ass_centiseconds(1034, 1101, pts_2997_mid)
    over_sm, over_em = snap_ass_centiseconds(1035, 1101, pts_2997_mid)
    assert over_sm == exact_sm
    assert over_em == exact_em
    assert 1001 < over_sm * 10 <= 1034
    assert [over_sm * 10 <= p < over_em * 10 for p in pts_2997_mid] == [False, True, True, False]


def test_snap_ass_centiseconds_tail_does_not_bleed_past_unaligned_trim_end():
    pts_23976 = [0, 42, 83, 125]
    start_cs, end_cs = snap_ass_centiseconds(0, 166, pts_23976)
    assert start_cs == 0
    assert end_cs == 16
    assert [start_cs * 10 <= p < end_cs * 10 for p in pts_23976] == [True, True, True, True]

    _, end_cs_169 = snap_ass_centiseconds(0, 169, pts_23976)
    assert end_cs_169 == 16

    _, end_cs_128 = snap_ass_centiseconds(0, 128, pts_23976)
    assert end_cs_128 == 13


def test_ass_template_styles_after_events_empty_styles_and_utf16_bom(tmp_path: Path):
    tpl_reordered = tmp_path / "styles_after_events.ass"
    tpl_reordered.write_text(
        "[Script Info]\n"
        "Title: Reordered\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1920\n"
        "PlayResY: 1080\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:00.00,0:00:01.00,LateStyle,,0,0,0,,OLD\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: LateStyle,Source Han Sans SC,56,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        "0,0,0,0,100,100,0,0,1,2,1,2,20,20,30,1\n"
        "\n"
        "[Fonts]\n"
        "fontname: test.ttf\n"
        "!!!!fontdata!!!!\n"
        "[A9]\n"
        "[0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ]\n"
        "[...!]\n"
        "AFTER_BRACKET_PAYLOAD\n",
        encoding="utf-8",
    )
    out_reordered = tmp_path / "out_reordered.ass"
    export_ass(
        [_event(1, 0, 1000, "顺序测试")],
        out_reordered,
        width=1920,
        height=1080,
        template_path=tpl_reordered,
    )
    reordered_text = out_reordered.read_text(encoding="utf-8")
    assert reordered_text.count("[V4+ Styles]") == 1
    assert reordered_text.index("[Script Info]") < reordered_text.index("[V4+ Styles]")
    assert reordered_text.index("[V4+ Styles]") < reordered_text.index("[Events]")
    assert reordered_text.index("[Events]") < reordered_text.index("[Fonts]")
    assert "OLD" not in reordered_text
    fonts_idx = reordered_text.index("[Fonts]")
    for payload_line in (
        "!!!!fontdata!!!!",
        "[A9]",
        "[0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ]",
        "[...!]",
        "AFTER_BRACKET_PAYLOAD",
    ):
        assert payload_line in reordered_text
        assert reordered_text.index(payload_line) > fonts_idx
    reordered_dialogues = [l for l in reordered_text.splitlines() if l.startswith("Dialogue:")]
    assert len(reordered_dialogues) == 1
    assert reordered_dialogues[0].removeprefix("Dialogue: ").split(",", 9)[3] == "LateStyle"

    tpl_empty_styles = tmp_path / "empty_styles.ass"
    tpl_empty_styles.write_text(
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1280\n"
        "PlayResY: 720\n"
        "\n"
        "[V4+ Styles]\n"
        "\n"
        "[Events]\n",
        encoding="utf-8",
    )
    out_empty_styles = tmp_path / "out_empty_styles.ass"
    export_ass(
        [_event(1, 0, 1000, "空样式测试")],
        out_empty_styles,
        width=1920,
        height=1080,
        template_path=tpl_empty_styles,
    )
    empty_styles_text = out_empty_styles.read_text(encoding="utf-8")
    assert empty_styles_text.count("[V4+ Styles]") == 1
    assert "Format: Name, Fontname," in empty_styles_text
    assert "Style: Default," in empty_styles_text
    assert "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,空样式测试" in empty_styles_text

    utf16_body = (
        "[Script Info]\r\n"
        "Title: UTF16 Template\r\n"
        "ScriptType: v4.00+\r\n"
        "PlayResX: 1920\r\n"
        "PlayResY: 1080\r\n"
        "\r\n"
        "[V4+ Styles]\r\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\r\n"
        "Style: Utf16Style,Microsoft YaHei,54,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        "-1,0,0,0,100,100,0,0,1,2,1,2,20,20,30,1\r\n"
        "\r\n"
        "[Events]\r\n"
    )
    tpl_utf16_le = tmp_path / "tpl_utf16_le.ass"
    tpl_utf16_le.write_bytes(b"\xff\xfe" + utf16_body.encode("utf-16-le"))
    assert list_ass_template_styles(tpl_utf16_le) == ["Utf16Style"]

    out_utf16_le = tmp_path / "out_utf16_le.ass"
    export_ass(
        [_event(1, 100, 900, "编码测试")],
        out_utf16_le,
        width=1920,
        height=1080,
        template_path=tpl_utf16_le,
    )
    out_utf16_bytes = out_utf16_le.read_bytes()
    assert not out_utf16_bytes.startswith((b"\xff\xfe", b"\xfe\xff", b"\xef\xbb\xbf"))
    utf16_out_text = out_utf16_bytes.decode("utf-8")
    assert "Title: UTF16 Template" in utf16_out_text
    assert ",Utf16Style,,0,0,0,,编码测试" in utf16_out_text

    tpl_utf16_be = tmp_path / "tpl_utf16_be.ass"
    tpl_utf16_be.write_bytes(b"\xfe\xff" + utf16_body.encode("utf-16-be"))
    assert list_ass_template_styles(tpl_utf16_be) == ["Utf16Style"]

    out_sanitized = tmp_path / "out_sanitized_style.ass"
    export_ass(
        [_event(1, 0, 1000, "安全样式")],
        out_sanitized,
        width=1920,
        height=1080,
        template_path=tpl_utf16_le,
        style_name="Bad,Style\nX",
    )
    sanitized_lines = out_sanitized.read_text(encoding="utf-8").splitlines()
    sanitized_dialogues = [l for l in sanitized_lines if l.startswith("Dialogue:")]
    assert len(sanitized_dialogues) == 1
    assert "X" not in sanitized_lines
    fields = sanitized_dialogues[0].removeprefix("Dialogue: ").split(",", 9)
    assert len(fields) == 10
    assert "," not in fields[3] and "\n" not in fields[3] and "\r" not in fields[3]
    assert fields[9] == "安全样式"

    out_fallback = tmp_path / "out_fallback_style.ass"
    export_ass(
        [_event(1, 0, 1000, "回退样式")],
        out_fallback,
        width=1920,
        height=1080,
        template_path=tpl_utf16_le,
        style_name=",\r\n,",
    )
    fallback_dialogue = [
        l for l in out_fallback.read_text(encoding="utf-8").splitlines() if l.startswith("Dialogue:")
    ][0]
    assert fallback_dialogue.removeprefix("Dialogue: ").split(",", 9)[3] == "Utf16Style"

    assert list(tmp_path.glob("*.tmp")) == []


def test_write_text_atomic_cleans_temp_file_and_preserves_target_on_error(
    tmp_path: Path, monkeypatch
):
    import transapp.exporters.sanitize as sanitize_mod

    target = tmp_path / "existing.ass"
    target.write_text("ORIGINAL CONTENT\n", encoding="utf-8")

    def fail_replace(src, dst):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(sanitize_mod.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        write_text_atomic(target, "NEW CONTENT\n")

    assert target.read_text(encoding="utf-8") == "ORIGINAL CONTENT\n"
    assert list(tmp_path.glob("*.tmp")) == []

