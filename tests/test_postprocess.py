from __future__ import annotations

import pytest

from transapp.core import (
    normalize_cjk_text as core_normalize_cjk_text,
    postprocess_segments as core_postprocess_segments,
    text_similarity as core_text_similarity,
)
from transapp.core.models import ExtractConfig, OcrSegment
from transapp.core.postprocess import (
    normalize_cjk_text,
    postprocess_segments,
    text_similarity,
)


def test_core_reexports_postprocess_functions():
    assert core_normalize_cjk_text is normalize_cjk_text
    assert core_text_similarity is text_similarity
    assert core_postprocess_segments is postprocess_segments


def test_normalize_cjk_text_strips_spurious_cjk_spaces_keeps_newlines():
    assert normalize_cjk_text(" 大家 都 怎么 样 了 \n 不用 亲眼 ") == "大家都怎么样了\n不用亲眼"


def test_normalize_cjk_text_handles_crlf_ascii_words_fullwidth_punct_and_blank_lines():
    raw = "\r\n  \r\n  Hello   world  2026 \r\n\t 你 好 ， 世 界 ！ \n\n  第 10 集 ：   Alpha  Beta  \n  "
    expected = "Hello world 2026\n你好，世界！\n第10集：Alpha Beta"
    assert normalize_cjk_text(raw) == expected
    assert normalize_cjk_text("这是 Hello   World 测试") == "这是Hello World测试"
    assert normalize_cjk_text("   \r\n\t \n  ") == ""


def test_text_similarity_cases():
    assert text_similarity("", "") == 1.0
    assert text_similarity("   \n ", "") == 1.0
    assert text_similarity("你好", "") == 0.0
    assert text_similarity("", "你好") == 0.0
    assert text_similarity("不用亲眼\n看见", "不用 亲眼 看见") == 1.0

    # 1-char difference on a 12-char CJK line -> 11/12 ~= 0.917 >= 0.85
    sim_12 = text_similarity("今天天气真的很好啊对不对", "今天天气真的很好啊对不时")
    assert sim_12 == pytest.approx(11 / 12)
    assert sim_12 >= 0.85

    assert text_similarity("完全不同的一句话", "另一段毫无关系的台词") < 0.5


def test_postprocess_merges_flickering_similar_cues_and_flags_low_conf():
    cfg = ExtractConfig(min_duration_ms=120, merge_gap_ms=160, merge_similarity=0.85, conf_threshold=0.85)
    segs = [
        OcrSegment(100, 180, "废", 0.40, b"png"),  # too short -> dropped
        OcrSegment(400, 1000, "不用亲眼看见即将开始的地狱", 0.94, b"png1"),
        OcrSegment(1080, 1700, "不用亲眼看见即将开始的池狱", 0.72, b"png2"),  # gap=80ms, 1-char OCR typo -> merge
        OcrSegment(2200, 3200, "大家都怎么样了", 0.78, b"png3"),  # low conf -> keep & flag!
    ]
    events = postprocess_segments(segs, cfg)
    assert len(events) == 2
    assert (events[0].index, events[0].start_ms, events[0].end_ms) == (1, 400, 1700)
    assert events[0].text == "不用亲眼看见即将开始的地狱"
    assert events[0].confidence == pytest.approx(0.94)
    assert events[0].thumbnail_png == b"png1"
    assert events[0].flagged is False
    assert (events[1].index, events[1].flagged) == (2, True)


def test_postprocess_empty_and_duration_and_empty_text_filtering():
    cfg = ExtractConfig(min_duration_ms=120)
    assert postprocess_segments([], cfg) == []

    segs = [
        OcrSegment(0, 500, "   \n\t  ", 0.99, b"empty"),
        OcrSegment(500, 619, "太短了", 0.99, b"short"),
        OcrSegment(700, 820, "刚好达标", 0.90, b"exact"),
    ]
    events = postprocess_segments(segs, cfg)
    assert len(events) == 1
    assert events[0].index == 1
    assert events[0].text == "刚好达标"
    assert events[0].flagged is False


def test_postprocess_three_cue_chain_merge_and_tie_breaking():
    cfg = ExtractConfig(min_duration_ms=120, merge_gap_ms=160, merge_similarity=0.85, conf_threshold=0.85)
    segs = [
        # 3-cue flicker chain: 1st (0.75) -> 2nd (0.95, wins text/thumb) -> 3rd (0.80)
        OcrSegment(100, 500, "不用亲眼看见即将开始的池狱", 0.75, b"p1"),
        OcrSegment(580, 900, "不用亲眼看见即将开始的地狱", 0.95, b"p2"),
        OcrSegment(1050, 1400, "不用亲眼看见即将开始的地狱。", 0.80, b"p3"),
        # Tie-breaking: equal confidence -> pick longer normalized text
        OcrSegment(2000, 2400, "不用亲眼看见即将开始的地狱", 0.90, b"t1"),
        OcrSegment(2500, 2900, "不用亲眼看见即将开始的地狱！", 0.90, b"t2"),
        # Tie-breaking: equal confidence & equal length -> keep earlier segment
        OcrSegment(3500, 3900, "不用亲眼看见即将开始的地狱", 0.90, b"e1"),
        OcrSegment(4000, 4400, "不用亲眼看见即将开始的池狱", 0.90, b"e2"),
    ]
    events = postprocess_segments(segs, cfg)
    assert len(events) == 3

    assert (events[0].index, events[0].start_ms, events[0].end_ms) == (1, 100, 1400)
    assert events[0].text == "不用亲眼看见即将开始的地狱"
    assert events[0].thumbnail_png == b"p2"
    assert events[0].confidence == pytest.approx(0.95)
    assert events[0].flagged is False

    assert (events[1].index, events[1].start_ms, events[1].end_ms) == (2, 2000, 2900)
    assert events[1].text == "不用亲眼看见即将开始的地狱！"
    assert events[1].thumbnail_png == b"t2"

    assert (events[2].index, events[2].start_ms, events[2].end_ms) == (3, 3500, 4400)
    assert events[2].text == "不用亲眼看见即将开始的地狱"
    assert events[2].thumbnail_png == b"e1"


def test_postprocess_does_not_merge_long_gaps_or_distinct_cues():
    cfg = ExtractConfig(min_duration_ms=120, merge_gap_ms=160, merge_similarity=0.85, conf_threshold=0.85)
    segs = [
        OcrSegment(100, 500, "完全相同的台词", 0.95, b"a"),
        OcrSegment(661, 1000, "完全相同的台词", 0.95, b"b"),  # gap = 161ms > 160ms
        OcrSegment(1100, 1500, "这是另一句完全不同的对话", 0.95, b"c"),  # gap = 100ms, distinct text
    ]
    events = postprocess_segments(segs, cfg)
    assert len(events) == 3
    assert [e.index for e in events] == [1, 2, 3]
    assert [e.text for e in events] == ["完全相同的台词", "完全相同的台词", "这是另一句完全不同的对话"]


def test_postprocess_flagging_noise_vs_valid_dialogue():
    cfg = ExtractConfig(conf_threshold=0.85)
    segs = [
        # Valid Chinese punctuation + ASCII letters/digits/common punctuation -> NOT flagged
        OcrSegment(0, 500, "「你好，世界！」这是第100集：2026-11-27，100% OK? Yes, it's fine!", 0.85, b"ok1"),
        OcrSegment(1000, 1500, "『真的吗？』——嗯…大概吧～（开玩笑的）【注】《序章》、·", 0.95, b"ok2"),
        # Stray OCR noise symbols even at high confidence -> flagged
        OcrSegment(2000, 2500, "大家|怎么^样", 0.96, b"bad1"),
        OcrSegment(3000, 3500, "测试\ufffd乱码□", 0.99, b"bad2"),
        OcrSegment(4000, 4500, "还行@#$", 0.92, b"bad3"),
    ]
    events = postprocess_segments(segs, cfg)
    assert len(events) == 5
    assert events[0].flagged is False
    assert events[1].flagged is False
    assert events[2].flagged is True
    assert events[3].flagged is True
    assert events[4].flagged is True


def test_postprocess_allows_iteration_marks_and_halfwidth_parens_slash_semicolon():
    assert normalize_cjk_text("人 々 二 〇 二 六") == "人々二〇二六"
    cfg = ExtractConfig(conf_threshold=0.85)
    segs = [
        OcrSegment(0, 500, "人々も二〇二六年 (10/12); OK", 0.95, b"ok"),
    ]
    events = postprocess_segments(segs, cfg)
    assert len(events) == 1
    assert events[0].flagged is False
