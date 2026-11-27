from __future__ import annotations

import pytest

from muzhen.core import (
    normalize_cjk_text as core_normalize_cjk_text,
    postprocess_segments as core_postprocess_segments,
    text_similarity as core_text_similarity,
)
from muzhen.core.models import ExtractConfig, OcrSegment
from muzhen.core.postprocess import (
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

    sim_12 = text_similarity("今天天气真的很好啊对不对", "今天天气真的很好啊对不时")
    assert sim_12 == pytest.approx(11 / 12)
    assert sim_12 >= 0.85

    assert text_similarity("完全不同的一句话", "另一段毫无关系的台词") < 0.5


def test_postprocess_merges_flickering_similar_cues_and_flags_low_conf():
    cfg = ExtractConfig(min_duration_ms=120, merge_gap_ms=160, merge_similarity=0.85, conf_threshold=0.85)
    segs = [
        OcrSegment(100, 180, "废", 0.40, b"png"),
        OcrSegment(400, 1000, "不用亲眼看见即将开始的地狱", 0.94, b"png1"),
        OcrSegment(1080, 1700, "不用亲眼看见即将开始的池狱", 0.72, b"png2"),
        OcrSegment(2200, 3200, "大家都怎么样了", 0.78, b"png3"),
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
        OcrSegment(100, 500, "不用亲眼看见即将开始的池狱", 0.75, b"p1"),
        OcrSegment(580, 900, "不用亲眼看见即将开始的地狱", 0.95, b"p2"),
        OcrSegment(1050, 1400, "不用亲眼看见即将开始的地狱。", 0.80, b"p3"),
        OcrSegment(2000, 2400, "不用亲眼看见即将开始的地狱", 0.90, b"t1"),
        OcrSegment(2500, 2900, "不用亲眼看见即将开始的地狱！", 0.90, b"t2"),
        OcrSegment(3500, 3900, "不用亲眼看见即将开始的地狱", 0.80, b"e1"),
        OcrSegment(4000, 4400, "不用亲眼看见即将开始的池狱", 0.80, b"e2"),
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
        OcrSegment(661, 1000, "完全相同的台词", 0.95, b"b"),
        OcrSegment(1100, 1500, "这是另一句完全不同的对话", 0.95, b"c"),
    ]
    events = postprocess_segments(segs, cfg)
    assert len(events) == 3
    assert [e.index for e in events] == [1, 2, 3]
    assert [e.text for e in events] == ["完全相同的台词", "完全相同的台词", "这是另一句完全不同的对话"]


def test_postprocess_flagging_noise_vs_valid_dialogue():
    cfg = ExtractConfig(conf_threshold=0.85)
    segs = [
        OcrSegment(0, 500, "「你好，世界！」这是第100集：2026-11-27，100% OK? Yes, it's fine!", 0.85, b"ok1"),
        OcrSegment(1000, 1500, "『真的吗？』——嗯…大概吧～（开玩笑的）【注】《序章》、·", 0.95, b"ok2"),
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


def test_postprocess_drops_low_conf_non_cjk_garbage_keeps_low_conf_cjk_and_high_conf_ascii():
    segs = [
        OcrSegment(0, 280, "VNY", 0.654, b"collar"),
        OcrSegment(400, 900, "II / -", 0.78, b"noise"),
        OcrSegment(900, 1100, "マ", 0.99, b"kana_ghost"),
        OcrSegment(1120, 1280, "バハ", 0.53, b"kana_low"),
        OcrSegment(1290, 1450, "》", 0.92, b"punct_only"),
        OcrSegment(1500, 1800, "大家都怎么样了", 0.72, b"cjk"),
        OcrSegment(2000, 2800, "Episode 01", 0.96, b"title"),
    ]

    filtered = postprocess_segments(segs, ExtractConfig(conf_threshold=0.85, filter_non_cjk_low_conf=True))
    assert [(e.index, e.text, e.flagged) for e in filtered] == [
        (1, "大家都怎么样了", True),
        (2, "Episode 01", False),
    ]

    unfiltered = postprocess_segments(segs, ExtractConfig(conf_threshold=0.85, filter_non_cjk_low_conf=False))
    assert [e.text for e in unfiltered] == ["VNY", "II / -", "マ", "バハ", "》", "大家都怎么样了", "Episode 01"]
    assert unfiltered[0].flagged is True


def test_postprocess_two_tier_smart_merge_containment_and_guards():
    cfg = ExtractConfig(
        min_duration_ms=120,
        merge_gap_ms=160,
        flash_bridge_max_ms=320,
        merge_similarity=0.85,
        conf_threshold=0.85,
    )
    segs = [
        OcrSegment(2000, 2640, "个蕴藏着神秘力量的", 0.99, b"clipped"),
        OcrSegment(2920, 3400, "一个蕴藏着神秘力量的", 0.97, b"full"),
        OcrSegment(5000, 5600, "我不会放过你的", 0.96, b"you"),
        OcrSegment(5820, 6400, "我不会放过他的", 0.96, b"him"),
        OcrSegment(8000, 8400, "快跑！", 0.98, b"run1"),
        OcrSegment(8640, 9000, "快跑！", 0.98, b"run2"),
        OcrSegment(10000, 10080, "战斗开始", 0.91, b"frag"),
        OcrSegment(10120, 10920, "战斗开始", 0.99, b"main"),
    ]
    events = postprocess_segments(segs, cfg)
    assert [(e.start_ms, e.end_ms, e.text) for e in events] == [
        (2000, 3400, "一个蕴藏着神秘力量的"),
        (5000, 5600, "我不会放过你的"),
        (5820, 6400, "我不会放过他的"),
        (8000, 8400, "快跑！"),
        (8640, 9000, "快跑！"),
        (10000, 10920, "战斗开始"),
    ]
    assert events[0].thumbnail_png == b"full"


def test_postprocess_similarity_ignores_punctuation_and_extended_whitelist_with_flag_reason():
    assert text_similarity("你好世界", "你好，世界！") == 1.0
    assert normalize_cjk_text("等一下―真的吗") == "等一下—真的吗"

    cfg = ExtractConfig(conf_threshold=0.85)
    segs = [
        OcrSegment(0, 500, "发行许可证：（粤）动审字[2013]第054号 ~ ♪", 0.98, b"ok"),
        OcrSegment(1000, 1500, "停下来了", 0.72, b"low"),
        OcrSegment(2000, 2500, "大家|怎么^样", 0.95, b"noise"),
    ]
    events = postprocess_segments(segs, cfg)
    assert events[0].flagged is False
    assert events[0].flag_reason == ""
    assert events[1].flagged is True
    assert "0.72" in events[1].flag_reason
    assert events[2].flagged is True
    assert "|" in events[2].flag_reason


def test_postprocess_does_not_merge_0ms_one_char_substitution_and_clamps_duration():
    cfg = ExtractConfig(
        min_duration_ms=120,
        merge_gap_ms=160,
        flash_bridge_max_ms=320,
        merge_similarity=0.85,
        conf_threshold=0.85,
    )
    segs = [
        OcrSegment(0, 1000, "我不会放过你的", 0.96, b"you"),
        OcrSegment(1000, 2000, "我不会放过他的", 0.96, b"him"),
        OcrSegment(2100, 2600, "第一句重叠台词", 0.95, b"ov1"),
        OcrSegment(2450, 3200, "第二句完全不同", 0.95, b"ov2"),
        OcrSegment(3100, 3600, "超出片尾", 0.95, b"past"),
    ]
    events = postprocess_segments(segs, cfg, duration_ms=3000)
    assert [(e.index, e.start_ms, e.end_ms, e.text) for e in events] == [
        (1, 0, 1000, "我不会放过你的"),
        (2, 1000, 2000, "我不会放过他的"),
        (3, 2100, 2450, "第一句重叠台词"),
        (4, 2450, 3000, "第二句完全不同"),
    ]
    assert all(ev.end_ms <= 3000 for ev in events)
    assert all(events[i].end_ms <= events[i + 1].start_ms for i in range(len(events) - 1))

    edge_and_dup = [
        OcrSegment(100, 500, "G我们是救世主", 0.90, b"g1"),
        OcrSegment(550, 950, "Y我们是救世主", 0.92, b"g2"),
        OcrSegment(1200, 1800, "重复重叠台词", 0.91, b"d1"),
        OcrSegment(1700, 2200, "重复重叠台词", 0.97, b"d2"),
    ]
    merged_edge = postprocess_segments(edge_and_dup, cfg)
    assert [(e.start_ms, e.end_ms, e.text, e.confidence) for e in merged_edge] == [
        (100, 950, "Y我们是救世主", pytest.approx(0.92)),
        (1200, 2200, "重复重叠台词", pytest.approx(0.97)),
    ]


def test_postprocess_drops_short_low_conf_hanzi_and_symbol_ghosts_keeps_real_dialogue():
    cfg = ExtractConfig(min_duration_ms=120, conf_threshold=0.85, filter_non_cjk_low_conf=True)
    segs = [
        OcrSegment(0, 240, "生", 0.52, b"ghost1"),
        OcrSegment(400, 640, "#一人", 0.56, b"ghost2"),
        OcrSegment(800, 1040, "八#", 0.70, b"ghost3"),
        OcrSegment(1200, 1440, "|了", 0.65, b"ghost4"),
        OcrSegment(1600, 1840, "嗯", 0.95, b"real1"),
        OcrSegment(2000, 2600, "生", 0.60, b"real_long"),
        OcrSegment(2800, 3400, "大家都怎么样了", 0.72, b"low_cjk"),
        OcrSegment(3600, 4200, "大家|怎么^样", 0.75, b"long_sym"),
    ]
    events = postprocess_segments(segs, cfg)
    assert [(e.index, e.text, e.flagged) for e in events] == [
        (1, "嗯", False),
        (2, "生", True),
        (3, "大家都怎么样了", True),
        (4, "大家|怎么^样", True),
    ]


def test_postprocess_preserves_fuller_winner_confidence_without_masking_flag():
    cfg = ExtractConfig(
        min_duration_ms=120,
        merge_gap_ms=160,
        flash_bridge_max_ms=320,
        merge_similarity=0.85,
        conf_threshold=0.85,
    )
    segs = [
        OcrSegment(1000, 1500, "个蕴藏着神秘力量的", 0.88, b"clipped"),
        OcrSegment(1600, 2200, "一个蕴藏着神秘力量的", 0.81, b"full"),
    ]
    events = postprocess_segments(segs, cfg)
    assert len(events) == 1
    assert events[0].text == "一个蕴藏着神秘力量的"
    assert events[0].thumbnail_png == b"full"
    assert events[0].confidence == pytest.approx(0.81)
    assert events[0].flagged is True
    assert "0.81" in events[0].flag_reason


def test_postprocess_non_overlap_runs_after_ghost_filter_and_guards_zero_duration_cues():
    cfg = ExtractConfig(min_duration_ms=120, conf_threshold=0.85, filter_non_cjk_low_conf=True)
    ghost_overlap = [
        OcrSegment(1000, 2000, "这是一句真实台词", 0.98, b"real"),
        OcrSegment(1050, 1250, "生", 0.50, b"ghost"),
    ]
    events = postprocess_segments(ghost_overlap, cfg)
    assert [(e.start_ms, e.end_ms, e.text) for e in events] == [
        (1000, 2000, "这是一句真实台词"),
    ]

    cfg_zero = ExtractConfig(min_duration_ms=0, conf_threshold=0.85, filter_non_cjk_low_conf=True)
    same_start = [
        OcrSegment(1000, 1500, "第一句", 0.95, b"a"),
        OcrSegment(1000, 2000, "第二句", 0.95, b"b"),
    ]
    same_start_events = postprocess_segments(same_start, cfg_zero)
    assert [(e.start_ms, e.end_ms, e.text) for e in same_start_events] == [
        (1000, 2000, "第二句"),
    ]
    assert all(e.end_ms > e.start_ms for e in same_start_events)
    assert all(
        same_start_events[i].end_ms <= same_start_events[i + 1].start_ms
        for i in range(len(same_start_events) - 1)
    )

    zero_dur_segs = [
        OcrSegment(1000, 1000, "好", 0.95, b"zero"),
        OcrSegment(1200, 1800, "你好世界", 0.95, b"valid"),
    ]
    zero_dur_events = postprocess_segments(zero_dur_segs, cfg_zero, duration_ms=0)
    assert [(e.index, e.start_ms, e.end_ms, e.text) for e in zero_dur_events] == [
        (1, 1200, 1800, "你好世界"),
    ]


def test_postprocess_remerges_identical_cues_after_intervening_40ms_ghost_removed():
    cfg = ExtractConfig(
        min_duration_ms=120,
        merge_gap_ms=160,
        flash_bridge_max_ms=320,
        conf_threshold=0.85,
        filter_non_cjk_low_conf=True,
    )
    segs = [
        OcrSegment(1000, 1500, "你好吗", 0.98, b"p1"),
        OcrSegment(1500, 1540, "|", 0.60, b"ghost"),
        OcrSegment(1540, 2500, "你好吗", 0.97, b"p2"),
    ]
    events = postprocess_segments(segs, cfg)
    assert len(events) == 1
    assert (events[0].index, events[0].start_ms, events[0].end_ms, events[0].text) == (
        1,
        1000,
        2500,
        "你好吗",
    )


def test_postprocess_preserves_numeric_countdown_minimal_pairs():
    cfg = ExtractConfig(
        min_duration_ms=120,
        merge_gap_ms=160,
        flash_bridge_max_ms=320,
        merge_similarity=0.85,
        conf_threshold=0.85,
    )
    segs = [
        OcrSegment(0, 1000, "距离爆炸还有3秒", 0.98, b"t3"),
        OcrSegment(1000, 2000, "距离爆炸还有2秒", 0.98, b"t2"),
    ]
    events = postprocess_segments(segs, cfg)
    assert [(e.index, e.start_ms, e.end_ms, e.text) for e in events] == [
        (1, 0, 1000, "距离爆炸还有3秒"),
        (2, 1000, 2000, "距离爆炸还有2秒"),
    ]


def test_pick_merged_winner_uses_stripped_length_for_containment():
    cfg = ExtractConfig(
        min_duration_ms=120,
        merge_gap_ms=160,
        flash_bridge_max_ms=320,
        merge_similarity=0.85,
        conf_threshold=0.85,
    )
    segs = [
        OcrSegment(1000, 1500, "个神秘力量！！！", 0.96, b"punct_trunc"),
        OcrSegment(1580, 2200, "一个神秘力量", 0.95, b"full"),
    ]
    events = postprocess_segments(segs, cfg)
    assert len(events) == 1
    assert events[0].text == "一个神秘力量"
    assert events[0].thumbnail_png == b"full"
    assert events[0].confidence == pytest.approx(0.95)

    segs_rev = [
        OcrSegment(1000, 1500, "一个神秘力量", 0.95, b"full"),
        OcrSegment(1580, 2200, "个神秘力量！！！", 0.94, b"punct_trunc"),
    ]
    events_rev = postprocess_segments(segs_rev, cfg)
    assert len(events_rev) == 1
    assert events_rev[0].text == "一个神秘力量"
    assert events_rev[0].thumbnail_png == b"full"
    assert events_rev[0].confidence == pytest.approx(0.95)

