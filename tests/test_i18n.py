from __future__ import annotations

import string
from types import SimpleNamespace

import pytest

from muzhen.ui import i18n
from muzhen.ui.i18n import (
    LANGUAGE_CHOICES,
    SUPPORTED_LANGUAGES,
    TRANSLATIONS,
    detect_system_language,
    resolve_language,
    tr,
)

REQUIRED_KEYS: tuple[str, ...] = (
    "panel_settings",
    "panel_monitor",
    "panel_subtitles",
    "render_method",
    "render_gpu",
    "render_cpu",
    "gpu_unavailable_tip",
    "model_label",
    "model_precision",
    "model_turbo",
    "flag_threshold",
    "roi_preset",
    "preset_none",
    "preset_save_tip",
    "preset_delete_tip",
    "export_format",
    "output_folder",
    "output_same",
    "output_custom",
    "output_custom_placeholder",
    "output_browse_tip",
    "language_label",
    "lang_auto",
    "btn_extract",
    "btn_stop",
    "status_ready",
    "status_cleaning_worker",
    "status_extracting",
    "status_scanning",
    "status_ocr_progress",
    "status_done",
    "status_cancelled",
    "status_testing_frame",
    "status_export_done",
    "status_checking_updates",
    "status_up_to_date",
    "status_update_failed",
    "monitor_empty_prompt",
    "open_video_tip",
    "roi_pill",
    "btn_settings",
    "btn_test_frame",
    "test_frame_tip",
    "trim_in",
    "trim_out",
    "trim_clear",
    "trim_full",
    "tip_seek_start",
    "tip_play_pause",
    "tip_step_back",
    "tip_step_fwd",
    "tip_seek_end",
    "tip_trim_in",
    "tip_trim_out",
    "tip_trim_clear",
    "btn_export",
    "btn_export_formats",
    "open_out_dir_tip",
    "btn_check_updates",
    "btn_update_available",
    "author_credit",
    "btn_star_github",
    "star_github_tip",
    "update_dialog_title",
    "update_dialog_header",
    "update_dialog_published",
    "update_dialog_fallback_notes",
    "btn_update_now",
    "btn_skip_version",
    "btn_later",
    "msg_up_to_date_title",
    "msg_up_to_date_body",
    "msg_update_failed_body",
    "filter_all",
    "filter_flagged",
    "col_index",
    "col_start",
    "col_end",
    "col_conf",
    "col_text",
    "btn_merge",
    "btn_split",
    "btn_mark_ok",
    "tip_find",
    "tip_undo",
    "tip_redo",
    "tip_merge",
    "tip_split",
    "tip_delete",
    "find_placeholder",
    "replace_placeholder",
    "btn_replace",
    "btn_replace_all",
    "inspector_title",
    "inspector_empty",
    "inspector_no_match",
    "inspector_meta",
    "badge_too_short",
    "badge_overlap",
    "flag_needs_review",
    "btn_jump_start",
    "btn_jump_end",
    "btn_snap_in",
    "btn_snap_out",
    "tip_jump_start",
    "tip_jump_end",
    "tip_snap_in",
    "tip_snap_out",
    "tip_nudge_back",
    "tip_nudge_fwd",
    "tip_mark_ok",
    "dlg_open_video_title",
    "dlg_open_video_filter",
    "dlg_select_out_dir",
    "dlg_save_preset_title",
    "dlg_save_preset_prompt",
    "dlg_delete_preset_title",
    "dlg_delete_preset_body",
    "dlg_unsaved_title",
    "dlg_unsaved_body",
    "dlg_overwrite_title",
    "dlg_overwrite_body",
    "dlg_error_title",
    "dlg_no_format_body",
    "dlg_no_cues_body",
    "dlg_preset_error_title",
)


def _extract_placeholders(template: str) -> set[tuple[str, str]]:
    formatter = string.Formatter()
    result: set[tuple[str, str]] = set()
    for _, field_name, format_spec, _ in formatter.parse(template):
        if field_name is not None:
            result.add((field_name, format_spec or ""))
    return result


def test_language_constants() -> None:
    assert SUPPORTED_LANGUAGES == ("en", "vi", "zh")
    assert LANGUAGE_CHOICES == (
        ("auto", "Auto"),
        ("en", "English"),
        ("vi", "Tiếng Việt"),
        ("zh", "中文 (简体)"),
    )


@pytest.mark.parametrize(
    ("locale_name", "expected"),
    [
        ("vi", "vi"),
        ("vi_VN", "vi"),
        ("vi-VN", "vi"),
        ("VI_vn", "vi"),
        ("zh", "zh"),
        ("zh_CN", "zh"),
        ("zh-Hans", "zh"),
        ("zh-TW", "zh"),
        ("ZH_HK", "zh"),
        ("en", "en"),
        ("en_US", "en"),
        ("en-GB", "en"),
        ("ja_JP", "en"),
        ("fr_FR", "en"),
        ("", "en"),
    ],
)
def test_detect_system_language_explicit_locale(locale_name: str, expected: str) -> None:
    assert detect_system_language(locale_name) == expected


def test_detect_system_language_uses_qlocale_and_locale_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        i18n.QLocale,
        "system",
        staticmethod(lambda: SimpleNamespace(name=lambda: "vi_VN")),
    )
    assert detect_system_language() == "vi"

    monkeypatch.setattr(
        i18n.QLocale,
        "system",
        staticmethod(lambda: SimpleNamespace(name=lambda: "zh_CN")),
    )
    assert detect_system_language() == "zh"

    monkeypatch.setattr(
        i18n.QLocale,
        "system",
        staticmethod(lambda: SimpleNamespace(name=lambda: "")),
    )
    monkeypatch.setattr(i18n.locale, "getlocale", lambda: ("zh_TW", "UTF-8"))
    assert detect_system_language() == "zh"

    monkeypatch.setattr(i18n.locale, "getlocale", lambda: (None, None))
    assert detect_system_language() == "en"


def test_resolve_language_preference_and_auto_fallback() -> None:
    assert resolve_language("en", "vi_VN") == "en"
    assert resolve_language("vi", "zh_CN") == "vi"
    assert resolve_language("zh", "en_US") == "zh"
    assert resolve_language(" VI ", "en_US") == "vi"
    assert resolve_language("ZH", "en_US") == "zh"

    assert resolve_language("auto", "vi_VN") == "vi"
    assert resolve_language("AUTO", "zh-Hans") == "zh"
    assert resolve_language(None, "vi-VN") == "vi"
    assert resolve_language("", "zh_CN") == "zh"
    assert resolve_language("fr", "vi_VN") == "vi"
    assert resolve_language("fr", "ja_JP") == "en"


def test_translations_have_complete_key_and_placeholder_parity() -> None:
    assert set(TRANSLATIONS.keys()) == set(SUPPORTED_LANGUAGES)

    en_keys = set(TRANSLATIONS["en"].keys())
    vi_keys = set(TRANSLATIONS["vi"].keys())
    zh_keys = set(TRANSLATIONS["zh"].keys())

    missing_required = set(REQUIRED_KEYS) - en_keys
    assert missing_required == set()
    assert en_keys == vi_keys == zh_keys

    for lang in SUPPORTED_LANGUAGES:
        for key, value in TRANSLATIONS[lang].items():
            assert isinstance(value, str), f"{lang}.{key} is not str"
            assert value.strip() != "", f"{lang}.{key} is empty"

    for key in en_keys:
        en_ph = _extract_placeholders(TRANSLATIONS["en"][key])
        vi_ph = _extract_placeholders(TRANSLATIONS["vi"][key])
        zh_ph = _extract_placeholders(TRANSLATIONS["zh"][key])
        assert en_ph == vi_ph == zh_ph, f"Placeholder mismatch for key {key!r}"


def test_tr_lookup_formatting_and_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    assert tr("en", "panel_settings") == "Settings"
    assert tr("vi", "panel_settings") != ""
    assert tr("zh", "panel_settings") != ""
    assert tr("vi", "panel_settings") != tr("en", "panel_settings")
    assert tr("zh", "panel_settings") != tr("en", "panel_settings")

    for lang in SUPPORTED_LANGUAGES:
        done_str = tr(lang, "status_done", total=12, flagged=3)
        assert "12" in done_str and "3" in done_str

        meta_str = tr(lang, "inspector_meta", seconds=1.64, chars=18)
        assert "1.64" in meta_str and "18" in meta_str

        filter_str = tr(lang, "filter_all", count=7)
        assert "7" in filter_str

        overwrite_str = tr(lang, "dlg_overwrite_body", names="- ep01.ass")
        assert "- ep01.ass" in overwrite_str

    assert tr("fr", "panel_settings") == tr("en", "panel_settings")
    assert tr("en", "nonexistent_i18n_key") == "nonexistent_i18n_key"

    vi_copy = dict(TRANSLATIONS["vi"])
    vi_copy.pop("panel_settings")
    monkeypatch.setitem(TRANSLATIONS, "vi", vi_copy)
    assert tr("vi", "panel_settings") == tr("en", "panel_settings")
