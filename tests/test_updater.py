from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.request import Request

import pytest

from muzhen.core import updater
from muzhen.core.updater import (
    GITHUB_REPO,
    RELEASES_API_URL,
    RELEASES_PAGE_URL,
    UpdateInfo,
    UpdatePrefsStore,
    check_for_updates,
    is_newer_version,
    parse_semver,
    select_release_download_url,
)


class _FakeHttpResponse:
    def __init__(self, payload: Any) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> _FakeHttpResponse:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


def test_parse_semver_and_is_newer_version() -> None:
    assert parse_semver("2.0.0") == (2, 0, 0)
    assert parse_semver("v2.1.0") == (2, 1, 0)
    assert parse_semver("V2.10.3") == (2, 10, 3)
    assert parse_semver("v2.1.0-rc.1+build.42") == (2, 1, 0)
    assert parse_semver("2.1.0+20261128") == (2, 1, 0)
    assert parse_semver("v2.1") == (2, 1, 0)

    for invalid in ("", "   ", "v", "not-a-version", "2.x.0", "2..0", "-1.0.0"):
        with pytest.raises(ValueError):
            parse_semver(invalid)

    assert is_newer_version("v2.1.0", "2.0.0") is True
    assert is_newer_version("2.0.1", "v2.0.0") is True
    assert is_newer_version("v2.1.0-beta.1", "2.0.0") is True
    assert is_newer_version("v2.0.0", "2.0.0") is False
    assert is_newer_version("v2.0.0-beta.1", "2.0.0") is False
    assert is_newer_version("v1.9.9", "2.0.0") is False
    assert is_newer_version("invalid", "2.0.0") is False
    assert is_newer_version("v2.1.0", "invalid") is False


def test_check_for_updates_parses_release_selects_portable_zip_and_ignores_drafts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert GITHUB_REPO == "wanngxuanji/Muzhen"
    assert RELEASES_API_URL == "https://api.github.com/repos/wanngxuanji/Muzhen/releases/latest"
    assert RELEASES_PAGE_URL == "https://github.com/wanngxuanji/Muzhen/releases"

    captured: dict[str, Any] = {}
    current_payload: dict[str, Any] = {}

    def _fake_urlopen(request: Request, timeout: float = 0.0) -> _FakeHttpResponse:
        captured["url"] = request.full_url
        captured["headers"] = {k.lower(): v for k, v in request.header_items()}
        captured["timeout"] = timeout
        return _FakeHttpResponse(current_payload)

    monkeypatch.setattr(updater.urllib.request, "urlopen", _fake_urlopen)

    portable_url = "https://github.com/wanngxuanji/Muzhen/releases/download/v2.1.0/Muzhen-v2.1.0-win64-portable.zip"
    generic_zip_url = "https://github.com/wanngxuanji/Muzhen/releases/download/v2.1.0/source-bundle.zip"
    release_html_url = "https://github.com/wanngxuanji/Muzhen/releases/tag/v2.1.0"

    current_payload = {
        "tag_name": "v2.1.0",
        "draft": False,
        "prerelease": False,
        "html_url": release_html_url,
        "body": "  - Faster OCR batching\n  - Bug fixes  ",
        "published_at": "2026-12-01T10:00:00Z",
        "assets": [
            {"name": "source-bundle.zip", "browser_download_url": generic_zip_url},
            {"name": "SHA256SUMS.txt", "browser_download_url": "https://example.invalid/SHA256SUMS.txt"},
            {"name": "Muzhen-v2.1.0-win64-portable.zip", "browser_download_url": portable_url},
        ],
    }

    info = check_for_updates("2.0.0", timeout=3.5)
    assert info == UpdateInfo(
        current_version="2.0.0",
        latest_version="2.1.0",
        tag_name="v2.1.0",
        release_url=release_html_url,
        download_url=portable_url,
        changelog="- Faster OCR batching\n  - Bug fixes",
        published_at="2026-12-01T10:00:00Z",
    )
    assert captured["url"] == RELEASES_API_URL
    assert captured["timeout"] == 3.5
    assert captured["headers"]["accept"] == "application/vnd.github+json"
    assert captured["headers"]["user-agent"] == "Muzhen/2.0.0"

    current_payload = {
        "tag_name": "v2.1.0",
        "draft": False,
        "prerelease": False,
        "html_url": release_html_url,
        "body": "",
        "published_at": "2026-12-01T10:00:00Z",
        "assets": [
            {"name": "Muzhen-v2.1.0-win64-portable.zip", "browser_download_url": ""},
            {"name": "custom-package.ZIP", "browser_download_url": generic_zip_url},
        ],
    }
    fallback_zip_info = check_for_updates("2.0.0")
    assert fallback_zip_info is not None
    assert fallback_zip_info.download_url == generic_zip_url

    current_payload = {
        "tag_name": "v2.1.0",
        "draft": False,
        "prerelease": False,
        "html_url": release_html_url,
        "body": None,
        "published_at": None,
        "assets": [
            {"name": "SHA256SUMS.txt", "browser_download_url": "https://example.invalid/SHA256SUMS.txt"},
        ],
    }
    fallback_html_info = check_for_updates("2.0.0")
    assert fallback_html_info is not None
    assert fallback_html_info.release_url == release_html_url
    assert fallback_html_info.download_url == release_html_url
    assert fallback_html_info.changelog == ""
    assert fallback_html_info.published_at == ""

    assert select_release_download_url([], RELEASES_PAGE_URL) == RELEASES_PAGE_URL

    current_payload = {
        "tag_name": "v2.0.0",
        "draft": False,
        "prerelease": False,
        "html_url": release_html_url,
        "assets": [],
    }
    assert check_for_updates("2.0.0") is None

    current_payload = {
        "tag_name": "v2.2.0",
        "draft": True,
        "prerelease": False,
        "html_url": release_html_url,
        "assets": [],
    }
    assert check_for_updates("2.0.0") is None

    current_payload = {
        "tag_name": "v2.2.0",
        "draft": False,
        "prerelease": True,
        "html_url": release_html_url,
        "assets": [],
    }
    assert check_for_updates("2.0.0") is None

    current_payload = {
        "draft": False,
        "prerelease": False,
        "html_url": release_html_url,
        "assets": [],
    }
    assert check_for_updates("2.0.0") is None


def test_update_prefs_store_persists_skipped_version_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefs_path = tmp_path / "nested" / "update_prefs.json"
    store = UpdatePrefsStore(prefs_path)

    assert store.get_skipped_version() is None
    assert store.is_version_skipped("v2.1.0") is False

    store.set_skipped_version("v2.1.0")
    assert prefs_path.exists()
    leftover_tmps = list(prefs_path.parent.glob("*.tmp"))
    assert leftover_tmps == []

    reloaded = UpdatePrefsStore(prefs_path)
    assert reloaded.get_skipped_version() == "v2.1.0"
    assert reloaded.is_version_skipped("v2.1.0") is True
    assert reloaded.is_version_skipped("2.1.0") is True
    assert reloaded.is_version_skipped("V2.1.0") is True
    assert reloaded.is_version_skipped("v2.2.0") is False

    reloaded.set_skipped_version(None)
    assert reloaded.get_skipped_version() is None
    assert reloaded.is_version_skipped("v2.1.0") is False

    prefs_path.write_text("{corrupt json", encoding="utf-8")
    assert reloaded.get_skipped_version() is None
    assert reloaded.is_version_skipped("v2.1.0") is False

    prefs_path.write_text('["not", "an", "object"]', encoding="utf-8")
    assert reloaded.get_skipped_version() is None

    monkeypatch.setenv("MUZHEN_DATA_DIR", str(tmp_path / "appdata"))
    default_store = UpdatePrefsStore()
    assert default_store.path == tmp_path / "appdata" / "update_prefs.json"
    assert not (tmp_path / "appdata").exists()


def test_update_prefs_store_persists_language_and_preserves_skipped_version(
    tmp_path: Path,
) -> None:
    prefs_path = tmp_path / "nested" / "update_prefs.json"
    store = UpdatePrefsStore(prefs_path)

    assert store.get_language() == "auto"

    store.set_skipped_version("v2.1.0")
    assert store.get_language() == "auto"

    store.set_language("vi")
    assert list(prefs_path.parent.glob("*.tmp")) == []

    reloaded = UpdatePrefsStore(prefs_path)
    assert reloaded.get_language() == "vi"
    assert reloaded.get_skipped_version() == "v2.1.0"
    assert reloaded.is_version_skipped("v2.1.0") is True

    reloaded.set_skipped_version("v2.2.0")
    assert reloaded.get_skipped_version() == "v2.2.0"
    assert reloaded.get_language() == "vi"

    reloaded.set_skipped_version(None)
    assert reloaded.get_skipped_version() is None
    assert reloaded.get_language() == "vi"

    reloaded.set_language("  ZH  ")
    assert reloaded.get_language() == "zh"

    reloaded.set_language("en")
    assert reloaded.get_language() == "en"

    reloaded.set_language("fr")
    assert reloaded.get_language() == "auto"

    prefs_path.write_text('{"language": "invalid", "skipped_version": "v2.3.0"}', encoding="utf-8")
    assert reloaded.get_language() == "auto"
    assert reloaded.get_skipped_version() == "v2.3.0"

    prefs_path.write_text("{corrupt json", encoding="utf-8")
    assert reloaded.get_language() == "auto"

    prefs_path.write_text('["not", "an", "object"]', encoding="utf-8")
    assert reloaded.get_language() == "auto"


def test_update_prefs_store_persists_discord_rpc_flags_and_preserves_other_keys(
    tmp_path: Path,
) -> None:
    prefs_path = tmp_path / "prefs" / "update_prefs.json"
    store = UpdatePrefsStore(prefs_path)

    assert store.get_discord_rpc_enabled() is True
    assert store.get_discord_rpc_hide_filename() is True
    assert store.get_discord_rpc_show_github_button() is True

    store.set_language("vi")
    store.set_skipped_version("v2.1.0")
    store.set_discord_rpc_enabled(False)
    store.set_discord_rpc_hide_filename(False)
    store.set_discord_rpc_show_github_button(False)

    reloaded = UpdatePrefsStore(prefs_path)
    assert reloaded.get_language() == "vi"
    assert reloaded.get_skipped_version() == "v2.1.0"
    assert reloaded.get_discord_rpc_enabled() is False
    assert reloaded.get_discord_rpc_hide_filename() is False
    assert reloaded.get_discord_rpc_show_github_button() is False

    prefs_path.write_text(
        '{"discord_rpc_enabled": "yes", "discord_rpc_hide_filename": 0}',
        encoding="utf-8",
    )
    assert reloaded.get_discord_rpc_enabled() is True
    assert reloaded.get_discord_rpc_hide_filename() is True
    assert reloaded.get_discord_rpc_show_github_button() is True
