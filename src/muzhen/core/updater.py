"""GitHub Releases auto-update checker and local preference storage."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import urllib.request

from muzhen.core import paths as runtime_paths

GITHUB_REPO = "wanngxuanji/Muzhen"
RELEASES_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
RELEASES_PAGE_URL = f"https://github.com/{GITHUB_REPO}/releases"

_SEMVER_CORE_RE = re.compile(r"^\d+(?:\.\d+){0,2}$")


@dataclass(frozen=True, slots=True)
class UpdateInfo:
    current_version: str
    latest_version: str
    tag_name: str
    release_url: str
    download_url: str
    changelog: str
    published_at: str


def _normalize_version_tag(version: str) -> str:
    cleaned = version.strip()
    if cleaned[:1] in ("v", "V"):
        return cleaned[1:]
    return cleaned


def parse_semver(version: str) -> tuple[int, int, int]:
    if not isinstance(version, str):
        raise ValueError("Version must be a string")
    normalized = _normalize_version_tag(version)
    core = re.split(r"[-+]", normalized, maxsplit=1)[0]
    if not _SEMVER_CORE_RE.fullmatch(core):
        raise ValueError(f"Invalid semantic version: {version!r}")
    parts = [int(part) for part in core.split(".")]
    while len(parts) < 3:
        parts.append(0)
    return (parts[0], parts[1], parts[2])


def is_newer_version(latest: str, current: str) -> bool:
    try:
        return parse_semver(latest) > parse_semver(current)
    except ValueError:
        return False


def select_release_download_url(assets: list[dict[str, Any]], fallback_url: str) -> str:
    any_zip_url: str | None = None
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        raw_name = asset.get("name")
        raw_url = asset.get("browser_download_url")
        if not isinstance(raw_name, str) or not isinstance(raw_url, str):
            continue
        url = raw_url.strip()
        if not url:
            continue
        name_lower = raw_name.strip().lower()
        if name_lower.startswith("muzhen") and name_lower.endswith("-win64-portable.zip"):
            return url
        if any_zip_url is None and name_lower.endswith(".zip"):
            any_zip_url = url
    return any_zip_url or fallback_url


def check_for_updates(
    current_version: str,
    *,
    timeout: float = 4.5,
    api_url: str = RELEASES_API_URL,
) -> UpdateInfo | None:
    request = urllib.request.Request(
        api_url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"Muzhen/{current_version}",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if not isinstance(payload, dict):
        return None
    if payload.get("draft") or payload.get("prerelease"):
        return None

    raw_tag = payload.get("tag_name")
    if not isinstance(raw_tag, str) or not raw_tag.strip():
        return None
    tag_name = raw_tag.strip()

    if not is_newer_version(tag_name, current_version):
        return None

    raw_html_url = payload.get("html_url")
    release_url = (
        raw_html_url.strip()
        if isinstance(raw_html_url, str) and raw_html_url.strip()
        else RELEASES_PAGE_URL
    )
    raw_assets = payload.get("assets")
    assets = raw_assets if isinstance(raw_assets, list) else []
    download_url = select_release_download_url(assets, release_url)
    changelog = str(payload.get("body") or "").strip()
    published_at = str(payload.get("published_at") or "").strip()

    return UpdateInfo(
        current_version=_normalize_version_tag(current_version),
        latest_version=_normalize_version_tag(tag_name),
        tag_name=tag_name,
        release_url=release_url,
        download_url=download_url,
        changelog=changelog,
        published_at=published_at,
    )


class UpdatePrefsStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = (
            Path(path)
            if path is not None
            else runtime_paths.get_app_data_dir(create=False) / "update_prefs.json"
        )

    def get_skipped_version(self) -> str | None:
        if not self.path.exists():
            return None
        try:
            raw = self.path.read_text(encoding="utf-8")
            if not raw.strip():
                return None
            data = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        skipped = data.get("skipped_version")
        if isinstance(skipped, str) and skipped.strip():
            return skipped.strip()
        return None

    def set_skipped_version(self, tag_name: str | None) -> None:
        normalized = (
            tag_name.strip()
            if isinstance(tag_name, str) and tag_name.strip()
            else None
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"skipped_version": normalized}, ensure_ascii=False, indent=2)
        fd, tmp_name = tempfile.mkstemp(
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, self.path)
        except BaseException:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def is_version_skipped(self, tag_name: str) -> bool:
        if not isinstance(tag_name, str) or not tag_name.strip():
            return False
        skipped = self.get_skipped_version()
        if not skipped:
            return False
        return _normalize_version_tag(skipped).lower() == _normalize_version_tag(tag_name).lower()
