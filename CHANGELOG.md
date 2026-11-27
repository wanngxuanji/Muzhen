# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] - 2026-11-28

### Added
- **Discord Rich Presence (`Discord RPC`)**: Zero-dependency pure-Python local IPC client over Windows Named Pipe / Unix socket displaying live workbench activity (`Idle`, `Previewing video`, `Extracting subtitles` with progress & elapsed timer, and `Editing subtitles` with cue/flagged counts), 15-second progress throttling, and silent auto-reconnect.
- **Discord RPC Settings & Privacy Controls**: Segmented **`On` / `Off`** toggle (enabled by default) inside **Settings**, plus conditional **`Hide file name`** (enabled by default) and **`Show GitHub button`** checkboxes persisted in `update_prefs.json` and localized across **EN / VI / ZH**.
- **Module Entry Point (`python -m muzhen`)**: Added `src/muzhen/__main__.py` for direct module execution.

### Changed
- Updated the Settings footer GitHub Star button label to **`★ Muzhen`** (with `wanngxuanji/Muzhen` tooltip) and bumped the native window title to `Muzhen - V2.2.0`.

## [2.1.0] - 2026-11-28

### Added
- Multi-format export support: **`.vtt`** (WebVTT) and **`.txt`** (clean single-line plain-text script) alongside `.ass` and `.srt`.
- Built-in **3-language interface (`English`, `Tiếng Việt`, `中文 (简体)`)** with automatic OS locale detection, instant live switching without restart, and persistent preference storage.
- Click-to-open directly on the empty **Program Monitor** canvas.
- Compact **Updates** and **`★ wanngxuanji`** GitHub Star footer inside the Settings panel.

### Changed
- Removed the redundant top bar chrome and moved version display into the native window title (`Muzhen - V2.1.0`), moving active video filename, resolution/FPS metadata, and the Open button into the **Program Monitor** header.
- Relocated the collapsible **Settings** toggle inline with Program Monitor trim controls (`In | Full episode | Out | × | Settings | Test Frame`).
- Streamlined the **Settings** panel: segmented **Render Method** (`GPU` / `CPU` with automatic GPU detection), branded **Model** selector (**`Precision`** / **`Turbo`**), inline **ROI Preset** row, 4 compact format chips (`.ass`, `.srt`, `.vtt`, `.txt`), conditional **Custom** output folder input, and removed legacy `ASS Style` controls.

## [2.0.0] - 2026-11-28

### Added
- Adobe Premiere Pro / After Effects inspired Type-A docked 3-pane workbench UI (`Settings | Program Monitor | Subtitles`) with collapsible Settings pane, active-panel `#2D8CEB` focus outline, and live blue monospace timecode + frame index readout.
- Two-stage `ch_PP-OCRv5` OCR pipeline with mobile detection screening, DirectML 4D batch inference, and CPU fallback.
- Aegisub-aligned midpoint snapping and video-frame timing quantization for crisp subtitle boundaries.
- OP/ED In/Out trim range controls on the Program Monitor to skip intros and outros during extraction.
- ReviewTable power-user editing tools: search & replace, undo/redo history, merge/split/delete rows, QA confidence flagging, and `-1f`/`+1f` frame-accurate timing adjustments.
- Portable Windows release packaging via `PyInstaller --onedir` (`build/muzhen.spec`, `scripts/build_release.ps1`) producing `Muzhen-v2.0.0-win64-portable.zip` and `SHA256SUMS.txt`.
- Frozen runtime asset and portable `models/` directory resolution plus `--smoke-test` startup verification flag.
- GitHub Actions CI (`ci.yml`) and tag-triggered Release (`release.yml`) workflows, `LICENSE` (MIT), `CHANGELOG.md`, and `CONTRIBUTING.md`.

### Changed
- Hardened core video decoding, ROI handling, OCR worker lifecycle, and UI state management following a full codebase audit.
- Enforced zero-warning test execution (`pytest -q -W error`) across unit and Qt UI suites.

## [1.0.0] - 2025-12-01

### Added
- Initial Windows desktop release of Muzhen for extracting hardcoded Chinese/CJK subtitles from video files.
- Interactive ROI canvas, basic timeline scrubbing, RapidOCR integration, subtitle review table, and ASS/SRT export.
