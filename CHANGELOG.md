# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
