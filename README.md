<div align="center">

<img src="src/muzhen/ui/assets/muzhen_icon.png" alt="Muzhen Icon" width="88" height="88">

# Muzhen (`幕帧`)

**Frame-accurate Simplified Chinese / CJK hardsub extractor for Windows**  
*Premiere-inspired NLE workbench · DirectML GPU / CPU `ch_PP-OCRv5` · `.ass`, `.srt`, `.vtt` & `.txt` export*

[![CI](https://github.com/wanngxuanji/Muzhen/actions/workflows/ci.yml/badge.svg)](https://github.com/wanngxuanji/Muzhen/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/wanngxuanji/Muzhen?color=2D8CEB)](https://github.com/wanngxuanji/Muzhen/releases)
[![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11%20x64-1D1D1D)](#system-requirements)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-10B981)](LICENSE)

<br>

![Muzhen v2.2.0 Workbench](docs/screenshots/muzhen-v2-workbench.png)

</div>

---

## Overview

**Muzhen (`幕帧` — *Subtitle · Frame*)** is a Windows desktop application for extracting **frame-accurate hardcoded subtitles** from anime, donghua, and CJK video episodes.

Built around a keyboard-first, three-pane NLE layout (`Settings | Program Monitor | Subtitles`), Muzhen combines a two-stage `ch_PP-OCRv5` recognition pipeline with synchronized audio/video preview, an interactive review table, and Aegisub/Premiere-ready multi-format export.

---

## Download & Quick Start

### Option 1 — Windows Portable Release (Recommended)

No Python installation required:

1. Download **`Muzhen-v2.2.0-win64-portable.zip`** (and optional `SHA256SUMS.txt`) from [**GitHub Releases**](https://github.com/wanngxuanji/Muzhen/releases).
2. Extract to any writable folder (for example `D:\Tools\Muzhen\`).
3. Run **`Muzhen.exe`**.

> **First-run OCR models:** On the first extraction or **Test Frame**, Muzhen automatically downloads the required `ch_PP-OCRv5` ONNX weights into the portable `models/` folder next to `Muzhen.exe` (or `%LOCALAPPDATA%\Muzhen\models` if extracted to a read-only location). Subsequent runs work completely offline.

### Option 2 — Run From Source

```powershell
git clone https://github.com/wanngxuanji/Muzhen.git
cd Muzhen
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m muzhen
```

Optionally create a console-free Windows shortcut (`Muzhen.lnk`):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\create_shortcut.ps1
# Or place it directly on the Desktop:
powershell -ExecutionPolicy Bypass -File scripts\create_shortcut.ps1 -Desktop
```

---

## Core Features

### Frame-Accurate Hardsub Extraction
- **2-Stage `ch_PP-OCRv5` Pipeline:** Detects subtitle onset/offset spans to `±0–1 frame` across CFR, NTSC (`23.976` / `29.97 fps`), and VFR videos, then applies multi-frame consensus recognition to handle glare, motion, and back-to-back dialogue cuts.
- **GPU & CPU Modes:** Automatically enables **DirectML GPU** acceleration on compatible Windows GPUs with seamless **CPU** fallback, plus selectable **Precision** (highest CJK accuracy) and **Turbo** (faster & lighter) models.
- **Reusable ROI Presets & OP/ED Trim:** Save subtitle-area presets per series and set `In` / `Out` trim marks (`I` / `O`) to skip openings and endings while keeping original episode timestamps intact.

### Interactive NLE Preview & Synchronized Audio
- **Program Monitor:** Click directly on the empty monitor or drag-and-drop a video file to open it, drag the emerald ROI overlay around the subtitle region, and run **Test Frame** to verify OCR before scanning the full episode.
- **Synchronized Audio Playback:** Preview episodes with synced container audio (`Space` to Play/Pause, `M` to Mute/Unmute), hold-to-repeat `±1` frame stepping, and VFR-safe single-frame step-back.

### Subtitle Review Table & Multi-Format Export
- **Click-to-Seek Review:** Clicking or navigating any row in **Subtitles** automatically jumps **Program Monitor** to that cue's start frame and shows its cropped ROI thumbnail in **Inspector & Timing**.
- **Power-User Editing & QA:** Inline text and timecode editing, low-confidence cue flagging (`Ctrl+D` to mark reviewed), playhead snapping (`Ctrl+[` / `Ctrl+]`), frame-exact `-1f` / `+1f` timing nudges, Merge (`Ctrl+M`), Split (`Ctrl+K`), multi-row Delete, 50-step Undo/Redo, and Find & Replace.
- **4 Export Formats:** Export UTF-8 **`.ass`** (with Aegisub/libass-aligned inter-frame midpoint snapping), **`.srt`**, **`.vtt`** (WebVTT), and **`.txt`** (clean script ready for translation) in one click.
- **Multi-Language UI & Update Checker:** Built-in **English**, **Tiếng Việt**, and **中文 (简体)** interface with automatic OS locale detection, live switching, and startup GitHub Releases update notifications.

<details>
<summary><strong>Technical details: 2-stage OCR & timing architecture</strong></summary>

- **Stage 1 (Span Detection):** Native 4D ONNX batching (`ch_PP-OCRv5 Det Mobile`) with shared-coordinate visual signature matching to resist DBNet box jitter, paired with a single-pass ring-buffer boundary refiner.
- **Stage 2 (Consensus OCR):** Borderless `ch_PP-OCRv5` recognition (`Server` in Precision mode, `Mobile` in Turbo mode) with interior-first adaptive early exit, character-level consensus voting, glare/brightness retry, and bounded binary-search splitting for back-to-back cues.
- **ASS Timing Parity:** Inter-frame midpoint centisecond snapping matched to libass rendering semantics plus strict non-overlap clamping across `.ass`, `.srt`, and `.vtt`.

</details>

---

## Workflow

1. **Open Video & Frame ROI:** Click **Program Monitor** (or drag and drop a video onto it), scrub to a line of dialogue, and drag the emerald ROI box over the subtitle region. Optionally save the region under **ROI Preset** in **Settings**.
2. **Trim & Test (Optional):** Scrub to the end of the opening and press `I` (**In**), then to the start of the ending and press `O` (**Out**). Click **Test Frame** to check OCR accuracy on the current frame.
3. **Extract:** Choose **Render Method** (`GPU` / `CPU`) and **Model** (`Precision` / `Turbo`) in **Settings**, then click **Extract**.
4. **Review & Export:** Toggle **Settings** off for a wide Monitor + Subtitles workspace, click subtitle rows to jump directly to their frames, review any flagged lines, pick export formats (`.ass`, `.srt`, `.vtt`, `.txt`), and click **Export**.

---

## Keyboard Shortcuts

| Context | Shortcut | Action |
| :--- | :--- | :--- |
| **Transport (Global)** | `Space` | Play / Pause preview (with audio) |
| | `M` | Mute / Unmute preview audio |
| | `←` / `→` · `,` / `.` · `<` / `>` | Step `±1` frame (hold to scrub) |
| | `Shift+←` / `Shift+→` | Jump `±1` second |
| | `Ctrl+←` / `Ctrl+→` | Jump `±5` seconds |
| | `Home` / `End` | Jump to start / end of video |
| | `I` / `O` | Set OP/ED trim `In` / `Out` point |
| **Timing Sync & QA** | Click row / `↑` `↓` | Auto-seek Program Monitor to cue Start |
| | `Alt+S` / `Alt+E` | Seek video to selected cue Start / End |
| | `Ctrl+[` / `Ctrl+]` | Snap selected cue Start / End to playhead |
| | `Ctrl+D` | Mark selected cue reviewed (clear flag) |
| **Subtitle Table** | `F2` / Double-click | Edit cell inline (`\N` for line breaks) |
| | `Delete` | Delete selected row(s) |
| | `Ctrl+M` | Merge contiguous selected rows |
| | `Ctrl+K` | Split selected row at playhead or midpoint |
| | `Ctrl+Z` / `Ctrl+Y` | Undo / Redo (up to 50 steps) |
| | `Ctrl+F` / `Ctrl+H` | Open Find / Find & Replace bar |

---

## System Requirements

| Component | Minimum | Recommended |
| :--- | :--- | :--- |
| **OS** | Windows 10 (1903+) 64-bit | Windows 10 / 11 64-bit |
| **CPU** | 64-bit x86_64 quad-core | 6+ cores with AVX2 |
| **RAM** | 4 GB | 8 GB+ (1080p episodes) |
| **GPU** | Optional (automatic CPU fallback) | DirectX 12 GPU (NVIDIA, AMD, Intel) via **DirectML** |
| **Storage** | ~450 MB extracted + ~180 MB OCR models | 1 GB+ free SSD space |
| **Display** | 1280×720 | 1920×1080 or higher |
| **Media** | `.mp4`, `.mkv`, `.ts`, `.avi`, `.mov`, `.webm` (bundled PyAV/FFmpeg) | CFR, NTSC (`23.976` / `29.97 fps`), and VFR |

---

## Building the Windows Portable Release

```powershell
.venv\Scripts\python -m pip install -e ".[dev,build]"

# Portable zip (downloads OCR models on first run into ./models)
powershell -ExecutionPolicy Bypass -File scripts\build_release.ps1

# Optional: include cached local .onnx weights for a fully offline zip
powershell -ExecutionPolicy Bypass -File scripts\build_release.ps1 -BundleModels
```

Builds `release/Muzhen-v<version>-win64-portable.zip` and `release/SHA256SUMS.txt` after verifying `dist\Muzhen\Muzhen.exe --smoke-test`.

---

## Project Structure

```text
Muzhen/
├── src/muzhen/
│   ├── __main__.py             # Enables `python -m muzhen`
│   ├── app.py                  # GUI entry point, --smoke-test, crash logging
│   ├── core/
│   │   ├── video_reader.py     # PTS-accurate PyAV reader, B-frame safe seeks, VFR step-back
│   │   ├── audio_player.py     # Synchronized PyAV + QtMultimedia audio preview player
│   │   ├── ocr_engine.py       # RapidOCR v3 (ch_PP-OCRv5), 4D ONNX batching, DirectML fallback
│   │   ├── deep_extractor.py   # 2-stage ring-buffer detector & consensus voter
│   │   ├── postprocess.py      # Normalization, merging, minimal-pair guard, QA flagging
│   │   ├── pipeline.py         # End-to-end extraction orchestrator & trim bounds
│   │   ├── updater.py          # GitHub Releases update checker & user preferences
│   │   ├── paths.py            # Dev vs frozen asset & portable models/ path resolution
│   │   ├── models.py           # Core dataclasses (RoiRect, ExtractConfig, SubtitleEvent)
│   │   └── presets.py          # Atomic JSON ROI preset storage
│   ├── exporters/
│   │   ├── sanitize.py         # Libass midpoint snapping, escaping, atomic UTF-8 writes
│   │   ├── ass_exporter.py     # Aegisub .ass exporter
│   │   ├── srt_exporter.py     # SubRip .srt exporter
│   │   ├── vtt_exporter.py     # WebVTT .vtt exporter
│   │   └── txt_exporter.py     # Plain-text .txt exporter
│   └── ui/
│       ├── assets/             # Rounded muzhen.ico, muzhen_icon.png, vector SVGs
│       ├── i18n.py             # Live English / Vietnamese / Simplified Chinese translations
│       ├── theme.py            # Premiere/AE neutral graphite dark palette & QSS
│       ├── main_window.py      # 3-pane workbench, transport, async workers, UpdateDialog
│       ├── review_table.py     # Editable subtitle table, QA inspector, undo/redo, find/replace
│       ├── roi_canvas.py       # Interactive ROI overlay & clickable drop monitor
│       └── icons.py            # HiDPI vector SVG icon renderer
├── build/muzhen.spec           # PyInstaller --onedir specification
├── scripts/
│   ├── build_release.ps1       # Portable release builder + smoke test + SHA256
│   └── create_shortcut.ps1     # Windows Muzhen.lnk launcher generator
├── tests/                      # Automated pytest + pytest-qt suite (225 tests, -W error)
└── .github/workflows/          # Windows CI (ci.yml) & tag-triggered Release (release.yml)
```

---

## Testing & Contributing

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.venv\Scripts\pytest -q -W error
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md) for development guidelines and release history.

## License

Released under the [MIT License](LICENSE).
