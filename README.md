<div align="center">

<img src="src/muzhen/ui/assets/muzhen_icon.png" alt="Muzhen Icon" width="84" height="84">

# Muzhen (`幕帧`)

**Frame-accurate Simplified Chinese / CJK hardsub extractor for Windows**

[![CI](https://github.com/wanngxuanji/Muzhen/actions/workflows/ci.yml/badge.svg)](https://github.com/wanngxuanji/Muzhen/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/wanngxuanji/Muzhen?color=2D8CEB)](https://github.com/wanngxuanji/Muzhen/releases)
[![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11%20x64-1D1D1D)](#requirements)
[![License: MIT](https://img.shields.io/badge/license-MIT-10B981)](LICENSE)

<br>

![Muzhen Workbench](docs/screenshots/muzhen-v2-workbench.png)

</div>

---

## Features

- **2-stage `ch_PP-OCRv5` extraction** with DirectML GPU acceleration, automatic CPU fallback, and `Precision` / `Turbo` models.
- **Frame-accurate timing** across CFR, NTSC (`23.976` / `29.97 fps`), and VFR videos, with OP/ED trim (`In` / `Out`) and reusable ROI presets.
- **Built-in review workbench** with synced audio/video preview, click-to-seek subtitle table, low-confidence flagging, inline editing, and `±1` frame timing controls.
- **Multi-format export** to `.ass` (Aegisub/libass-aligned), `.srt`, `.vtt`, and `.txt`.
- **Multilingual UI** in English, Tiếng Việt, and 中文 (简体).

## Download

1. Download **`Muzhen-v2.2.0-win64-portable.zip`** from [**GitHub Releases**](https://github.com/wanngxuanji/Muzhen/releases).
2. Extract to any folder and run **`Muzhen.exe`**.

> On first extraction (or **Test Frame**), Muzhen downloads the `ch_PP-OCRv5` ONNX models once into `./models`. Subsequent runs work offline.

## Requirements

- **OS:** Windows 10 / 11 (64-bit)
- **RAM:** 4 GB minimum (8 GB recommended)
- **GPU:** Optional DirectX 12 GPU for DirectML acceleration (falls back to CPU automatically)
- **Disk:** ~1 GB free space
- **Video:** `.mp4`, `.mkv`, `.ts`, `.avi`, `.mov`, `.webm`

## Shortcuts

| Shortcut | Action |
| :--- | :--- |
| `Space` / `M` | Play / Pause · Mute / Unmute |
| `←` / `→` (or `,` / `.`) | Step `±1` frame (`Shift` ±1s, `Ctrl` ±5s) |
| `I` / `O` | Set trim `In` / `Out` |
| `Alt+S` / `Alt+E` | Seek to cue Start / End |
| `Ctrl+[` / `Ctrl+]` | Snap cue Start / End to playhead |
| `Ctrl+D` | Mark flagged cue as reviewed |
| `Ctrl+M` / `Ctrl+K` | Merge / Split subtitle rows |
| `Ctrl+F` / `Ctrl+H` | Find / Replace |
| `Ctrl+Z` / `Ctrl+Y` | Undo / Redo |

## Development

```powershell
git clone https://github.com/wanngxuanji/Muzhen.git
cd Muzhen
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,build]"

# Run app
.venv\Scripts\python -m muzhen

# Run tests
$env:QT_QPA_PLATFORM='offscreen'
.venv\Scripts\pytest -q -W error

# Build portable release
powershell -ExecutionPolicy Bypass -File scripts\build_release.ps1
```

See [CHANGELOG.md](CHANGELOG.md) and [CONTRIBUTING.md](CONTRIBUTING.md) for release history and contribution guidelines.

## License

[MIT](LICENSE)
