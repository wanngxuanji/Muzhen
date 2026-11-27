# TransApp

Windows desktop application (PySide6 Dark Studio) for extracting frame-accurate hardcoded Simplified Chinese / CJK subtitles from anime, donghua, and video episodes into editable `.ass` (Aegisub) and `.srt` (Adobe Premiere Pro) tracks.

## Highlights (v2)

- **2-Stage Deep Learning Pipeline (`ch_PP-OCRv5`)**:
  - **Stage 1**: Native 4D ONNX-batched `ch_PP-OCRv5 Det Mobile` + tight-box visual signatures + single-pass streaming ring-buffer refinement for **frame-accurate (`±0–1 frame`)** start, end, and `0ms` back-to-back dialogue cuts across CFR, NTSC (`23.976` / `29.97fps`), and VFR videos.
  - **Stage 2**: Borderless `ch_PP-OCRv5 Server` (with optional `Mobile` preset) using interior-first adaptive early-exit, confidence-weighted & character-level consensus voting, selective brightness-filtered retry, and targeted binary-search splitting.
- **Aegisub & Premiere Pro Ready Export**:
  - Aegisub/libass inter-frame midpoint centisecond snapping (zero 1-frame late starts or cut bleeds), automatic `{}` & `\N` escaping, `[Fonts]`/`[Graphics]` preservation, multi-style `.ass` template selector, identical `.ass`/`.srt` cue parity, and strict non-overlap clamping.
- **Interactive Video Preview & OP/ED Trim**:
  - Sequential `Play / Pause` (`Space`), hold-to-repeat stepping (`<` / `>`, `←` / `→`, `,` / `.`), `Shift`/`Ctrl` fast jumps, drag-and-drop video loading, and `In / Out` (`I` / `O`) OP/ED trim that preserves original video timestamps.
- **Pro-Subber Review Table**:
  - Video↔Table Start/End sync (`Alt+S` / `Alt+E`), snap-to-playhead (`Ctrl+[` / `Ctrl+]`), `±1f` / `±100ms` nudge, `flag_reason` QA banner with in-place auto-clear on edit (`Ctrl+D`), multi-row delete, Merge (`Ctrl+M`), Split (`Ctrl+K`), 50-step Undo/Redo (`Ctrl+Z` / `Ctrl+Y`), and Find & Replace (`Ctrl+F` / `Ctrl+H`).

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\transapp
```

### One-Click Windows Launcher (No Console Window)

`transapp` is registered under `[project.gui-scripts]`, so `.venv\Scripts\transapp.exe` launches directly as a Windows GUI app without a black console window. To create a clickable `TransApp.lnk` shortcut with the app icon:

```powershell
# Creates TransApp.lnk in the project folder (pin to Taskbar or copy anywhere)
powershell -ExecutionPolicy Bypass -File scripts\create_shortcut.ps1

# Or place directly on Desktop:
powershell -ExecutionPolicy Bypass -File scripts\create_shortcut.ps1 -Desktop
```

> On Windows, dependencies include `onnxruntime-directml` so both **DirectML (GPU)** and **CPU (ONNX Runtime)** are available in the OCR Backend selector.

## Keyboard Shortcuts

| Context | Shortcuts |
| :--- | :--- |
| **Video Preview** | `Space` Play/Pause · `←`/`→` or `,`/`.` or `<`/`>` ±1 frame · `Shift+←/→` ±1s · `Ctrl+←/→` ±5s · `Home`/`End` · `I`/`O` Set In/Out |
| **Timing Sync** | `Alt+S` / `Alt+E` Jump to Start/End · `Ctrl+[` / `Ctrl+]` Snap Start/End to playhead · `Ctrl+D` Mark reviewed |
| **Review Table** | `F2` / Double-click Edit cell · `Delete` Delete selected · `Ctrl+M` Merge · `Ctrl+K` Split · `Ctrl+Z` / `Ctrl+Y` Undo/Redo · `Ctrl+F` / `Ctrl+H` Find/Replace |

## Testing

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.venv\Scripts\pytest -v -W error
```
