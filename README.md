# TransApp

Windows desktop application for extracting hardcoded Chinese/CJK subtitles from video files into editable ASS/SRT subtitle tracks.

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

> On Windows, dependencies include `onnxruntime-directml` so both `DirectML (GPU)` and `CPU (ONNX Runtime)` are available in the OCR Backend selector.

## Testing

```powershell
.venv\Scripts\pytest -v
```
