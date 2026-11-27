# TransApp

Windows desktop application for extracting hardcoded Chinese/CJK subtitles from video files into editable ASS/SRT subtitle tracks.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\transapp
```

> On Windows, dependencies include `onnxruntime-directml` so both `DirectML (GPU)` and `CPU (ONNX Runtime)` are available in the OCR Backend selector.

## Testing

```powershell
.venv\Scripts\pytest -v
```
