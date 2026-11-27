# Contributing to Muzhen

Thank you for contributing to Muzhen. This guide covers local Windows setup, testing, building portable releases, and pull request conventions.

## Prerequisites

- Windows 10 or Windows 11 (64-bit)
- Python 3.11 or newer
- Git

## Development Setup

Create a virtual environment and install Muzhen in editable mode with development dependencies:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install -e ".[dev]"
```

Launch the desktop app from source:

```powershell
.venv\Scripts\python -m muzhen.app
```

## Running Tests

Run the full test suite headless with warnings treated as errors (PowerShell 5.1):

```powershell
$env:QT_QPA_PLATFORM='offscreen'; .venv\Scripts\pytest -q -W error
```

All existing and new tests must pass with zero warnings before opening a pull request.

## Building a Portable Windows Release

Install the optional PyInstaller build extra and run the release build script:

```powershell
.venv\Scripts\pip install -e ".[dev,build]"
powershell -ExecutionPolicy Bypass -File scripts\build_release.ps1
```

Optional flags for `scripts\build_release.ps1`:

- `-Version 2.0.0`: override the version read from `pyproject.toml`.
- `-BundleModels`: copy cached `.onnx` OCR models from `.venv` into `dist\Muzhen\models\` for offline portable archives.
- `-SkipSmokeTest`: skip the headless `Muzhen.exe --smoke-test` verification step.

Artifacts are written to `release\Muzhen-v<version>-win64-portable.zip` and `release\SHA256SUMS.txt`.

## Pull Request Conventions

- Keep pull requests focused on a single feature, fix, or refactor.
- Use clear [Conventional Commits](https://www.conventionalcommits.org/) messages (for example, `feat(ui): ...`, `fix(ocr): ...`, `build: ...`, `docs: ...`).
- Preserve existing widget attribute names, signals, keyboard shortcuts, and the `MainWindow.minimumSizeHint() <= 1280x672` layout floor when modifying the UI.
- Add or update unit/UI tests in `tests/` for behavioral changes and verify with `pytest -q -W error`.
- Do not commit large media files, generated subtitle outputs, `dist/`, `release/`, or PyInstaller temporary build outputs.
