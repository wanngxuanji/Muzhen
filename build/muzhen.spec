import importlib.util
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

ROOT = Path(SPECPATH).resolve().parent
EXCLUDED_RAPIDOCR_ENGINES = (
    "rapidocr.inference_engine.mnn",
    "rapidocr.inference_engine.openvino",
    "rapidocr.inference_engine.paddle",
    "rapidocr.inference_engine.pytorch",
    "rapidocr.inference_engine.tensorrt",
)

rapidocr_datas = [
    item
    for item in collect_data_files("rapidocr")
    if not item[0].lower().endswith(".onnx")
    and "pytorch" not in Path(item[0]).parts
]

datas = [
    (str(ROOT / "src" / "muzhen" / "ui" / "assets"), "muzhen/ui/assets"),
    *rapidocr_datas,
    *collect_data_files("onnxruntime"),
]

av_binaries = list(collect_dynamic_libs("av"))
if not av_binaries:
    av_spec = importlib.util.find_spec("av")
    if av_spec is not None and av_spec.origin:
        av_libs_dir = Path(av_spec.origin).resolve().parent.parent / "av.libs"
        if av_libs_dir.is_dir():
            av_binaries = [
                (str(dll), "av.libs") for dll in sorted(av_libs_dir.glob("*.dll"))
            ]

binaries = [
    *collect_dynamic_libs("onnxruntime"),
    *av_binaries,
]

rapidocr_hiddenimports = collect_submodules(
    "rapidocr",
    filter=lambda name: not any(
        name == prefix or name.startswith(f"{prefix}.")
        for prefix in EXCLUDED_RAPIDOCR_ENGINES
    ),
)

hiddenimports = sorted(
    set(
        [
            *rapidocr_hiddenimports,
            "rapidocr.ch_ppocr_det",
            "rapidocr.ch_ppocr_rec",
            "rapidocr.ch_ppocr_cls",
            "rapidocr.cal_rec_boxes",
            "rapidocr.inference_engine.onnxruntime",
            "rapidocr.inference_engine.onnxruntime.main",
            "rapidocr.inference_engine.onnxruntime.provider_config",
            "omegaconf",
            "yaml",
            "pyclipper",
            "shapely",
            "colorlog",
            "cv2",
            "av",
            "onnxruntime",
            "PySide6.QtSvg",
            "PySide6.QtMultimedia",
        ]
    )
)

excludes = [
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtWebChannel",
    "PySide6.QtWebSockets",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.Qt3DInput",
    "PySide6.Qt3DLogic",
    "PySide6.Qt3DExtras",
    "PySide6.Qt3DAnimation",
    "PySide6.QtQuick",
    "PySide6.QtQuickWidgets",
    "PySide6.QtQuickControls2",
    "PySide6.QtQuick3D",
    "PySide6.QtQml",
    "PySide6.QtQmlModels",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.QtGraphs",
    "PySide6.QtBluetooth",
    "PySide6.QtSensors",
    "PySide6.QtPositioning",
    "PySide6.QtLocation",
    "PySide6.QtNfc",
    "PySide6.QtSerialPort",
    "PySide6.QtSql",
    "PySide6.QtTest",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtDesigner",
    "PySide6.QtHelp",
    "tkinter",
    "_tkinter",
    "matplotlib",
    "pytest",
    "pytestqt",
]

a = Analysis(
    [str(ROOT / "src" / "muzhen" / "app.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Muzhen",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "src" / "muzhen" / "ui" / "assets" / "muzhen.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Muzhen",
)
