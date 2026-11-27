param(
    [string]$Version = "",
    [switch]$BundleModels,
    [switch]$SkipSmokeTest
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $repoRoot
try {
    $pyprojectPath = Join-Path $repoRoot "pyproject.toml"
    if ([string]::IsNullOrWhiteSpace($Version)) {
        if (-not (Test-Path -LiteralPath $pyprojectPath)) {
            throw "pyproject.toml not found at $pyprojectPath"
        }
        $pyprojectText = Get-Content -LiteralPath $pyprojectPath -Raw -Encoding UTF8
        $match = [regex]::Match($pyprojectText, '(?m)^\s*version\s*=\s*"([^"]+)"')
        if (-not $match.Success) {
            throw "Could not read project version from $pyprojectPath"
        }
        $Version = $match.Groups[1].Value
    }
    $Version = $Version.Trim().TrimStart("v", "V")
    if ([string]::IsNullOrWhiteSpace($Version)) {
        throw "Resolved version is empty"
    }

    $venvPyInstaller = Join-Path $repoRoot ".venv\Scripts\pyinstaller.exe"
    if (Test-Path -LiteralPath $venvPyInstaller) {
        $pyinstallerExe = $venvPyInstaller
    } else {
        $cmd = Get-Command pyinstaller -ErrorAction SilentlyContinue
        if ($null -eq $cmd) {
            throw "PyInstaller not found in .venv\Scripts\pyinstaller.exe or PATH"
        }
        $pyinstallerExe = $cmd.Source
    }

    $specPath = Join-Path $repoRoot "build\transapp.spec"
    if (-not (Test-Path -LiteralPath $specPath)) {
        throw "PyInstaller spec file not found at $specPath"
    }

    $workPath = Join-Path $repoRoot "build\pyinstaller"
    $distRoot = Join-Path $repoRoot "dist"
    $distDir = Join-Path $distRoot "TransApp"
    $exePath = Join-Path $distDir "TransApp.exe"

    $previousEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $pyinstallerExe --clean --noconfirm --workpath $workPath --distpath $distRoot $specPath
        $pyinstallerExit = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousEap
    }
    if ($pyinstallerExit -ne 0) {
        throw "PyInstaller failed with exit code $pyinstallerExit"
    }

    if (-not (Test-Path -LiteralPath $exePath)) {
        throw "Expected executable not found at $exePath"
    }

    if ($BundleModels) {
        $adjacentPython = Join-Path (Split-Path -Parent $pyinstallerExe) "python.exe"
        if (Test-Path -LiteralPath $adjacentPython) {
            $pythonExe = $adjacentPython
        } else {
            $pyCmd = Get-Command python -ErrorAction SilentlyContinue
            if ($null -eq $pyCmd) {
                throw "Python executable not found next to $pyinstallerExe or on PATH"
            }
            $pythonExe = $pyCmd.Source
        }

        $previousPyEap = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $resolvedModelsDir = (& $pythonExe -c "import pathlib, rapidocr; print(pathlib.Path(rapidocr.__file__).resolve().parent / 'models')" | Select-Object -Last 1)
            $pyExit = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $previousPyEap
        }
        if ($pyExit -ne 0 -or [string]::IsNullOrWhiteSpace($resolvedModelsDir)) {
            throw "Failed to resolve RapidOCR models directory using $pythonExe"
        }

        $sourceModelsDir = $resolvedModelsDir.Trim()
        if (-not (Test-Path -LiteralPath $sourceModelsDir)) {
            throw "BundleModels requested, but RapidOCR models directory does not exist at $sourceModelsDir"
        }

        $onnxFiles = @(Get-ChildItem -LiteralPath $sourceModelsDir -Filter "*.onnx" -File)
        if ($onnxFiles.Count -eq 0) {
            throw "BundleModels requested, but 0 .onnx model files were found in $sourceModelsDir"
        }

        $targetModelsDir = Join-Path $distDir "models"
        New-Item -ItemType Directory -Force -Path $targetModelsDir | Out-Null
        foreach ($modelFile in $onnxFiles) {
            Copy-Item -LiteralPath $modelFile.FullName -Destination (Join-Path $targetModelsDir $modelFile.Name) -Force
        }
    }

    if (-not $SkipSmokeTest) {
        $previousPlatform = $env:QT_QPA_PLATFORM
        $env:QT_QPA_PLATFORM = "offscreen"
        try {
            $smokeProc = Start-Process -FilePath $exePath -ArgumentList "--smoke-test" -WorkingDirectory $distDir -Wait -PassThru
            if ($smokeProc.ExitCode -ne 0) {
                if (-not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
                    $crashLogPath = Join-Path $env:LOCALAPPDATA "TransApp\crash.log"
                    if (Test-Path -LiteralPath $crashLogPath) {
                        Write-Output "Recent crash.log lines ($crashLogPath):"
                        Get-Content -LiteralPath $crashLogPath -Tail 80 -Encoding UTF8 | ForEach-Object { Write-Output $_ }
                    }
                }
                throw "Smoke test failed with exit code $($smokeProc.ExitCode)"
            }
        } finally {
            if ($null -eq $previousPlatform) {
                Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
            } else {
                $env:QT_QPA_PLATFORM = $previousPlatform
            }
        }
    }

    $releaseDir = Join-Path $repoRoot "release"
    New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null

    $zipName = "TransApp-v$Version-win64-portable.zip"
    $zipPath = Join-Path $releaseDir $zipName
    if (Test-Path -LiteralPath $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }

    Compress-Archive -LiteralPath $distDir -DestinationPath $zipPath -Force

    $sha256 = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $checksumPath = Join-Path $releaseDir "SHA256SUMS.txt"
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($checksumPath, "$sha256  $zipName`n", $utf8NoBom)

    Write-Output "Archive: $zipPath"
    Write-Output "SHA256:  $sha256"
} finally {
    Pop-Location
}
