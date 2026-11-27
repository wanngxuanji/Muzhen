param(
    [switch]$Desktop
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$TargetExe = Join-Path $ProjectRoot ".venv\Scripts\transapp.exe"
$IconPath = Join-Path $ProjectRoot "src\transapp\ui\assets\transapp.ico"

if (-not (Test-Path -LiteralPath $TargetExe)) {
    throw "Cannot find $TargetExe. Run '.venv\Scripts\python -m pip install -e "".[dev]""' first."
}

$ShortcutDir = if ($Desktop) {
    [Environment]::GetFolderPath("Desktop")
} else {
    $ProjectRoot
}
if ([string]::IsNullOrWhiteSpace($ShortcutDir) -or -not (Test-Path -LiteralPath $ShortcutDir)) {
    throw "Cannot resolve shortcut directory '$ShortcutDir'."
}
$ShortcutPath = Join-Path $ShortcutDir "TransApp.lnk"

$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $TargetExe
$Shortcut.WorkingDirectory = $ProjectRoot
if (Test-Path -LiteralPath $IconPath) {
    $Shortcut.IconLocation = "$IconPath,0"
}
$Shortcut.Description = "TransApp - Tach Sub Raw Tieng Trung"
$Shortcut.Save()

Write-Output "Created shortcut: $ShortcutPath -> $TargetExe"
