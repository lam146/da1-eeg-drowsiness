param(
    [switch]$NoGui
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    throw "Chưa có .venv. Hãy chạy scripts\setup.ps1 trước."
}

Push-Location $ProjectRoot
try {
    & $VenvPython -m n16_eeg.workflow --root $ProjectRoot
    & $VenvPython -m n16_eeg.gui --check
    if (-not $NoGui) {
        & $VenvPython -m n16_eeg.gui
    }
}
finally {
    Pop-Location
}
