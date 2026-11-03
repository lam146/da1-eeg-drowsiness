$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -m venv (Join-Path $ProjectRoot ".venv")
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        python -m venv (Join-Path $ProjectRoot ".venv")
    }
    else {
        throw "Không tìm thấy Python 3.10 trở lên."
    }
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -e "${ProjectRoot}[dev,ml-extra]"

Write-Host "Đã cài đặt xong. Python: $VenvPython"
