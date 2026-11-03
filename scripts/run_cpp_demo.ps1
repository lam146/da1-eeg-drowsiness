$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Executable = Join-Path $ProjectRoot "cpp\build\n16_runtime.exe"
$Model = Join-Path $ProjectRoot "output\model\n16_logistic_model.json"
$Session = Join-Path $ProjectRoot "data\synthetic\demo_session.csv"
$Log = Join-Path $ProjectRoot "output\demo\cpp_replay_log.csv"

if (-not (Test-Path $Executable)) {
    & (Join-Path $PSScriptRoot "build_cpp.ps1")
}

if (-not (Test-Path $Model) -or -not (Test-Path $Session)) {
    $VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $VenvPython)) {
        throw "Thiếu model/dữ liệu demo. Hãy chạy scripts\setup.ps1 trước."
    }
    Push-Location $ProjectRoot
    try {
        & $VenvPython -m n16_eeg.workflow --root $ProjectRoot
    }
    finally {
        Pop-Location
    }
}

Push-Location $ProjectRoot
try {
    & $Executable replay --input $Session --model $Model --log $Log
}
finally {
    Pop-Location
}
