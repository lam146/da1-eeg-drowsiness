$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$CppRoot = Join-Path $ProjectRoot "cpp"
$BuildRoot = Join-Path $CppRoot "build"
$Executable = Join-Path $BuildRoot "n16_runtime.exe"
$Compiler = Get-Command g++ -ErrorAction SilentlyContinue

if (-not $Compiler) {
    throw "Không tìm thấy g++. Hãy cài MinGW-w64 hoặc dùng CMake với Visual Studio Build Tools."
}

New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null
$Sources = (Get-ChildItem -LiteralPath (Join-Path $CppRoot "src") -Filter "*.cpp" -File).FullName

& $Compiler.Source -std=c++17 -O2 -Wall -Wextra -Wpedantic `
    -I (Join-Path $CppRoot "include") $Sources -o $Executable

if ($LASTEXITCODE -ne 0) {
    throw "Biên dịch C++ thất bại."
}

& $Executable self-test
if ($LASTEXITCODE -ne 0) {
    throw "C++ self-test thất bại."
}

Write-Host "C++ build completed: $Executable"
