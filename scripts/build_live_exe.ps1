param(
    [string]$ReleaseDirectory,
    [string]$Destination,
    [string]$Name = 'N16_NhanBietBuonNgu_054',
    [string]$Python
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = Join-Path $Root '.venv\Scripts\python.exe' }
if (-not $ReleaseDirectory) { $ReleaseDirectory = Join-Path $Root 'output\N16_Chot_20261110' }
if (-not $Destination) { $Destination = Join-Path $Root 'output\N16_Live_Windows' }
$ManifestPath = (Resolve-Path -LiteralPath (Join-Path $ReleaseDirectory 'release.json')).Path
$Profile = (Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json).variants.LowRepeat
$ModelPath = (Resolve-Path -LiteralPath (Join-Path $ReleaseDirectory $Profile.file)).Path
if ((Get-FileHash -LiteralPath $ModelPath -Algorithm SHA256).Hash -ne $Profile.sha256) {
    throw 'LowRepeat model differs from the release manifest.'
}

$Work = Join-Path $Root 'tmp\exe-packaging'
New-Item -ItemType Directory -Path $Work -Force | Out-Null
$env:PYINSTALLER_CONFIG_DIR = Join-Path $Work 'cache'
$env:TEMP = $Work
$env:TMP = $Work
$BuildArguments = @(
    '-m', 'PyInstaller', '--noconfirm', '--onefile', '--windowed', '--noupx',
    '--name', $Name, '--distpath', $Destination,
    '--specpath', $Work, '--workpath', (Join-Path $Work 'build'),
    '--paths', (Join-Path $Root 'src'),
    '--add-data', "${ManifestPath}:n16_release", '--add-data', "${ModelPath}:n16_release",
    (Join-Path $PSScriptRoot 'n16_live_app.py')
)
& $Python @BuildArguments
if ($LASTEXITCODE -ne 0) { throw "PyInstaller exited with code $LASTEXITCODE." }
Write-Host (Join-Path $Destination "$Name.exe")
