param(
    [string]$ReleaseDirectory,
    [string]$Destination,
    [string]$Name = 'N16_EEG_Monitor',
    [string]$Python
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = Join-Path $Root '.venv\Scripts\python.exe' }
$Published = Join-Path $Root 'releases\v0.4.0'
if (-not $ReleaseDirectory) {
    $ReleaseDirectory = Join-Path $Root 'output\N16_Optimize4_20261128\profile'
    if (-not (Test-Path -LiteralPath (Join-Path $ReleaseDirectory 'release.json'))) {
        $ReleaseDirectory = Join-Path $Root 'output\N16_Reviewed_20261127\release'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $ReleaseDirectory 'release.json'))) {
        $ReleaseDirectory = Join-Path $Root 'output\N16_Chot_20261110'
    }
    if (Test-Path -LiteralPath (Join-Path $Published 'candidate\release.json')) {
        $ReleaseDirectory = Join-Path $Published 'candidate'
    }
}
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
$ComparisonArgs = @()
$ComparisonDirectory = Join-Path $Root 'output\N16_Research_20261128\welch_blend'
if (Test-Path -LiteralPath (Join-Path $Published 'reference\release.json')) {
    $ComparisonDirectory = Join-Path $Published 'reference'
}
$ComparisonManifest = Join-Path $ComparisonDirectory 'release.json'
if (Test-Path -LiteralPath $ComparisonManifest) {
    $ComparisonProfile = (Get-Content -LiteralPath $ComparisonManifest -Raw -Encoding UTF8 | ConvertFrom-Json).variants.LowRepeat
    $ComparisonModel = Join-Path $ComparisonDirectory $ComparisonProfile.file
    if ((Get-FileHash -LiteralPath $ComparisonModel -Algorithm SHA256).Hash -ne $ComparisonProfile.sha256) {
        throw 'Comparison model differs from the release manifest.'
    }
    $ComparisonArgs = @('--add-data', "${ComparisonManifest}:n16_reference", '--add-data', "${ComparisonModel}:n16_reference")
}
$BuildArguments = @(
    '-m', 'PyInstaller', '--noconfirm', '--onefile', '--windowed', '--noupx',
    '--name', $Name, '--distpath', $Destination,
    '--specpath', $Work, '--workpath', (Join-Path $Work 'build'),
    '--paths', (Join-Path $Root 'src'),
    '--icon', (Join-Path $PSScriptRoot 'n16_monitor.ico'),
    '--add-data', "$(Join-Path $PSScriptRoot 'n16_monitor.ico'):.",
    '--add-data', "${ManifestPath}:n16_release", '--add-data', "${ModelPath}:n16_release"
) + $ComparisonArgs + @((Join-Path $PSScriptRoot 'n16_live_app.py'))
& $Python @BuildArguments
if ($LASTEXITCODE -ne 0) { throw "PyInstaller exited with code $LASTEXITCODE." }
Write-Host (Join-Path $Destination "$Name.exe")
