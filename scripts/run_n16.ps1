param(
    [ValidateSet('Main', 'Spectral', 'LowRepeat')]
    [string]$Variant = 'LowRepeat',
    [string]$ReleaseDirectory,
    [ValidatePattern('^COM\d+$')]
    [string]$Port,
    [ValidateRange(0, 100)]
    [double]$Speed = 1,
    [ValidateRange(0.01, 120)]
    [double]$DurationMin = 5,
    [switch]$Sound
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw 'Python environment missing. Run scripts/setup.ps1 first.'
}
if (-not $ReleaseDirectory) {
    $ReleaseDirectory = Join-Path $Root 'output\N16_Reviewed_20261127\release'
    if (-not (Test-Path -LiteralPath (Join-Path $ReleaseDirectory 'release.json'))) {
        $ReleaseDirectory = Join-Path $Root 'output\N16_Chot_20261110'
    }
}
$ManifestPath = Join-Path $ReleaseDirectory 'release.json'
if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
    throw 'Local N16 release missing. Use scripts/run_demo.ps1 for the synthetic smoke test on a clean checkout.'
}
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$Profile = $Manifest.variants.PSObject.Properties[$Variant].Value
if (-not $Profile) { throw "Variant $Variant missing from $ReleaseDirectory. Select an available variant using -Variant." }

function Get-VerifiedArtifact($Artifact) {
    $Path = Join-Path $ReleaseDirectory $Artifact.file
    if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -ne $Artifact.sha256) {
        throw "Artifact differs from the frozen N16 release: $Path"
    }
    return $Path
}

$Model = Get-VerifiedArtifact $Profile
$OutputDirectory = Join-Path $ReleaseDirectory 'runs'
$Timestamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$Log = Join-Path $OutputDirectory "${Timestamp}_${Variant}.csv"
$Policy = @('--model', $Model, '--window-s', $Manifest.window_s,
    '--step-s', $Manifest.step_s, '--consecutive', $Manifest.consecutive,
    '--release-consecutive', $Profile.release_consecutive)
if ($null -ne $Profile.support_threshold) {
    $Policy += @('--support-threshold', $Profile.support_threshold,
        '--support-consecutive', $Profile.support_consecutive)
}
if ($Profile.screen_raw) { $Policy += '--screen-raw' }

Write-Host "N16 $Variant | threshold=$($Profile.threshold) | log=$Log"
if ($Port) {
    $Arguments = @('-m', 'n16_eeg.live', '--port', $Port, '--log-output', $Log,
        '--duration-min', $DurationMin) + $Policy
    if (-not $Sound) { $Arguments += '--no-sound' }
} else {
    $Fixture = Get-VerifiedArtifact $Manifest.fixture
    $Arguments = @('-m', 'n16_eeg.demo', $Fixture, '--output-log', $Log, '--speed', $Speed) + $Policy
    if ($Sound) { $Arguments += '--sound' }
}
& $Python @Arguments
if ($LASTEXITCODE -ne 0) { throw "N16 exited with code $LASTEXITCODE" }
