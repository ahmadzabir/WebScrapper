param(
    [switch]$Clean,
    [string]$Version = "1.0.0"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv-desktop"
$DistPath = Join-Path $ProjectRoot "dist"
$BuildPath = Join-Path $ProjectRoot "build"
$InstallerPath = Join-Path $ProjectRoot "dist\WebScrapperDesktop-Setup.exe"

Write-Host "==> Project root: $ProjectRoot"

if ($Clean) {
    Write-Host "==> Cleaning old build artifacts"
    if (Test-Path $DistPath) { Remove-Item -Recurse -Force $DistPath }
    if (Test-Path $BuildPath) { Remove-Item -Recurse -Force $BuildPath }
}

if (-not (Test-Path $VenvPath)) {
    Write-Host "==> Creating virtual environment"
    py -3.11 -m venv $VenvPath
}

$PythonExe = Join-Path $VenvPath "Scripts\python.exe"
$PipExe = Join-Path $VenvPath "Scripts\pip.exe"

Write-Host "==> Installing dependencies"
& $PythonExe -m pip install --upgrade pip wheel setuptools
& $PipExe install -r (Join-Path $ProjectRoot "requirements.txt")
& $PipExe install -r (Join-Path $ProjectRoot "desktop_app\requirements-desktop.txt")

# Write VERSION file for launcher to read
$VersionFile = Join-Path $ProjectRoot "VERSION"
$Version | Set-Content -Path $VersionFile -NoNewline

Write-Host "==> Building desktop executable (version $Version)"
Push-Location $ProjectRoot
try {
    & $PythonExe -m PyInstaller --noconfirm --clean (Join-Path $ProjectRoot "desktop_app\WebScrapperDesktop.spec")
}
finally {
    Pop-Location
}

$ExePath = Join-Path $ProjectRoot "dist\WebScrapperDesktop.exe"
if (-not (Test-Path $ExePath)) {
    throw "Build failed: executable not found at $ExePath"
}

$ReleaseZip = Join-Path $ProjectRoot "dist\WebScrapperDesktop-Windows.zip"
if (Test-Path $ReleaseZip) {
    Remove-Item -Force $ReleaseZip
}

Write-Host "==> Creating release zip"
Compress-Archive -Path $ExePath -DestinationPath $ReleaseZip

Write-Host "==> Attempting to build Windows installer (.exe)"
$isccCandidates = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles}\Inno Setup 6\ISCC.exe"
)
$isccExe = $isccCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $isccExe) {
    $isccExe = (Get-Command "iscc.exe" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue)
}

if ($isccExe) {
    Write-Host "==> Inno Setup compiler found at: $isccExe"
    Push-Location $ProjectRoot
    try {
        & $isccExe "/DMyAppVersion=$Version" (Join-Path $ProjectRoot "desktop_app\WebScrapperDesktop.iss")
    }
    finally {
        Pop-Location
    }
}
else {
    Write-Warning "Inno Setup compiler not found. Skipping installer build."
    Write-Host "Install Inno Setup 6 to produce WebScrapperDesktop-Setup.exe."
}

Write-Host ""
Write-Host "Build completed successfully."
Write-Host "Executable: $ExePath"
Write-Host "Zip contains: WebScrapperDesktop.exe (extract and run)"
Write-Host "Release zip: $ReleaseZip"
if (Test-Path $InstallerPath) {
    Write-Host "Installer: $InstallerPath"
}
