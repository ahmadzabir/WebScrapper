param(
    [switch]$Clean,
    [switch]$Test,
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
    # Use 'python' from PATH when available (e.g. GitHub Actions setup-python); else py -3.11
    $pythonCmd = Get-Command "python" -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        & python -m venv $VenvPath
    } else {
        & py -3.11 -m venv $VenvPath
    }
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

if ($Test) {
    Write-Host "==> Running launch test (5 seconds) to verify exe starts without crash..."
    $proc = Start-Process -FilePath $ExePath -PassThru -WindowStyle Normal
    Start-Sleep -Seconds 5
    if ($proc.HasExited) {
        if ($proc.ExitCode -ne 0) {
            throw "Launch test FAILED: exe exited with code $($proc.ExitCode). Fix the build before releasing."
        }
    }
    if (-not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        Write-Host "==> Launch test PASSED: exe started and ran for 5 seconds."
    }
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
    $issPath = Join-Path $ProjectRoot "desktop_app\WebScrapperDesktop.iss"
    if (-not (Test-Path $issPath)) {
        throw "Inno Setup script not found: $issPath"
    }
    # Remove any InfoBeforeFile= line to avoid desktop_app\desktop_app path resolution in CI
    $issLines = Get-Content -Path $issPath -Encoding UTF8
    $changed = $false
    $issLines = $issLines | ForEach-Object {
        if ($_ -match '^\s*InfoBeforeFile=') {
            $changed = $true
            return "; $_ (removed for CI)"
        }
        $_
    }
    if ($changed) {
        $issLines | Set-Content -Path $issPath -Encoding UTF8
        Write-Host "==> Removed InfoBeforeFile line from WebScrapperDesktop.iss for CI"
    }
    # Run ISCC from desktop_app so .iss relative paths (..\dist, etc.) resolve correctly
    Push-Location (Join-Path $ProjectRoot "desktop_app")
    try {
        & $isccExe "/DMyAppVersion=$Version" "WebScrapperDesktop.iss"
        if ($LASTEXITCODE -ne 0) {
            throw "Inno Setup failed with exit code $LASTEXITCODE"
        }
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
