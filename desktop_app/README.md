# WebScrapper Desktop (Windows)

This package wraps the existing Streamlit app in a Windows launcher.

## What It Does

- Starts your existing `python-scraper.py` app locally.
- Opens the app in the default browser.
- Uses a local loopback address (`127.0.0.1`).
- Bundles into a desktop-ready Windows executable.

## Build Locally (Windows)

From the project root:

```powershell
./desktop_app/build_windows.ps1 -Clean
```

Outputs:

- `dist/WebScrapperDesktop/WebScrapperDesktop.exe`
- `dist/WebScrapperDesktop-Windows.zip`
- `dist/WebScrapperDesktop-Setup.exe` (if Inno Setup is installed)

## Installer Packaging

This project includes an Inno Setup script:

- `desktop_app/WebScrapperDesktop.iss`

The installer creates Start Menu entries and optional desktop shortcut, so end users install it like a normal Windows app.

## Smoke Test Checklist

1. Run `WebScrapperDesktop.exe`.
2. Click `Start App`.
3. Confirm browser opens a local URL.
4. Upload a small CSV and run scrape.
5. Validate downloads and app shutdown.

## Troubleshooting

**Windows SmartScreen / Antivirus:** The app is unsigned. If Windows blocks it, click "More info" → "Run anyway", or right-click the exe → Properties → Unblock.

**Crash logs:** Stored at `%APPDATA%\WebScrapperDesktop\crash_log.txt` (Windows) or `~/Library/Application Support/WebScrapperDesktop/crash_log.txt` (Mac).
