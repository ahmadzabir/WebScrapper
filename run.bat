@echo off
title Web Scraper
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python not found. Install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat

echo Installing/updating dependencies...
pip install -q -r requirements.txt
python -m playwright install chromium 2>nul
if errorlevel 1 (
    echo Note: Playwright browser not installed. Run "playwright install chromium" for better Cloudflare/timeout handling.
) else (
    echo Playwright Chromium ready for Cloudflare bypass.
)

echo.
echo Starting Web Scraper...
echo Open the URL shown below in your browser (usually http://localhost:8501)
echo.
streamlit run python-scraper.py

pause
