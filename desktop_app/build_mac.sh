#!/usr/bin/env bash
set -e

VERSION="${1:-1.0.0}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_PATH="$PROJECT_ROOT/.venv-desktop-mac"
DIST_PATH="$PROJECT_ROOT/dist"

echo "==> Project root: $PROJECT_ROOT"

# Create venv if needed
if [ ! -d "$VENV_PATH" ]; then
    echo "==> Creating virtual environment"
    python3 -m venv "$VENV_PATH"
fi

source "$VENV_PATH/bin/activate"

echo "==> Installing dependencies"
pip install --upgrade pip wheel setuptools
pip install -r "$PROJECT_ROOT/requirements.txt"
pip install pyinstaller

# Write VERSION
echo -n "$VERSION" > "$PROJECT_ROOT/VERSION"

echo "==> Building macOS app"
cd "$PROJECT_ROOT"
pyinstaller --noconfirm --clean desktop_app/WebScrapperDesktop.spec

APP_PATH="$DIST_PATH/WebScrapperDesktop"
if [ ! -f "$APP_PATH" ]; then
    echo "Build failed: executable not found at $APP_PATH"
    exit 1
fi

echo "==> Creating macOS zip"
cd "$DIST_PATH"
zip WebScrapperDesktop-macOS.zip WebScrapperDesktop

echo ""
echo "Build completed."
echo "App: $APP_PATH"
echo "Zip: $DIST_PATH/WebScrapperDesktop-macOS.zip"
