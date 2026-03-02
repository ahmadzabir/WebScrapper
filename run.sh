#!/usr/bin/env bash
cd "$(dirname "$0")"

if ! command -v python3 &>/dev/null && ! command -v python &>/dev/null; then
    echo "Python not found. Install Python 3.10+ from https://www.python.org/downloads/"
    exit 1
fi

PYTHON=python3
command -v python3 &>/dev/null || PYTHON=python

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    "$PYTHON" -m venv venv
fi
source venv/bin/activate

echo "Installing/updating dependencies..."
pip install -q -r requirements.txt

echo ""
echo "Starting Web Scraper..."
echo "Open the URL shown below in your browser (usually http://localhost:8501)"
echo ""
exec streamlit run python-scraper.py
