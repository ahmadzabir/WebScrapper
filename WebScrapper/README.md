# Launcher scripts (fallback)

This folder contains **run.bat** and **run.sh** as a fallback if you opened the repo from this subfolder.

**Recommended:** Run the app from the **parent folder** (the repo root), where `python-scraper.py` and `requirements.txt` live:

- **Windows:** Use the **run.bat** in the parent folder, or double-click it from File Explorer.
- **Mac / Linux:** In the parent folder run: `chmod +x run.sh` then `./run.sh`.

The scripts in *this* folder will automatically switch to the parent directory if they don’t find `requirements.txt` here, so they work from either location.
