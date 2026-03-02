# Project structure

This document describes the repo layout for **local runs**, **Streamlit Cloud**, and **Git**.

---

## App root (run from here)

These files must stay at the **repository root** so Streamlit Cloud and local runs work:

| File / folder        | Purpose |
|----------------------|--------|
| `python-scraper.py`  | Main Streamlit app. Streamlit Cloud runs this file. |
| `requirements.txt`   | Python dependencies. Used by `pip install -r requirements.txt` and Streamlit Cloud. |
| `run.bat`            | Windows launcher: creates venv, installs deps, runs Streamlit. **Run from this folder.** |
| `run.sh`             | Mac/Linux launcher: same as above. **Run from this folder.** |
| `.streamlit/config.toml` | Streamlit settings (theme, server, upload size). Committed; do not put secrets here. |
| `.streamlit/secrets.toml` | API keys (optional). **Gitignored** — create locally or in Streamlit Cloud secrets. |
| `runtime.txt`        | Optional; used by some hosts (e.g. Streamlit) to set Python version. |
| `outputs/`           | Generated run folders and checkpoints. **Gitignored.** |

**Local run:** Open a terminal in this folder, then:

- Windows: `run.bat`
- Mac/Linux: `chmod +x run.sh` then `./run.sh`

Or manually: `pip install -r requirements.txt` then `streamlit run python-scraper.py`.

**Streamlit Cloud:** Point the app to this repo; main file = `python-scraper.py`, root = repo root.

---

## Other folders

| Path              | Purpose |
|-------------------|--------|
| `WebScrapper/`     | Fallback copies of `run.bat` and `run.sh`. They switch to the parent directory if needed. See `WebScrapper/README.md`. |
| `desktop_app/`     | PyInstaller build for a standalone desktop executable. Not required for web or local Streamlit runs. |
| `landing-page/`    | Static landing page (e.g. for Vercel). |
| `.github/`         | CI/CD (e.g. deploy landing page). |

---

## Git

- **Committed:** All app code, `requirements.txt`, `.streamlit/config.toml`, `run.bat`, `run.sh`, docs, `desktop_app` and `landing-page` source.
- **Ignored (see `.gitignore`):** `venv/`, `.venv*`, `outputs/`, `*.csv`, `*.xlsx`, `*.zip`, `.streamlit/secrets.toml`, `build/`, `dist/`, IDE/OS files.

---

## One-line checklist

- **Local:** Run `run.bat` or `run.sh` from the folder that contains `python-scraper.py` and `requirements.txt`.
- **Streamlit Cloud:** Repo root = app root; main file = `python-scraper.py`.
- **Git:** Don’t commit `venv`, `outputs`, or `secrets.toml`.
