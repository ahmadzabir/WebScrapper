# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for WebScrapper Desktop.
Bundles Streamlit + dependencies. Metadata for packages using importlib.metadata
must be included or the frozen exe will crash with PackageNotFoundError.
"""

from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, copy_metadata


project_root = Path.cwd()
desktop_dir = project_root / "desktop_app"

hiddenimports = []
# Exclude optional langchain integration to avoid PyInstaller warning when langchain is not installed
hiddenimports += [m for m in collect_submodules("streamlit") if "streamlit.external.langchain" not in m]
hiddenimports += [
    "aiohttp",
    "pandas",
    "openpyxl",
    "bs4",
    "trafilatura",
    "lxml",
    "html5lib",
    "google.generativeai",
    "openai",
]


def _collect_metadata(*packages: str) -> list:
    """Include dist-info for packages that use importlib.metadata at runtime."""
    result = []
    for pkg in packages:
        try:
            result += copy_metadata(pkg)
        except Exception:
            pass
    return result


# Packages that use importlib.metadata.version() at import time; missing metadata = crash
_metadata_packages = [
    "streamlit",   # Required: launcher imports streamlit first
    "altair",      # Streamlit charts
    "pandas",      # Data handling
    "pillow",      # Images
    "numpy",       # Often used with metadata
]
_metadata_datas = _collect_metadata(*_metadata_packages)


a = Analysis(
    [str(desktop_dir / "launcher.py")],
    pathex=[str(project_root), str(desktop_dir)],
    binaries=[],
    datas=[
        (str(project_root / "python-scraper.py"), "."),
    ]
    + ([(str(project_root / ".streamlit" / "config.toml"), ".streamlit")] if (project_root / ".streamlit" / "config.toml").exists() else [])
    + _metadata_datas
    + [(str(project_root / "VERSION"), ".") if (project_root / "VERSION").exists() else (str(desktop_dir / "VERSION.default"), "VERSION")],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["langchain", "streamlit.external.langchain"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="WebScrapperDesktop",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
