# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules


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


a = Analysis(
    [str(desktop_dir / "launcher.py")],
    pathex=[str(project_root), str(desktop_dir)],
    binaries=[],
    datas=[
        (str(project_root / "python-scraper.py"), "."),
    ]
    + ([(str(project_root / ".streamlit" / "config.toml"), ".streamlit")] if (project_root / ".streamlit" / "config.toml").exists() else [])
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
