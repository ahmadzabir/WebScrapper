import os
import socket
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

from streamlit.web import bootstrap

def _get_version() -> str:
    """Read embedded version from build or version.py."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)  # type: ignore[attr-defined]
        candidates = [base / "VERSION", base / "VERSION.default"]
    else:
        proj = Path(__file__).resolve().parents[1]
        candidates = [proj / "VERSION", proj / "desktop_app" / "VERSION", proj / "desktop_app" / "VERSION.default"]
    for vpath in candidates:
        if vpath.exists():
            return vpath.read_text().strip()
    try:
        from desktop_app.version import __version__
        return __version__
    except ImportError:
        return "0.0.0"

APP_VERSION = _get_version()

APP_TITLE = "WebScrapper Desktop"
READY_CHECK_MAX_ATTEMPTS = 90
READY_CHECK_INTERVAL_MS = 1000
GITHUB_RELEASES_API = "https://api.github.com/repos/ahmadzabir/WebScrapper/releases/latest"
GITHUB_RELEASES_PAGE = "https://github.com/ahmadzabir/WebScrapper/releases/latest"


def resource_path(relative_path: str) -> Path:
    """Resolve resources for both source and PyInstaller builds."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / relative_path  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[1] / relative_path


def find_free_port(start_port: int = 8501, max_tries: int = 50) -> int:
    """Find an available localhost port."""
    for port in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("No free local port available for Streamlit server.")


def is_server_ready(port: int, timeout_sec: float = 1.0) -> bool:
    """Check whether localhost:port accepts TCP connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_sec)
        return sock.connect_ex(("127.0.0.1", port)) == 0


class DesktopLauncher:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("640x360")
        self.root.minsize(640, 360)

        self.server_thread: threading.Thread | None = None
        self.server_port: int | None = None
        self.server_url: str | None = None
        self.started = False
        self.opened_once = False
        self.error_message: str | None = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=20)
        outer.pack(fill="both", expand=True)
        self._outer_frame = outer

        title = ttk.Label(outer, text=f"{APP_TITLE}  v{APP_VERSION}", font=("Segoe UI", 20, "bold"))
        title.pack(anchor="w")

        self.update_frame: ttk.Frame | None = None
        self.update_btn: ttk.Button | None = None

        subtitle = ttk.Label(
            outer,
            text="Run the scraper locally with one click. No browser setup needed.",
            font=("Segoe UI", 10),
        )
        subtitle.pack(anchor="w", pady=(4, 16))

        card = ttk.Frame(outer, padding=16)
        card.pack(fill="x")

        self.status_var = tk.StringVar(value="Ready. Click Start to launch the scraper.")
        self.status_label = ttk.Label(card, textvariable=self.status_var, font=("Segoe UI", 10))
        self.status_label.pack(anchor="w")

        self.detail_var = tk.StringVar(value="Server: not started")
        self.detail_label = ttk.Label(card, textvariable=self.detail_var, font=("Segoe UI", 9))
        self.detail_label.pack(anchor="w", pady=(8, 0))

        self.progress = ttk.Progressbar(card, mode="indeterminate")
        self.progress.pack(fill="x", pady=(12, 0))

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(20, 0))

        self.start_btn = ttk.Button(actions, text="Start App", command=self.start_app)
        self.start_btn.pack(side="left")

        self.open_btn = ttk.Button(actions, text="Open in Browser", command=self.open_browser, state="disabled")
        self.open_btn.pack(side="left", padx=(10, 0))

        self.exit_btn = ttk.Button(actions, text="Exit", command=self.on_close)
        self.exit_btn.pack(side="right")

        notes = ttk.Label(
            outer,
            text=(
                "Tip: Keep this launcher open while scraping. Closing it will stop the local server."
            ),
            font=("Segoe UI", 9),
        )
        notes.pack(anchor="w", pady=(20, 0))

        # Start update check in background
        threading.Thread(target=self._check_for_updates, daemon=True).start()

    def _check_for_updates(self) -> None:
        """Fetch latest release from GitHub; show update button if newer."""
        try:
            import urllib.request
            req = urllib.request.Request(GITHUB_RELEASES_API)
            req.add_header("Accept", "application/vnd.github.v3+json")
            with urllib.request.urlopen(req, timeout=5) as resp:
                import json
                data = json.loads(resp.read().decode())
            tag = data.get("tag_name", "")
            if tag.startswith("desktop-v"):
                latest = tag.replace("desktop-v", "").strip()
            else:
                latest = tag.lstrip("v").strip()
            if not latest:
                return
            # Simple semver compare: normalize and compare
            def parse(v):
                parts = v.replace("-", ".").split(".")
                return [int(p) if p.isdigit() else 0 for p in (parts + ["0", "0"])[:3]]
            cur = parse(APP_VERSION)
            new = parse(latest)
            if new > cur:
                self.root.after(0, lambda: self._show_update_available(latest))
        except Exception:
            pass

    def _show_update_available(self, version: str) -> None:
        """Show update-available UI."""
        if self.update_frame is not None:
            return
        self.update_frame = ttk.Frame(self._outer_frame)
        self.update_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(self.update_frame, text=f"Update available: v{version}", font=("Segoe UI", 9)).pack(side="left")
        self.update_btn = ttk.Button(
            self.update_frame,
            text="Download",
            command=lambda: webbrowser.open(GITHUB_RELEASES_PAGE, new=2),
        )
        self.update_btn.pack(side="left", padx=(8, 0))

    def start_app(self) -> None:
        if self.started:
            self.status_var.set("App is already running.")
            return

        app_script = resource_path("python-scraper.py")
        if not app_script.exists():
            messagebox.showerror("Missing File", f"Cannot find app file:\n{app_script}")
            return

        try:
            self.server_port = find_free_port()
        except RuntimeError as exc:
            messagebox.showerror("Port Error", str(exc))
            return

        self.server_url = f"http://127.0.0.1:{self.server_port}"
        self.error_message = None
        self.opened_once = False
        self.started = True

        self.start_btn.config(state="disabled")
        self.open_btn.config(state="disabled")
        self.progress.start(10)
        self.status_var.set("Starting local server...")
        self.detail_var.set(f"Server target: {self.server_url}")

        self.server_thread = threading.Thread(
            target=self._run_streamlit_server,
            args=(str(app_script),),
            daemon=True,
        )
        self.server_thread.start()

        self.root.after(READY_CHECK_INTERVAL_MS, self._poll_until_ready, 0)

    def _run_streamlit_server(self, script_path: str) -> None:
        try:
            os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")
            os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
            os.environ.setdefault("STREAMLIT_GLOBAL_DEVELOPMENT_MODE", "false")

            flag_options = {
                "server.headless": True,
                "server.port": self.server_port,
                "server.address": "127.0.0.1",
                "browser.gatherUsageStats": False,
            }
            bootstrap.run(script_path, "", [], flag_options)
        except Exception as exc:
            self.error_message = str(exc)
            self.started = False

    def _poll_until_ready(self, attempt: int) -> None:
        if self.server_port and is_server_ready(self.server_port):
            self.progress.stop()
            self.status_var.set("Running. Opened in your default browser.")
            self.detail_var.set(f"Server: {self.server_url}")
            self.open_btn.config(state="normal")
            if not self.opened_once:
                self.opened_once = True
                self.open_browser()
            return

        if self.error_message:
            self.progress.stop()
            self.start_btn.config(state="normal")
            self.status_var.set("Failed to start.")
            self.detail_var.set(f"Error: {self.error_message}")
            messagebox.showerror("Launch Failed", self.error_message)
            return

        if attempt >= READY_CHECK_MAX_ATTEMPTS:
            self.progress.stop()
            self.start_btn.config(state="normal")
            self.status_var.set("Startup timed out.")
            self.detail_var.set("The app did not become ready in time.")
            messagebox.showwarning(
                "Startup Timeout",
                "The server did not start in time. Please retry.",
            )
            self.started = False
            return

        self.status_var.set("Starting local server... please wait")
        self.root.after(READY_CHECK_INTERVAL_MS, self._poll_until_ready, attempt + 1)

    def open_browser(self) -> None:
        if not self.server_url:
            messagebox.showinfo("Not Running", "Start the app first.")
            return
        webbrowser.open(self.server_url, new=2)

    def on_close(self) -> None:
        if self.started:
            should_exit = messagebox.askyesno(
                "Exit",
                "Closing this launcher will stop the scraper server. Exit now?",
            )
            if not should_exit:
                return

        # Streamlit server runs in-process; exiting the launcher cleanly stops everything.
        self.root.destroy()
        time.sleep(0.2)
        os._exit(0)


def _log_crash(exc: BaseException) -> None:
    """Write crash log to user's app data or temp."""
    try:
        if sys.platform == "win32":
            base = os.environ.get("APPDATA", os.path.expanduser("~"))
        elif sys.platform == "darwin":
            base = os.path.expanduser("~/Library/Application Support")
        else:
            base = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
        log_dir = Path(base) / "WebScrapperDesktop"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "crash_log.txt"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            traceback.print_exc(file=f)
    except Exception:
        pass


def main() -> None:
    try:
        root = tk.Tk()
        DesktopLauncher(root)
        root.mainloop()
    except Exception as exc:
        _log_crash(exc)
        raise


if __name__ == "__main__":
    main()
