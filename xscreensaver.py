#!/usr/bin/env python3

import argparse
import os
import random
import select
import signal
import shutil
import subprocess
import sys
import termios
import time
import tty
from pathlib import Path


DEFAULT_X = 2823
DEFAULT_Y = 268
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 128
DEFAULT_TITLE = "XScreenSaver Window"
POLL_INTERVAL_MS = 250
DEFAULT_SEARCH_DIRS = (
    "/usr/libexec/xscreensaver",
    "/usr/lib/xscreensaver",
    "/usr/local/libexec/xscreensaver",
    "/usr/local/lib/xscreensaver",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch an XScreenSaver hack inside a fixed on-screen window.",
    )
    parser.add_argument("hack", nargs="?", help="Hack name or absolute path. Defaults to the first discoverable hack.")
    parser.add_argument("--x", type=int, default=DEFAULT_X, help="Window X position in pixels.")
    parser.add_argument("--y", type=int, default=DEFAULT_Y, help="Window Y position in pixels.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Window width in pixels.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Window height in pixels.")
    parser.add_argument(
        "--title",
        default=DEFAULT_TITLE,
        help="Window title when decorations are enabled.",
    )
    parser.add_argument(
        "--decorated",
        action="store_true",
        help="Keep the normal window manager frame instead of a borderless window.",
    )
    parser.add_argument(
        "--topmost",
        action="store_true",
        help="Request that the window stays above other windows.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List discoverable XScreenSaver hacks and exit.",
    )
    parser.add_argument(
        "hack_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to the hack. Use -- before them.",
    )
    args = parser.parse_args()
    if args.hack_args and args.hack_args[0] == "--":
        args.hack_args = args.hack_args[1:]
    return args


def iter_search_dirs() -> list[Path]:
    dirs: list[Path] = []
    for raw in DEFAULT_SEARCH_DIRS:
        path = Path(raw)
        if path.is_dir():
            dirs.append(path)
    return dirs


def is_screensaver_hack_name(name: str) -> bool:
    return not name.startswith("xscreensaver-")


def list_hack_entries() -> list[tuple[str, str]]:
    hack_paths: dict[str, str] = {}
    for directory in iter_search_dirs():
        for entry in directory.iterdir():
            if entry.is_file() and os.access(entry, os.X_OK) and is_screensaver_hack_name(entry.name):
                hack_paths.setdefault(entry.name, str(entry))
    return sorted(hack_paths.items())


def list_hacks() -> list[str]:
    return [name for name, _path in list_hack_entries()]


def format_hacks_in_columns(hacks: list[str]) -> str:
    if not hacks:
        return ""

    terminal_width = shutil.get_terminal_size(fallback=(100, 24)).columns
    max_name_width = max(len(name) for name in hacks)
    min_gap = 2

    columns = max(1, terminal_width // (max_name_width + min_gap))
    rows = (len(hacks) + columns - 1) // columns
    columns = (len(hacks) + rows - 1) // rows

    col_widths: list[int] = []
    for col in range(columns):
        start = col * rows
        end = min(start + rows, len(hacks))
        col_widths.append(max(len(name) for name in hacks[start:end]))

    lines: list[str] = []
    for row in range(rows):
        parts: list[str] = []
        for col in range(columns):
            index = col * rows + row
            if index >= len(hacks):
                continue
            name = hacks[index]
            if col == columns - 1:
                parts.append(name)
            else:
                parts.append(name.ljust(col_widths[col] + min_gap))
        lines.append("".join(parts).rstrip())
    return "\n".join(lines)


def resolve_hack(hack: str) -> str:
    if os.sep in hack or hack.startswith("."):
        candidate = Path(hack).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            if not is_screensaver_hack_name(candidate.name):
                raise SystemExit(f"'{candidate.name}' is an xscreensaver helper, not a screensaver hack")
            return str(candidate)
        raise SystemExit(f"hack not found or not executable: {hack}")

    direct = shutil.which(hack)
    if direct:
        if not is_screensaver_hack_name(Path(direct).name):
            raise SystemExit(f"'{Path(direct).name}' is an xscreensaver helper, not a screensaver hack")
        return direct

    for name, path in list_hack_entries():
        if name == hack:
            return path

    available = list_hacks()
    suffix = f"\nknown hacks: {', '.join(available[:20])}" if available else ""
    raise SystemExit(f"could not find hack '{hack}' in PATH or standard xscreensaver directories.{suffix}")


def select_hacks(args: argparse.Namespace) -> tuple[list[str], list[str], int]:
    entries = list_hack_entries()
    if not entries:
        raise SystemExit("no xscreensaver hacks found in the standard directories.")

    names = [name for name, _path in entries]
    paths = [path for _name, path in entries]

    if not args.hack:
        return names, paths, 0

    if os.sep not in args.hack and not args.hack.startswith("."):
        try:
            return names, paths, names.index(args.hack)
        except ValueError:
            pass

    resolved = resolve_hack(args.hack)
    try:
        return names, paths, paths.index(resolved)
    except ValueError:
        return [Path(resolved).name], [resolved], 0


class XScreenSaverWindow:
    def __init__(
        self,
        tk_module,
        args: argparse.Namespace,
        hack_names: list[str],
        hack_paths: list[str],
        current_index: int,
    ) -> None:
        self.tk = tk_module
        self.args = args
        self.hack_names = hack_names
        self.hack_paths = hack_paths
        self.current_index = current_index
        self.proc: subprocess.Popen[str] | None = None
        self.proc_group_id: int | None = None
        self.closed = False
        self.poll_after_id: str | None = None
        self.stdin_fd: int | None = None
        self.stdin_attrs: list[int] | None = None
        self.terminal_poll_after_id: str | None = None

        self.root = self.tk.Tk()
        self.root.geometry(f"{args.width}x{args.height}+{args.x}+{args.y}")
        self.root.resizable(False, False)
        self.root.configure(bg="black")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        if not args.decorated:
            self.root.overrideredirect(True)
        if args.topmost:
            self.root.attributes("-topmost", True)

        self.root.bind("<Escape>", self.close)
        self.root.bind("n", self.next_hack)
        self.root.bind("r", self.random_hack)
        self.root.bind("q", self.close)

        self.container = self.tk.Frame(self.root, bg="black", highlightthickness=0, bd=0)
        self.container.pack(fill="both", expand=True)
        self._update_title()
        self._setup_terminal_controls()

    def current_hack_name(self) -> str:
        return self.hack_names[self.current_index]

    def current_hack_path(self) -> str:
        return self.hack_paths[self.current_index]

    def _update_title(self) -> None:
        self.root.title(f"{self.args.title}: {self.current_hack_name()}")

    def launch_hack(self) -> None:
        self.root.update_idletasks()
        self.root.update()

        window_id = self.container.winfo_id()
        window_id_hex = f"0x{window_id:x}"

        env = os.environ.copy()
        env["XSCREENSAVER_WINDOW"] = window_id_hex

        hack_path = self.current_hack_path()
        command = [hack_path, "-window-id", window_id_hex, *self.args.hack_args]
        try:
            self.proc = subprocess.Popen(command, env=env, start_new_session=True)
            self.proc_group_id = self.proc.pid
        except OSError as exc:
            self._restore_terminal_controls()
            self.root.destroy()
            raise SystemExit(f"failed to start hack '{hack_path}': {exc}") from exc

        print(f"Showing screensaver: {self.current_hack_name()}", file=sys.stderr)
        self.poll_after_id = self.root.after(POLL_INTERVAL_MS, self.poll_child)

    def poll_child(self) -> None:
        self.poll_after_id = None
        if self.closed:
            return
        assert self.proc is not None
        rc = self.proc.poll()
        if rc is not None:
            failed_hack = self.current_hack_name()
            self.proc = None
            self.proc_group_id = None
            if rc != 0 and len(self.hack_paths) > 1:
                print(f"{failed_hack} exited with status {rc}; cycling to next screensaver", file=sys.stderr)
                self.current_index = (self.current_index + 1) % len(self.hack_paths)
                self._update_title()
                self.launch_hack()
                return

            self.closed = True
            self._restore_terminal_controls()
            self.root.destroy()
            if rc != 0:
                print(f"{failed_hack} exited with status {rc}", file=sys.stderr)
            return
        self.poll_after_id = self.root.after(POLL_INTERVAL_MS, self.poll_child)

    def _signal_current_hack_group(self, sig: signal.Signals) -> bool:
        if self.proc_group_id is None:
            return False
        try:
            os.killpg(self.proc_group_id, sig)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return False

    def _current_hack_group_exists(self) -> bool:
        if self.proc_group_id is None:
            return False
        try:
            os.killpg(self.proc_group_id, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _stop_current_hack(self) -> None:
        if self.poll_after_id is not None:
            try:
                self.root.after_cancel(self.poll_after_id)
            except Exception:
                pass
            self.poll_after_id = None

        self._signal_current_hack_group(signal.SIGTERM)

        if self.proc is not None:
            try:
                self.proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._signal_current_hack_group(signal.SIGKILL)
                try:
                    self.proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass

        if self._current_hack_group_exists():
            time.sleep(0.1)
            if self._current_hack_group_exists():
                self._signal_current_hack_group(signal.SIGKILL)

        self.proc = None
        self.proc_group_id = None

    def next_hack(self, _event=None) -> None:
        if self.closed or not self.hack_paths:
            return
        self._stop_current_hack()
        self.current_index = (self.current_index + 1) % len(self.hack_paths)
        self._update_title()
        self.launch_hack()

    def random_hack(self, _event=None) -> None:
        if self.closed or not self.hack_paths:
            return
        self._stop_current_hack()
        if len(self.hack_paths) > 1:
            offset = random.randrange(1, len(self.hack_paths))
            self.current_index = (self.current_index + offset) % len(self.hack_paths)
        self._update_title()
        self.launch_hack()

    def _setup_terminal_controls(self) -> None:
        if not sys.stdin.isatty():
            return
        try:
            self.stdin_fd = sys.stdin.fileno()
            self.stdin_attrs = termios.tcgetattr(self.stdin_fd)
            tty.setcbreak(self.stdin_fd)
        except (OSError, ValueError, termios.error):
            self.stdin_fd = None
            self.stdin_attrs = None
            return

        if hasattr(self.root, "createfilehandler"):
            self.root.createfilehandler(sys.stdin, self.tk.READABLE, self._handle_terminal_ready)
        else:
            self.terminal_poll_after_id = self.root.after(POLL_INTERVAL_MS, self._poll_terminal_input)

        print("Terminal controls: n=next screensaver, r=random screensaver, q=quit", file=sys.stderr)

    def _restore_terminal_controls(self) -> None:
        if self.terminal_poll_after_id is not None:
            try:
                self.root.after_cancel(self.terminal_poll_after_id)
            except Exception:
                pass
            self.terminal_poll_after_id = None

        if hasattr(self.root, "deletefilehandler"):
            try:
                self.root.deletefilehandler(sys.stdin)
            except (AttributeError, OSError, ValueError):
                pass

        if self.stdin_fd is not None and self.stdin_attrs is not None:
            try:
                termios.tcsetattr(self.stdin_fd, termios.TCSADRAIN, self.stdin_attrs)
            except termios.error:
                pass
            self.stdin_attrs = None

    def _handle_terminal_ready(self, _file, _mask) -> None:
        self._drain_terminal_input()

    def _poll_terminal_input(self) -> None:
        self.terminal_poll_after_id = None
        if self.closed:
            return
        self._drain_terminal_input()
        self.terminal_poll_after_id = self.root.after(POLL_INTERVAL_MS, self._poll_terminal_input)

    def _drain_terminal_input(self) -> None:
        if self.stdin_fd is None:
            return

        while True:
            ready, _, _ = select.select([self.stdin_fd], [], [], 0)
            if not ready:
                return

            try:
                raw = os.read(self.stdin_fd, 32)
            except OSError:
                return
            if not raw:
                return

            for char in raw.decode(errors="ignore"):
                if char.lower() == "n":
                    self.next_hack()
                elif char.lower() == "r":
                    self.random_hack()
                elif char.lower() == "q":
                    self.close()
                    return

    def close(self, _event=None) -> None:
        if self.closed:
            return
        self.closed = True
        self._restore_terminal_controls()
        self._stop_current_hack()
        self.root.destroy()

    def run(self) -> None:
        self.launch_hack()
        self.root.mainloop()


def main() -> None:
    args = parse_args()

    if args.list:
        hacks = list_hacks()
        if not hacks:
            raise SystemExit("no xscreensaver hacks found in the standard directories.")
        print(format_hacks_in_columns(hacks))
        return

    if "DISPLAY" not in os.environ:
        raise SystemExit("DISPLAY is not set.")

    try:
        import tkinter as tk
    except ModuleNotFoundError as exc:
        raise SystemExit("tkinter is not installed for this Python interpreter.") from exc

    hack_names, hack_paths, current_index = select_hacks(args)
    XScreenSaverWindow(tk, args, hack_names, hack_paths, current_index).run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
