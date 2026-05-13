#!/usr/bin/env python3

import argparse
import os
import random
import select
import sys
import termios
import tty
from pathlib import Path


WINDOW_WIDTH = 512
WINDOW_HEIGHT = 128
DEFAULT_INTERVAL_MS = 100
DEFAULT_IMAGE_DIR = Path(__file__).resolve().parent / "astro"
IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".ppm",
    ".webp",
}


def find_image_paths(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise SystemExit(f"image directory not found: {image_dir}")

    paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise SystemExit(f"no images found in: {image_dir}")
    return paths


class PanelApp:
    def __init__(self, tk_module, args: argparse.Namespace, image_paths: list[Path]) -> None:
        self.tk = tk_module
        self.interval_ms = args.interval_ms
        self.off_color = "#000000"
        self.random = random.Random(args.seed)
        self.image_paths = image_paths
        self.current_image_path: Path | None = None
        self.closed = False
        self.stdin_fd: int | None = None
        self.stdin_attrs: list[int] | None = None

        self.root = self.tk.Tk()
        self.root.title(args.title)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{args.x}+{args.y}")
        self.root.resizable(False, False)
        self.root.configure(bg=self.off_color)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        if not args.decorated:
            self.root.overrideredirect(True)
        if args.topmost:
            self.root.attributes("-topmost", True)

        self.root.bind("<Escape>", self.quit)
        self.root.bind("q", self.quit)
        self.root.bind("r", self.random_image)

        self.ppm_header = f"P6 {WINDOW_WIDTH} {WINDOW_HEIGHT} 255\n".encode("ascii")
        self.image_label = self.tk.Label(self.root, bd=0, highlightthickness=0, bg=self.off_color)
        self.image_label.pack()
        self.photo = None
        self._setup_terminal_controls()

    def _tile_to_panel(self, source_image) -> bytes:
        from PIL import Image

        source = source_image.convert("RGB")
        panel = Image.new("RGB", (WINDOW_WIDTH, WINDOW_HEIGHT))
        source_width, source_height = source.size

        for y in range(0, WINDOW_HEIGHT, source_height):
            for x in range(0, WINDOW_WIDTH, source_width):
                panel.paste(source, (x, y))

        return self.ppm_header + panel.tobytes()

    def _load_panel_ppm(self, image_path: Path) -> bytes:
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(image_path) as image:
                return self._tile_to_panel(image)
        except UnidentifiedImageError as exc:
            raise SystemExit(f"could not read image: {image_path}") from exc

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
            self.root.after(self.interval_ms, self._poll_terminal_input)

        print("Terminal controls: q=quit r=random image", file=sys.stderr)

    def _restore_terminal_controls(self) -> None:
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
        if self.closed:
            return
        self._drain_terminal_input()
        self.root.after(self.interval_ms, self._poll_terminal_input)

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
                if char.lower() == "q":
                    self.quit()
                    return
                if char.lower() == "r":
                    self.random_image()

    def _choose_random_path(self) -> Path:
        if len(self.image_paths) == 1 or self.current_image_path is None:
            return self.random.choice(self.image_paths)

        choices = [path for path in self.image_paths if path != self.current_image_path]
        return self.random.choice(choices)

    def random_image(self, _event=None) -> None:
        if self.closed:
            return

        image_path = self._choose_random_path()
        self.current_image_path = image_path
        self.photo = self.tk.PhotoImage(data=self._load_panel_ppm(image_path), format="PPM")
        self.image_label.configure(image=self.photo)
        self.root.title(f"Bowling: {image_path.name}")
        print(f"Showing image: {image_path.name}", file=sys.stderr)

    def quit(self, _event=None) -> None:
        if self.closed:
            return
        self.closed = True
        self._restore_terminal_controls()
        self.root.destroy()

    def run(self) -> None:
        self.random_image()
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display random bowling images in a 512x128 window.",
    )
    parser.add_argument("--x", type=int, default=2823, help="Window X position in pixels.")
    parser.add_argument("--y", type=int, default=268, help="Window Y position in pixels.")
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=DEFAULT_INTERVAL_MS,
        help="Milliseconds between terminal input polls on platforms without Tk file handlers.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory of images to display. Default: astro next to this script.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for deterministic random image selection.",
    )
    parser.add_argument(
        "--title",
        default="Bowling",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = find_image_paths(args.image_dir.expanduser())

    try:
        import tkinter as tk
    except ModuleNotFoundError as exc:
        raise SystemExit("tkinter is not installed for this Python interpreter.") from exc

    try:
        import PIL  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit("Pillow is required to load the images. Install it with: python3 -m pip install Pillow") from exc

    PanelApp(tk, args, image_paths).run()


if __name__ == "__main__":
    main()
