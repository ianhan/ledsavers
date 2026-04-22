#!/usr/bin/env python3

import argparse
import http.server
import os
import shutil
import subprocess
import threading
import urllib.parse
from contextlib import contextmanager
from pathlib import Path


DEFAULT_X = 2823
DEFAULT_Y = 268
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 128
DEFAULT_SNAKES = 6
DEFAULT_SNAKE_LENGTH = 48
DEFAULT_SNAKE_THICKNESS = 3
DEFAULT_SNAKE_TRAIL_LIFETIME = 192
DEFAULT_SNAKE_CONSUME_SIZE = 3
DEFAULT_SNAKE_TURN_CHANCE = 0.18
DEFAULT_SNAKE_STEP_INTERVAL = 1
BROWSER_CANDIDATES = (
    "chromium-browser",
    "chromium",
    "google-chrome",
    "google-chrome-stable",
)


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, _format: str, *_args) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch the digital-ecosystem showcase as a GPU-backed LED panel screensaver "
            "in a fixed Chromium app window."
        )
    )
    parser.add_argument("--x", type=int, default=DEFAULT_X, help="Window X position in pixels.")
    parser.add_argument("--y", type=int, default=DEFAULT_Y, help="Window Y position in pixels.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Window width in pixels.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Window height in pixels.")
    parser.add_argument(
        "--browser",
        default=None,
        help="Explicit Chromium/Chrome executable path. Defaults to the first browser found in PATH.",
    )
    parser.add_argument(
        "--snake-count",
        type=int,
        default=DEFAULT_SNAKES,
        help="Number of autonomous snakes drawing temporary wall boundaries.",
    )
    parser.add_argument(
        "--snake-length",
        type=int,
        default=DEFAULT_SNAKE_LENGTH,
        help="Body length used for snake self-avoidance, in grid cells.",
    )
    parser.add_argument(
        "--snake-thickness",
        type=int,
        default=DEFAULT_SNAKE_THICKNESS,
        help="Wall thickness for each snake trail, in grid cells.",
    )
    parser.add_argument(
        "--snake-trail-lifetime",
        type=int,
        default=DEFAULT_SNAKE_TRAIL_LIFETIME,
        help="How many snake moves a wall trail persists before fading.",
    )
    parser.add_argument(
        "--snake-consume-size",
        type=int,
        default=DEFAULT_SNAKE_CONSUME_SIZE,
        help="Square brush size used when snakes consume life.",
    )
    parser.add_argument(
        "--snake-turn-chance",
        type=float,
        default=DEFAULT_SNAKE_TURN_CHANCE,
        help="Chance of turning each step when a straight move is available.",
    )
    parser.add_argument(
        "--snake-step-interval",
        type=int,
        default=DEFAULT_SNAKE_STEP_INTERVAL,
        help="Simulation steps between snake moves.",
    )
    parser.add_argument(
        "--browser-arg",
        action="append",
        default=[],
        help="Extra argument passed through to Chromium. Repeat to supply multiple flags.",
    )
    return parser.parse_args()


def resolve_browser(explicit: str | None) -> str:
    if explicit:
        browser = shutil.which(explicit)
        if browser:
            return browser
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            return str(candidate)
        raise SystemExit(f"browser not found: {explicit}")

    for candidate in BROWSER_CANDIDATES:
        browser = shutil.which(candidate)
        if browser:
            return browser

    raise SystemExit(
        "No Chromium/Chrome browser was found in PATH. "
        "Install one or pass --browser /path/to/chromium."
    )


def build_handler(directory: Path):
    class Handler(QuietHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

    return Handler


@contextmanager
def serve_directory(directory: Path):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), build_handler(directory))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)


def build_url(port: int, args: argparse.Namespace) -> str:
    query = urllib.parse.urlencode(
        {
            "screensaver": "1",
            "showcase": "1",
            "grid": f"{args.width}x{args.height}",
            "snakes": str(args.snake_count),
            "snakeLength": str(args.snake_length),
            "snakeThickness": str(args.snake_thickness),
            "snakeTrailLifetime": str(args.snake_trail_lifetime),
            "snakeConsumeSize": str(args.snake_consume_size),
            "snakeTurnChance": str(args.snake_turn_chance),
            "snakeStepInterval": str(args.snake_step_interval),
        }
    )
    return f"http://127.0.0.1:{port}/index.html?{query}"


def build_browser_command(browser: str, url: str, args: argparse.Namespace) -> list[str]:
    browser_args = list(args.browser_arg)
    if (
        os.environ.get("XDG_SESSION_TYPE") == "wayland"
        and not any(arg.startswith("--ozone-platform=") for arg in browser_args)
    ):
        # Native Wayland Chromium ignores absolute window geometry for this use case.
        browser_args.insert(0, "--ozone-platform=x11")

    command = [
        browser,
        f"--app={url}",
        f"--window-position={args.x},{args.y}",
        f"--window-size={args.width},{args.height}",
        "--disable-session-crashed-bubble",
        "--disable-infobars",
        "--disable-features=Translate",
        "--force-device-scale-factor=1",
        "--no-first-run",
        *browser_args,
    ]
    return command


def main() -> int:
    args = parse_args()
    app_dir = Path(__file__).resolve().parent / "digital-ecosystem"
    if not app_dir.is_dir():
        raise SystemExit(f"digital-ecosystem directory not found: {app_dir}")

    browser = resolve_browser(args.browser)
    with serve_directory(app_dir) as server:
        port = server.server_address[1]
        url = build_url(port, args)
        command = build_browser_command(browser, url, args)
        process = subprocess.Popen(command)
        try:
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                return process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
