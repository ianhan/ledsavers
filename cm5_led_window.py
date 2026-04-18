#!/usr/bin/env python3

import argparse


WINDOW_WIDTH = 512
WINDOW_HEIGHT = 128
NUM_ROWS = 32
NUM_ROWS_DISPLAYED = WINDOW_HEIGHT
NUM_COLS = WINDOW_WIDTH
ROW_BYTES = NUM_COLS // 8
DEFAULT_INTERVAL_MS = 100
LFSR_WIDTH = 16
# Primitive polynomial: x^16 + x^12 + x^3 + x^1 + 1
LFSR_TAPS = (0, 4, 13, 15)
INITIAL_SEED = 1


def lfsr_feedback(value: int) -> int:
    feedback = 0
    for tap in LFSR_TAPS:
        feedback ^= (value >> tap) & 1
    return feedback


class PleasingMode7:
    def __init__(self) -> None:
        self.gen = INITIAL_SEED
        self.rows = [0] * NUM_ROWS
        self.lfsr_mask = (1 << LFSR_WIDTH) - 1
        self.row_mask = (1 << NUM_COLS) - 1

    def next_frame(self) -> list[int]:
        for index in range(NUM_ROWS):
            gen_bit = lfsr_feedback(self.gen)
            row_bit = ((self.gen >> 0) & (self.gen >> 1)) & 1
            self.gen = ((gen_bit << (LFSR_WIDTH - 1)) | (self.gen >> 1)) & self.lfsr_mask

            if index & 4:
                self.rows[index] = ((row_bit << (NUM_COLS - 1)) | (self.rows[index] >> 1)) & self.row_mask
            else:
                self.rows[index] = ((self.rows[index] << 1) | row_bit) & self.row_mask

        return [self.rows[index & 31] for index in range(NUM_ROWS_DISPLAYED)]


class PanelApp:
    def __init__(self, tk_module, args: argparse.Namespace) -> None:
        self.tk = tk_module
        self.interval_ms = args.interval_ms
        self.off_color = "#000000"
        self.algorithm = PleasingMode7()

        self.root = self.tk.Tk()
        self.root.title(args.title)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{args.x}+{args.y}")
        self.root.resizable(False, False)
        self.root.configure(bg=self.off_color)

        if not args.decorated:
            self.root.overrideredirect(True)
        if args.topmost:
            self.root.attributes("-topmost", True)

        self.root.bind("<Escape>", self.quit)
        self.root.bind("q", self.quit)

        self.on_pixel = self._color_to_rgb(args.color)
        self.off_pixel = b"\x00\x00\x00"
        self.ppm_header = f"P6 {WINDOW_WIDTH} {WINDOW_HEIGHT} 255\n".encode("ascii")
        self.byte_lookup = [self._expand_byte(value) for value in range(256)]

        self.image_label = self.tk.Label(self.root, bd=0, highlightthickness=0, bg=self.off_color)
        self.image_label.pack()
        self.photo = None

    def _color_to_rgb(self, color: str) -> bytes:
        red, green, blue = self.root.winfo_rgb(color)
        return bytes((red >> 8, green >> 8, blue >> 8))

    def _expand_byte(self, value: int) -> bytes:
        pixels = bytearray()
        for bit_index in range(7, -1, -1):
            pixels.extend(self.on_pixel if value & (1 << bit_index) else self.off_pixel)
        return bytes(pixels)

    def _frame_to_ppm(self, rows: list[int]) -> bytes:
        payload = bytearray(self.ppm_header)
        for row_value in rows:
            row_bytes = row_value.to_bytes(ROW_BYTES, "big")
            for value in row_bytes:
                payload.extend(self.byte_lookup[value])
        return bytes(payload)

    def render_frame(self) -> None:
        rows = self.algorithm.next_frame()
        self.photo = self.tk.PhotoImage(data=self._frame_to_ppm(rows), format="PPM")
        self.image_label.configure(image=self.photo)

        self.root.after(self.interval_ms, self.render_frame)

    def quit(self, _event=None) -> None:
        self.root.destroy()

    def run(self) -> None:
        self.render_frame()
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display the CM-5 random-and-pleasing panel animation in a 512x128 window.",
    )
    parser.add_argument("--x", type=int, default=2823, help="Window X position in pixels.")
    parser.add_argument("--y", type=int, default=268, help="Window Y position in pixels.")
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=DEFAULT_INTERVAL_MS,
        help="Milliseconds between animation updates.",
    )
    parser.add_argument(
        "--color",
        default="#ff0000",
        help="LED color for lit pixels. Default: red (#ff0000).",
    )
    parser.add_argument(
        "--title",
        default="CM-5 Pleasing Mode 7",
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

    try:
        import tkinter as tk
    except ModuleNotFoundError as exc:
        raise SystemExit("tkinter is not installed for this Python interpreter.") from exc

    PanelApp(tk, args).run()


if __name__ == "__main__":
    main()
