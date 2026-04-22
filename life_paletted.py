#!/usr/bin/env python3

import argparse
import math
import os
import random
import select
import sys
import termios
import tty

from fluere_led_window import (
    DEFAULT_IMAGE_DURATION_MS,
    DEFAULT_NUM_KNOTS,
    PALETTE_CYCLE_STEP,
    PALETTE_SIZE,
    STYLE_NAMES,
    FluereDrawing,
    get_color_table,
    load_palettes,
)


WINDOW_WIDTH = 512
WINDOW_HEIGHT = 128
DEFAULT_INTERVAL_MS = 5
DEFAULT_DECAY_STEP = 1
DEFAULT_DENSITY = 0.10
CHANNEL_RED = 1
CHANNEL_GREEN = 2
CHANNEL_BLUE = 4
PRIMARY_CHANNELS = (CHANNEL_RED, CHANNEL_GREEN, CHANNEL_BLUE)
RGB_WORLD_MASKS = PRIMARY_CHANNELS
WORLD_MASKS = (
    CHANNEL_RED,
    CHANNEL_GREEN,
    CHANNEL_BLUE,
    CHANNEL_RED | CHANNEL_GREEN,
    CHANNEL_RED | CHANNEL_BLUE,
    CHANNEL_GREEN | CHANNEL_BLUE,
)
WORMS_PER_COLOR = 4


class PhosphorLife:
    def __init__(self, width: int, height: int, density: float, decay_step: int, seed: int | None) -> None:
        self.width = width
        self.height = height
        self.density = min(1.0, max(0.0, density))
        self.decay_step = min(255, max(0, decay_step))
        self.random = random.Random(seed)

        self.prev_rows = [height - 1, *range(height - 1)]
        self.next_rows = [*range(1, height), 0]
        self.left_cols = [width - 1, *range(width - 1)]
        self.right_cols = [(index + 2) % width for index in range(width)]

        self.cells = [bytearray(width) for _ in range(height)]
        self.brightness = [bytearray(width) for _ in range(height)]
        self._seed_board()

    def is_alive(self, x: int, y: int) -> bool:
        return bool(self.cells[y][x])

    def clear_cell(self, x: int, y: int, clear_brightness: bool = False) -> None:
        self.cells[y][x] = 0
        if clear_brightness:
            self.brightness[y][x] = 0

    def seed_cell(self, x: int, y: int) -> None:
        self.cells[y][x] = 1
        self.brightness[y][x] = 255

    def write_pixel(self, x: int, y: int, level: int = 255) -> None:
        if level > self.brightness[y][x]:
            self.brightness[y][x] = level

    def _seed_board(self) -> None:
        if self.density <= 0.0:
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            self.cells[y][x] = 1
            self.brightness[y][x] = 255
            return

        live_count = 0
        while live_count == 0:
            live_count = 0
            for row, glow in zip(self.cells, self.brightness):
                for index in range(self.width):
                    alive = 1 if self.random.random() < self.density else 0
                    row[index] = alive
                    glow[index] = 255 if alive else 0
                    live_count += alive

    def next_frame(self) -> list[bytearray]:
        next_cells = [bytearray(self.width) for _ in range(self.height)]
        live_count = 0

        for y in range(self.height):
            above = self.cells[self.prev_rows[y]]
            current = self.cells[y]
            below = self.cells[self.next_rows[y]]
            next_row = next_cells[y]
            glow = self.brightness[y]

            block_sum = (
                above[-1] + above[0] + above[1]
                + current[-1] + current[0] + current[1]
                + below[-1] + below[0] + below[1]
            )

            for x in range(self.width):
                alive = current[x]
                neighbors = block_sum - alive
                next_alive = 1 if neighbors == 3 or (alive and neighbors == 2) else 0
                next_row[x] = next_alive

                if next_alive:
                    glow[x] = 255
                    live_count += 1
                elif glow[x]:
                    glow[x] = max(0, glow[x] - self.decay_step)

                left = self.left_cols[x]
                right = self.right_cols[x]
                block_sum += above[right] + current[right] + below[right]
                block_sum -= above[left] + current[left] + below[left]

        self.cells = next_cells
        if live_count == 0:
            self._seed_board()

        return self.brightness


class Snake:
    SPEED = 1.0
    STRAIGHT_CHANCE = 0.65
    MAX_TURN_RADIANS = math.pi / 3

    def __init__(self, width: int, height: int, seed: int | None) -> None:
        self.width = width
        self.height = height
        self.random = random.Random(seed)
        self.position = (0.0, 0.0)
        self.heading = self.random.uniform(0.0, math.tau)
        self._respawn()

    def _respawn(self) -> None:
        x = self.random.randrange(self.width)
        y = self.random.randrange(self.height)
        self.position = (x + 0.5, y + 0.5)
        self.heading = self.random.uniform(0.0, math.tau)

    def _cell_position(self) -> tuple[int, int]:
        x, y = self.position
        return int(x) % self.width, int(y) % self.height

    def _choose_move(self) -> tuple[float, float, int, int]:
        head_x, head_y = self.position
        if self.random.random() >= self.STRAIGHT_CHANCE:
            self.heading = (
                self.heading + self.random.uniform(-self.MAX_TURN_RADIANS, self.MAX_TURN_RADIANS)
            ) % math.tau
        next_x = (head_x + math.cos(self.heading) * self.SPEED) % self.width
        next_y = (head_y + math.sin(self.heading) * self.SPEED) % self.height
        return next_x, next_y, int(next_x) % self.width, int(next_y) % self.height

    def step(
        self,
        worlds: list[tuple[int, PhosphorLife]],
        own_world: PhosphorLife,
    ) -> None:
        next_pos_x, next_pos_y, next_x, next_y = self._choose_move()

        foreign_worlds = [
            world
            for _, world in worlds
            if world is not own_world and world.is_alive(next_x, next_y)
        ]
        if foreign_worlds:
            for world in foreign_worlds:
                world.clear_cell(next_x, next_y, clear_brightness=True)
            own_world.seed_cell(next_x, next_y)

        self.position = (next_pos_x, next_pos_y)

    def draw(self, own_world: PhosphorLife) -> None:
        x, y = self._cell_position()
        own_world.write_pixel(x, y)


class PanelApp:
    def __init__(self, tk_module, args: argparse.Namespace) -> None:
        self.tk = tk_module
        self.interval_ms = args.interval_ms
        self.off_color = "#000000"
        self.random = random.Random(args.seed)
        self.fluere_enabled = args.fluere
        self.closed = False
        self.stdin_fd: int | None = None
        self.stdin_attrs: list[int] | None = None
        self.world_masks = RGB_WORLD_MASKS if args.rgb else WORLD_MASKS
        self.palettes = load_palettes(None)
        self.worlds = [
            (
                mask,
                PhosphorLife(
                    WINDOW_WIDTH,
                    WINDOW_HEIGHT,
                    args.density,
                    args.decay_step,
                    self._plane_seed(args.seed, index),
                ),
            )
            for index, mask in enumerate(self.world_masks)
        ]
        self.world_by_mask = {
            mask: world
            for mask, world in self.worlds
        }
        self.world_color_tables: list[bytes] = []
        self.world_palette_offsets: list[int] = []
        self.world_palette_steps: list[int] = []
        self._randomize_world_palettes()
        self.fluere_image_data = b""
        self.fluere_color_table = b""
        self.fluere_palette_offset = 0
        self.fluere_palette_step = PALETTE_CYCLE_STEP
        self.fluere_frame_counter = 0
        self.fluere_frames_per_scene = max(1, DEFAULT_IMAGE_DURATION_MS // max(1, self.interval_ms))
        if self.fluere_enabled:
            self._new_fluere_scene()
        self.worms = [
            (
                world_mask,
                Snake(
                    WINDOW_WIDTH,
                    WINDOW_HEIGHT,
                    self._plane_seed(args.seed, 100 + world_index * WORMS_PER_COLOR + worm_index),
                ),
            )
            for world_index, world_mask in enumerate(self.world_masks)
            for worm_index in range(WORMS_PER_COLOR)
        ]

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
        self.root.bind("p", self.next_palette)
        self.root.bind("q", self.quit)

        self.ppm_header = f"P6 {WINDOW_WIDTH} {WINDOW_HEIGHT} 255\n".encode("ascii")
        self.pixel_count = WINDOW_WIDTH * WINDOW_HEIGHT
        self.ppm_buffer = bytearray(len(self.ppm_header) + self.pixel_count * 3)
        self.ppm_buffer[: len(self.ppm_header)] = self.ppm_header
        self.rgb_buffer = memoryview(self.ppm_buffer)[len(self.ppm_header) :]

        self.image_label = self.tk.Label(self.root, bd=0, highlightthickness=0, bg=self.off_color)
        self.image_label.pack()
        self.photo = None
        self._setup_terminal_controls()

    def _plane_seed(self, base_seed: int | None, offset: int) -> int | None:
        if base_seed is None:
            return None
        return base_seed + offset

    def _build_color_table(self, stripes: bool) -> tuple[bytes, int, int]:
        palette_rng = random.Random(self.random.randrange(1 << 30))
        palette = self.palettes[self.random.randrange(len(self.palettes))]
        color_table = get_color_table(
            palette=palette,
            rng=palette_rng,
            randomize=bool(palette_rng.randrange(2)),
            stripes=stripes,
        )
        palette_offset = palette_rng.randrange(PALETTE_SIZE)
        cycle_step = 1 + palette_rng.randrange(3)
        if not palette_rng.randrange(2):
            cycle_step *= -1
        return color_table, palette_offset, cycle_step

    def _randomize_world_palettes(self) -> None:
        self.world_color_tables.clear()
        self.world_palette_offsets.clear()
        self.world_palette_steps.clear()

        for _ in self.world_masks:
            color_table, palette_offset, cycle_step = self._build_color_table(stripes=False)
            self.world_color_tables.append(color_table)
            self.world_palette_offsets.append(palette_offset)
            self.world_palette_steps.append(cycle_step)

    def _choose_fluere_style(self) -> int:
        return self.random.randrange(len(STYLE_NAMES))

    def _build_fluere_image(self) -> bytes:
        drawing = FluereDrawing(
            width=WINDOW_WIDTH,
            height=WINDOW_HEIGHT,
            num_knots=DEFAULT_NUM_KNOTS,
            style1=self._choose_fluere_style(),
            style2=self._choose_fluere_style(),
            checkerboard=True,
            rng=random.Random(self.random.randrange(1 << 30)),
        )
        return drawing.fill_pixels()

    def _randomize_fluere_palette(self) -> None:
        self.fluere_color_table, self.fluere_palette_offset, _ = self._build_color_table(
            stripes=bool(self.random.randrange(2))
        )
        self.fluere_palette_step = PALETTE_CYCLE_STEP

    def _new_fluere_scene(self) -> None:
        self.fluere_image_data = self._build_fluere_image()
        self._randomize_fluere_palette()
        self.fluere_frame_counter = 0

    def _current_world_channel_tables(self) -> list[tuple[bytearray, bytearray, bytearray]]:
        channel_tables: list[tuple[bytearray, bytearray, bytearray]] = []
        for color_table, palette_offset in zip(self.world_color_tables, self.world_palette_offsets):
            red_table = bytearray(PALETTE_SIZE)
            green_table = bytearray(PALETTE_SIZE)
            blue_table = bytearray(PALETTE_SIZE)
            for level in range(1, PALETTE_SIZE):
                base = (palette_offset + (PALETTE_SIZE - 1 - level)) * 3
                red_table[level] = color_table[base] * level // 255
                green_table[level] = color_table[base + 1] * level // 255
                blue_table[level] = color_table[base + 2] * level // 255
            channel_tables.append((red_table, green_table, blue_table))
        return channel_tables

    def _current_fluere_channel_tables(self) -> tuple[bytes, bytes, bytes]:
        start = self.fluere_palette_offset * 3
        stop = (self.fluere_palette_offset + PALETTE_SIZE) * 3
        segment = self.fluere_color_table[start:stop]
        return bytes(segment[0::3]), bytes(segment[1::3]), bytes(segment[2::3])

    def _advance_fluere(self) -> None:
        if not self.fluere_enabled:
            return

        self.fluere_palette_offset = (self.fluere_palette_offset + self.fluere_palette_step) % PALETTE_SIZE
        self.fluere_frame_counter += 1
        if self.fluere_frame_counter >= self.fluere_frames_per_scene:
            self._new_fluere_scene()

    def _frame_to_ppm(self, frames: list[list[bytearray]]) -> bytes:
        channel_tables = self._current_world_channel_tables()
        if self.fluere_enabled:
            fluere_red_table, fluere_green_table, fluere_blue_table = self._current_fluere_channel_tables()
        else:
            fluere_red_table = None
            fluere_green_table = None
            fluere_blue_table = None
        pixel_offset = 0
        pixel_index = 0

        for rows in zip(*frames):
            row_tables = tuple(zip(rows, channel_tables))
            for x in range(WINDOW_WIDTH):
                red = 0
                green = 0
                blue = 0
                has_life = False
                for row, (red_table, green_table, blue_table) in row_tables:
                    level = row[x]
                    if not level:
                        continue
                    has_life = True
                    red = 255 - ((255 - red) * (255 - red_table[level]) // 255)
                    green = 255 - ((255 - green) * (255 - green_table[level]) // 255)
                    blue = 255 - ((255 - blue) * (255 - blue_table[level]) // 255)
                if not has_life and fluere_red_table is not None:
                    level = self.fluere_image_data[pixel_index]
                    red = fluere_red_table[level]
                    green = fluere_green_table[level]
                    blue = fluere_blue_table[level]
                self.rgb_buffer[pixel_offset] = red
                self.rgb_buffer[pixel_offset + 1] = green
                self.rgb_buffer[pixel_offset + 2] = blue
                pixel_offset += 3
                pixel_index += 1

        return bytes(self.ppm_buffer)

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

        print("Terminal controls: p=next palette, q=quit", file=sys.stderr)

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
                lowered = char.lower()
                if lowered == "p":
                    self.next_palette()
                elif lowered == "q":
                    self.quit()
                    return

    def render_frame(self) -> None:
        if self.closed:
            return
        for world_mask, worm in self.worms:
            worm.step(self.worlds, self.world_by_mask[world_mask])
        frames = [world.next_frame() for _, world in self.worlds]
        for world_mask, worm in self.worms:
            worm.draw(self.world_by_mask[world_mask])
        self.photo = self.tk.PhotoImage(data=self._frame_to_ppm(frames), format="PPM")
        self.image_label.configure(image=self.photo)
        for index, cycle_step in enumerate(self.world_palette_steps):
            self.world_palette_offsets[index] = (self.world_palette_offsets[index] + cycle_step) % PALETTE_SIZE
        self._advance_fluere()
        self.root.after(self.interval_ms, self.render_frame)

    def next_palette(self, _event=None) -> None:
        self._randomize_world_palettes()
        if self.fluere_enabled:
            self._randomize_fluere_palette()

    def quit(self, _event=None) -> None:
        if self.closed:
            return
        self.closed = True
        self._restore_terminal_controls()
        self.root.destroy()

    def run(self) -> None:
        self.render_frame()
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display Life worlds with phosphor decay, Fluere palettes, and four random worms per active world.",
    )
    parser.add_argument("--x", type=int, default=2823, help="Window X position in pixels.")
    parser.add_argument("--y", type=int, default=268, help="Window Y position in pixels.")
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=DEFAULT_INTERVAL_MS,
        help="Milliseconds between display updates.",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=DEFAULT_DENSITY,
        help="Initial live-cell density between 0.0 and 1.0.",
    )
    parser.add_argument(
        "--decay-step",
        type=int,
        default=DEFAULT_DECAY_STEP,
        help="Brightness removed from unwritten pixels each frame.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible restarts.",
    )
    parser.add_argument(
        "--rgb",
        action="store_true",
        help="Use only the red, green, and blue Life worlds.",
    )
    parser.add_argument(
        "--fluere",
        action="store_true",
        help="Render a separate Fluere animation behind pixels where all Life worlds are dark.",
    )
    parser.add_argument(
        "--title",
        default="Conway Life Palette Worms",
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
