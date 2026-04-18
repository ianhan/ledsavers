#!/usr/bin/env python3

import argparse
import random


WINDOW_WIDTH = 512
WINDOW_HEIGHT = 128
DEFAULT_INTERVAL_MS = 5
DEFAULT_DECAY_STEP = 1
DEFAULT_DENSITY = 0.18
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
    DIRECTIONS = (
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    )

    def __init__(self, width: int, height: int, seed: int | None) -> None:
        self.width = width
        self.height = height
        self.random = random.Random(seed)
        self.position = (0, 0)
        self.direction = self.random.choice(self.DIRECTIONS)
        self._respawn()

    def _respawn(self) -> None:
        x = self.random.randrange(self.width)
        y = self.random.randrange(self.height)
        self.position = (x, y)
        self.direction = self.random.choice(self.DIRECTIONS)

    def _choose_move(self) -> tuple[int, int, int, int]:
        head_x, head_y = self.position
        directions = list(self.DIRECTIONS)
        self.random.shuffle(directions)

        if self.direction in directions:
            directions.remove(self.direction)
            directions.insert(0, self.direction)

        if directions[0] == self.direction and self.random.random() < 0.65:
            dx, dy = directions[0]
        else:
            dx, dy = self.random.choice(directions)
        nx = (head_x + dx) % self.width
        ny = (head_y + dy) % self.height
        return dx, dy, nx, ny

    def step(
        self,
        worlds: list[tuple[int, PhosphorLife]],
        own_world: PhosphorLife,
    ) -> None:
        move = self._choose_move()
        dx, dy, next_x, next_y = move
        self.direction = (dx, dy)

        foreign_worlds = [
            world
            for _, world in worlds
            if world is not own_world and world.is_alive(next_x, next_y)
        ]
        if foreign_worlds:
            for world in foreign_worlds:
                world.clear_cell(next_x, next_y, clear_brightness=True)
            own_world.seed_cell(next_x, next_y)

        self.position = (next_x, next_y)

    def draw(self, own_world: PhosphorLife) -> None:
        x, y = self.position
        own_world.write_pixel(x, y)


class PanelApp:
    def __init__(self, tk_module, args: argparse.Namespace) -> None:
        self.tk = tk_module
        self.interval_ms = args.interval_ms
        self.off_color = "#000000"
        self.world_masks = RGB_WORLD_MASKS if args.rgb else WORLD_MASKS
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

        if not args.decorated:
            self.root.overrideredirect(True)
        if args.topmost:
            self.root.attributes("-topmost", True)

        self.root.bind("<Escape>", self.quit)
        self.root.bind("q", self.quit)

        self.ppm_header = f"P6 {WINDOW_WIDTH} {WINDOW_HEIGHT} 255\n".encode("ascii")

        self.image_label = self.tk.Label(self.root, bd=0, highlightthickness=0, bg=self.off_color)
        self.image_label.pack()
        self.photo = None

    def _plane_seed(self, base_seed: int | None, offset: int) -> int | None:
        if base_seed is None:
            return None
        return base_seed + offset

    def _frame_to_ppm(self, frames: list[list[bytearray]]) -> bytes:
        payload = bytearray(self.ppm_header)
        for rows in zip(*frames):
            for levels in zip(*rows):
                red = 0
                green = 0
                blue = 0
                for level, mask in zip(levels, self.world_masks):
                    if mask & CHANNEL_RED and level > red:
                        red = level
                    if mask & CHANNEL_GREEN and level > green:
                        green = level
                    if mask & CHANNEL_BLUE and level > blue:
                        blue = level
                payload.extend((red, green, blue))
        return bytes(payload)

    def render_frame(self) -> None:
        for world_mask, worm in self.worms:
            worm.step(self.worlds, self.world_by_mask[world_mask])
        frames = [world.next_frame() for _, world in self.worlds]
        for world_mask, worm in self.worms:
            worm.draw(self.world_by_mask[world_mask])
        self.photo = self.tk.PhotoImage(data=self._frame_to_ppm(frames), format="PPM")
        self.image_label.configure(image=self.photo)
        self.root.after(self.interval_ms, self.render_frame)

    def quit(self, _event=None) -> None:
        self.root.destroy()

    def run(self) -> None:
        self.render_frame()
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display additive RGB Life worlds with phosphor decay and four random worms per active color.",
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
        "--title",
        default="Conway Life Additive Worms",
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
