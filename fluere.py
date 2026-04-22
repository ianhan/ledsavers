#!/usr/bin/env python3

import argparse
import math
import os
import random
import select
import sys
import termios
import tty
from dataclasses import dataclass
from pathlib import Path


WINDOW_WIDTH = 512
WINDOW_HEIGHT = 128
DEFAULT_X = 2823
DEFAULT_Y = 268
DEFAULT_INTERVAL_MS = 33
DEFAULT_IMAGE_DURATION_MS = 120_000
DEFAULT_NUM_KNOTS = 16
PALETTE_SIZE = 256
PALETTE_CYCLE_STEP = 2
SATORI_COVERAGE_PER_FRAME = 512
SATORI_START_LEVEL_OFFSET = 0
BLACK = (0, 0, 0)
STYLE_NAMES = ("flow", "wave", "spin", "leaf", "rays")
STYLE_NAME_TO_ID = {name: index for index, name in enumerate(STYLE_NAMES)}
EMBEDDED_PALETTES_TEXT = """\
Number_of_palettes 22

Cold        4 0x33ccff 0x0099ff 0x0033cc 0x0033ff
Easter      4 0x66ff00 0x6600ff 0x3399ff 0xffff33
Grayscale   6 0xffffff 0x333333 0xcccccc 0x999999 0x666666 0x000000
Hot         5 0xffff33 0xffcc00 0xff6600 0xbb0033 0xff3300
Modern      7 0xdcd386 0x767475 0xd35a63 0xd5a0a8 0x81bfbc 0xe2d788 0x95718d
Primary     4 0xffff00 0xff0000 0x00ff00 0x0000ff
Rainbow     7 0xff0066 0xff7f00 0xffff00 0x99ff00 0x33ff99 0x0066ff 0x9900cc
Romantic    4 0xffff66 0xff99cc 0x9933ff 0x990099
Southwest   6 0xddbb7b 0x5c5447 0xcabdb5 0x883c2e 0x4a5d5b 0xdabfa2
Subdued     4 0x669966 0xcccc99 0x666633 0x336666
Tranquil    4 0xcc99ff 0xcc9966 0xcccc33 0x669966
Tropical    4 0xff33cc 0x003399 0x00cc00 0xffff00
Victorian  10 0xc19ea8 0x99887e 0x5b484e 0x892b23 0xd09455
              0x80805a 0x334f43 0x78a495 0xcad6a8 0x8b4c2b
Wood        7 0x98785c 0x64563c 0x362828 0xf2bd79 0x816d52 0xfbe2ba 0xa4785f
DarkDepths  8 0x0a0f1f 0x102a3f 0x21455a 0x3f6b2f 0x8c8a22 0x47207a 0x7d1646 0x020305
PaleShadows 8 0xe2f2b8 0xb8d6a2 0x9eb6a7 0x9d97b0 0x8f7c92 0xc98590 0xd5c4ae 0x5e6664
SatinCopper 7 0x12060a 0x3b1817 0x6d2e20 0xa1543b 0xca865e 0xe8b58a 0x7d4f46
SatinTeal   7 0x09121b 0x102f45 0x14677a 0x2ba58f 0x8bddb1 0x5f43a2 0x24113a
OilSpill    8 0x20123d 0x3a3f8c 0x2787a0 0x58a331 0xb48f27 0x9f5d2e 0x6d2d77 0x09070f
Tunnels     7 0x02050b 0x03192d 0x083a58 0x0f6f83 0x15a0a0 0x2cd0c3 0x1a3fb2
Cloverfield 8 0x090b07 0x173015 0x2e5719 0x597722 0x8f8b1f 0xa55a18 0x5f2319 0x1e1208
RoseMist    7 0xc5d9b8 0x8fb69d 0x85989d 0xb58b9f 0xd08a97 0x9fcb90 0x6a7280
"""


@dataclass(frozen=True)
class Palette:
    name: str
    colors: tuple[tuple[int, int, int], ...]


@dataclass
class Knot:
    x: float
    y: float
    flowsign: float
    spinsign: float
    sectors: float
    amplitude: float
    frequency: float
    decay: float
    wavesign: float
    leafsign: int
    rayssign: int


def clamp_byte(value: float) -> int:
    if value < 0:
        return 0
    if value > 255:
        return 255
    return int(value)


def blend(
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    return (
        clamp_byte(start[0] * (1.0 - t) + end[0] * t + 0.5),
        clamp_byte(start[1] * (1.0 - t) + end[1] * t + 0.5),
        clamp_byte(start[2] * (1.0 - t) + end[2] * t + 0.5),
    )


def parse_palette_tokens(tokens: list[str], source: str) -> list[Palette]:
    if len(tokens) < 2 or tokens[0] != "Number_of_palettes":
        raise SystemExit(f"invalid palette data: {source}")

    count = int(tokens[1])
    index = 2
    palettes: list[Palette] = []

    for _ in range(count):
        if index + 1 >= len(tokens):
            raise SystemExit(f"truncated palette data: {source}")

        name = tokens[index]
        num_colors = int(tokens[index + 1])
        index += 2

        colors: list[tuple[int, int, int]] = []
        for _ in range(num_colors):
            if index >= len(tokens):
                raise SystemExit(f"truncated palette data: {source}")
            color = int(tokens[index], 16)
            colors.append(((color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF))
            index += 1

        palettes.append(Palette(name=name, colors=tuple(colors)))

    return palettes


def load_palettes(path: Path | None) -> list[Palette]:
    if path is None:
        return parse_palette_tokens(EMBEDDED_PALETTES_TEXT.split(), "embedded palette set")
    return parse_palette_tokens(path.read_text(encoding="ascii").split(), str(path))


def trunc_div_multiple(value: int, discrete: int) -> int:
    return int(value / discrete) * discrete


def build_satori_levels(width: int, height: int) -> tuple[int, ...]:
    max_block = 1
    limit = max(1, min(width, height) // 4)
    while max_block * 2 <= limit:
        max_block *= 2

    levels: list[int] = []
    block = max_block
    while block >= 1:
        levels.append(block)
        block //= 2
    return tuple(levels)


def build_spiral_indices(cols: int, rows: int) -> list[tuple[int, int]]:
    left = 0
    right = cols - 1
    top = 0
    bottom = rows - 1
    order: list[tuple[int, int]] = []

    while left <= right and top <= bottom:
        for col in range(left, right + 1):
            order.append((col, top))

        for row in range(top + 1, bottom + 1):
            order.append((right, row))

        if top < bottom:
            for col in range(right - 1, left - 1, -1):
                order.append((col, bottom))

        if left < right:
            for row in range(bottom - 1, top, -1):
                order.append((left, row))

        left += 1
        right -= 1
        top += 1
        bottom -= 1

    return order


class FluereDrawing:
    def __init__(
        self,
        width: int,
        height: int,
        num_knots: int,
        style1: int,
        style2: int,
        checkerboard: bool,
        rng: random.Random,
    ) -> None:
        self.width = width
        self.height = height
        self.num_knots = num_knots
        self.style1 = style1
        self.style2 = style2
        self.checkerboard = checkerboard
        self.leaf_discrete = 1 + 3 * rng.randrange(3)
        self.rays_discrete = 1 + 3 * rng.randrange(3)
        self.knots = self._define_knots(rng)

    def _define_knots(self, rng: random.Random) -> list[Knot]:
        zoom = 1.1
        origin_x = 0.5 * (zoom - 1.0) * self.width
        origin_y = 0.5 * (zoom - 1.0) * self.height
        knots: list[Knot] = []

        for _ in range(self.num_knots):
            x = zoom * self.width * rng.random() - origin_x
            y = zoom * self.height * rng.random() - origin_y
            flowsign = 1.0 if rng.randrange(2) else -1.0
            spinsign = 1.0 if rng.randrange(2) else -1.0
            leafsign = 1 if rng.randrange(2) else -1
            rayssign = 1 if rng.randrange(2) else -1
            wavesign = 1.0 if rng.randrange(2) else -1.0

            nspokes = 1 + rng.randrange(7)
            sectors = nspokes / (2.0 * math.pi)
            frequency = 6.0 * rng.random() + 3.0
            amplitude = 0.0 if rng.randrange(2) else 8.0 * frequency / (nspokes * nspokes)
            decay = 20.0 + rng.random() * 30.0

            knots.append(
                Knot(
                    x=x,
                    y=y,
                    flowsign=flowsign,
                    spinsign=spinsign,
                    sectors=sectors,
                    amplitude=amplitude,
                    frequency=frequency,
                    decay=decay,
                    wavesign=wavesign,
                    leafsign=leafsign,
                    rayssign=rayssign,
                )
            )

        return knots

    def fill_pixels(self) -> bytes:
        data = bytearray(self.width * self.height)
        offset = 0
        for y in range(self.height):
            for x in range(self.width):
                data[offset] = self.get_pixel_value(x, y)
                offset += 1
        return bytes(data)

    def get_pixel_value(self, x: int, y: int) -> int:
        if not self.checkerboard:
            style = self.style2
        elif (x + y) % 2 == 0:
            style = self.style1
        else:
            style = self.style2
        return self._get_value(style, x, y)

    def _get_value(self, style: int, x: int, y: int) -> int:
        if style == STYLE_NAME_TO_ID["flow"]:
            return self._get_flow_value(x, y)
        if style == STYLE_NAME_TO_ID["wave"]:
            return self._get_wave_value(x, y)
        if style == STYLE_NAME_TO_ID["spin"]:
            return self._get_spin_value(x, y)
        if style == STYLE_NAME_TO_ID["leaf"]:
            return self._get_leaf_value(x, y)
        if style == STYLE_NAME_TO_ID["rays"]:
            return self._get_rays_value(x, y)
        return 0

    def _get_spin_value(self, x: int, y: int) -> int:
        value = 0.0
        for knot in self.knots:
            dx = x - knot.x
            dy = y - knot.y
            radius = math.sqrt(dx * dx + dy * dy)

            if dx == 0.0 and dy == 0.0:
                angle = 0.0
            else:
                angle = math.atan2(dy, dx)

            angle += (
                knot.amplitude
                * knot.sectors
                * math.sin(radius / knot.frequency)
                * math.exp(-radius / knot.decay)
            )
            angle = knot.sectors * math.fmod(angle, 1.0 / knot.sectors)
            value += knot.spinsign * angle

        return int(256.0 * value) % 256

    def _get_flow_value(self, x: int, y: int) -> int:
        value = 0.0
        for knot in self.knots:
            dx = x - knot.x
            dy = y - knot.y
            radius_sq = dx * dx + dy * dy
            value += knot.flowsign * math.log(radius_sq if radius_sq > 0.0 else 1e-12)
        value *= 100.0 / self.num_knots
        return int(value) % 256

    def _get_wave_value(self, x: int, y: int) -> int:
        value = 0.0
        for knot in self.knots:
            dx = x - knot.x
            dy = y - knot.y
            radius_sq = dx * dx + dy * dy
            value += knot.wavesign * math.sin(1.5 * math.log(radius_sq if radius_sq > 0.0 else 1e-12))
        value *= 100.0 / self.num_knots
        return int(value) % 256

    def _get_leaf_value(self, x: int, y: int) -> int:
        value = 0
        for knot in self.knots:
            dx = abs(x - knot.x)
            dy = abs(y - knot.y)
            big = max(dx, dy)
            small = min(dx, dy)
            if big == 0.0:
                angle_value = 0.0
            else:
                angle_value = knot.leafsign * 75.0 * (small / big) * (small / big)
            value += trunc_div_multiple(int(angle_value), self.leaf_discrete)
        return value % 256

    def _get_rays_value(self, x: int, y: int) -> int:
        value = 0
        for knot in self.knots:
            dx = abs(x - knot.x)
            dy = abs(y - knot.y)
            big = max(dx, dy)
            small = min(dx, dy)
            if big == 0.0:
                angle_value = 0.0
            else:
                angle_value = knot.rayssign * 75.0 * (small / big) * (small / big)
            value += trunc_div_multiple(int(angle_value), self.rays_discrete)
        return value % 256


def get_color_table(
    palette: Palette,
    rng: random.Random,
    randomize: bool,
    stripes: bool,
) -> bytes:
    if randomize:
        if stripes:
            num_steps = rng.randrange(3) + 3
        else:
            num_steps = rng.randrange(6) + 5
    else:
        num_steps = len(palette.colors)

    if stripes:
        num_steps *= 2

    colors: list[tuple[int, int, int]] = []
    palette_index = 0
    for color_index in range(num_steps):
        if stripes and color_index % 2:
            colors.append(BLACK)
        elif randomize:
            colors.append(palette.colors[rng.randrange(len(palette.colors))])
        else:
            colors.append(palette.colors[palette_index])
            palette_index += 1

    table = bytearray(PALETTE_SIZE * 3 * 2)
    for step_index in range(num_steps):
        start = colors[step_index]
        end = colors[(step_index + 1) % num_steps]
        start_idx = step_index * PALETTE_SIZE // num_steps
        end_idx = (step_index + 1) * PALETTE_SIZE // num_steps
        for index in range(start_idx, end_idx):
            t = num_steps / PALETTE_SIZE * (index - start_idx)
            mixed = blend(start, end, t)
            base = index * 3
            wrap = (index + PALETTE_SIZE) * 3
            table[base] = mixed[0]
            table[base + 1] = mixed[1]
            table[base + 2] = mixed[2]
            table[wrap] = mixed[0]
            table[wrap + 1] = mixed[1]
            table[wrap + 2] = mixed[2]
    return bytes(table)


class FluerePanelApp:
    def __init__(self, tk_module, args: argparse.Namespace) -> None:
        self.tk = tk_module
        self.args = args
        self.rng = random.Random(args.seed)
        self.palettes = load_palettes(args.palette_file)
        self.fixed_palette = self._find_palette(args.palette)
        self.fixed_style1 = self._resolve_style(args.style1)
        self.fixed_style2 = self._resolve_style(args.style2)
        self.image_data = b""
        self.color_table = b""
        self.current_palette_name: str | None = None
        self.palette_offset = 0
        self.frame_counter = 0
        self.satori_drawing: FluereDrawing | None = None
        self.satori_levels = build_satori_levels(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.satori_level_index = 0
        self.satori_block_size = 1
        self.satori_blocks: list[tuple[int, int, int]] = []
        self.satori_block_cursor = 0
        self.closed = False
        self.scene_locked = False
        self.stdin_fd: int | None = None
        self.stdin_attrs: list[int] | None = None

        self.root = self.tk.Tk()
        self.root.title(args.title)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{args.x}+{args.y}")
        self.root.resizable(False, False)
        self.root.configure(bg="black")
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        if not args.decorated:
            self.root.overrideredirect(True)
        if args.topmost:
            self.root.attributes("-topmost", True)

        self.root.bind("<Escape>", self.quit)
        self.root.bind("m", self.new_map)
        self.root.bind("n", self.next_scene)
        self.root.bind("p", self.next_palette)
        self.root.bind("l", self.toggle_scene_lock)
        self.root.bind("q", self.quit)

        self.image_label = self.tk.Label(self.root, bd=0, highlightthickness=0, bg="black")
        self.image_label.pack()
        self.photo = None

        self.pixel_count = WINDOW_WIDTH * WINDOW_HEIGHT
        self.ppm_header = f"P6 {WINDOW_WIDTH} {WINDOW_HEIGHT} 255\n".encode("ascii")
        self.ppm_buffer = bytearray(len(self.ppm_header) + self.pixel_count * 3)
        self.ppm_buffer[: len(self.ppm_header)] = self.ppm_header
        self.rgb_buffer = memoryview(self.ppm_buffer)[len(self.ppm_header) :]

        self.frames_per_image = max(1, args.image_duration_ms // max(1, args.interval_ms))
        self.new_scene()
        self._setup_terminal_controls()

    def _resolve_style(self, name: str | None) -> int | None:
        if name is None:
            return None
        return STYLE_NAME_TO_ID[name]

    def _find_palette(self, name: str | None) -> Palette | None:
        if name is None:
            return None
        for palette in self.palettes:
            if palette.name.lower() == name.lower():
                return palette
        choices = ", ".join(palette.name for palette in self.palettes)
        raise SystemExit(f"unknown palette '{name}'. available palettes: {choices}")

    def _choose_style(self, fixed: int | None) -> int:
        if fixed is not None:
            return fixed
        return self.rng.randrange(len(STYLE_NAMES))

    def _choose_palette(self) -> Palette:
        if self.fixed_palette is not None:
            return self.fixed_palette
        return self.palettes[self.rng.randrange(len(self.palettes))]

    def _build_color_table(self, prefer_new_palette: bool) -> bytes:
        palette = self._choose_palette()
        if (
            prefer_new_palette
            and self.fixed_palette is None
            and len(self.palettes) > 1
            and self.current_palette_name is not None
        ):
            while palette.name == self.current_palette_name:
                palette = self._choose_palette()

        self.current_palette_name = palette.name
        return get_color_table(
            palette=palette,
            rng=self.rng,
            randomize=self.args.randomize_palette if self.args.randomize_palette is not None else bool(self.rng.randrange(2)),
            stripes=self.args.stripes if self.args.stripes is not None else bool(self.rng.randrange(2)),
        )

    def _build_scene_image(self) -> None:
        style1 = self._choose_style(self.fixed_style1)
        style2 = self._choose_style(self.fixed_style2)
        drawing = FluereDrawing(
            width=WINDOW_WIDTH,
            height=WINDOW_HEIGHT,
            num_knots=self.args.knots,
            style1=style1,
            style2=style2,
            checkerboard=self.args.checkerboard,
            rng=self.rng,
        )
        if self.args.satori:
            self._start_satori_scene(drawing)
        else:
            self.satori_drawing = None
            self.image_data = bytearray(drawing.fill_pixels())

    def new_scene(self) -> None:
        self._build_scene_image()
        self.color_table = self._build_color_table(prefer_new_palette=False)
        self.palette_offset = 0
        self.frame_counter = 0

    def _current_channel_tables(self) -> tuple[bytes, bytes, bytes]:
        start = self.palette_offset * 3
        stop = (self.palette_offset + PALETTE_SIZE) * 3
        segment = self.color_table[start:stop]
        return bytes(segment[0::3]), bytes(segment[1::3]), bytes(segment[2::3])

    def _start_satori_scene(self, drawing: FluereDrawing) -> None:
        self.satori_drawing = drawing
        self.image_data = bytearray(self.pixel_count)

        start_level_index = min(SATORI_START_LEVEL_OFFSET, len(self.satori_levels) - 1)
        coarse_block = self.satori_levels[start_level_index]
        initial_blocks = self._iter_satori_blocks(coarse_block)
        self._fill_satori_blocks(initial_blocks)
        self.satori_level_index = start_level_index + 1
        self.satori_blocks = []
        self.satori_block_cursor = 0
        if self.satori_level_index < len(self.satori_levels):
            self._prepare_satori_level(self.satori_levels[self.satori_level_index])

    def _iter_satori_blocks(self, block_size: int) -> list[tuple[int, int, int]]:
        cols = (WINDOW_WIDTH + block_size - 1) // block_size
        rows = (WINDOW_HEIGHT + block_size - 1) // block_size
        return [
            (col * block_size, row * block_size, block_size)
            for col, row in build_spiral_indices(cols, rows)
        ]

    def _prepare_satori_level(self, block_size: int) -> None:
        self.satori_block_size = block_size
        self.satori_blocks = self._iter_satori_blocks(block_size)
        self.satori_block_cursor = 0

    def _fill_satori_blocks(self, blocks: list[tuple[int, int, int]]) -> None:
        assert self.satori_drawing is not None
        for start_x, start_y, block_size in blocks:
            end_x = min(start_x + block_size, WINDOW_WIDTH)
            end_y = min(start_y + block_size, WINDOW_HEIGHT)
            sample_x = min(end_x - 1, start_x + block_size // 2)
            sample_y = min(end_y - 1, start_y + block_size // 2)
            value = self.satori_drawing.get_pixel_value(sample_x, sample_y)
            row_fill = bytes((value,)) * (end_x - start_x)
            for y in range(start_y, end_y):
                row_offset = y * WINDOW_WIDTH + start_x
                self.image_data[row_offset : row_offset + (end_x - start_x)] = row_fill

    def _advance_satori(self) -> None:
        if self.satori_drawing is None or self.satori_level_index >= len(self.satori_levels):
            return

        budget = SATORI_COVERAGE_PER_FRAME
        while budget > 0 and self.satori_block_cursor < len(self.satori_blocks):
            start_x, start_y, block_size = self.satori_blocks[self.satori_block_cursor]
            self.satori_block_cursor += 1
            budget -= max(1, block_size * block_size)
            self._fill_satori_blocks([(start_x, start_y, block_size)])

        if self.satori_block_cursor >= len(self.satori_blocks):
            self.satori_level_index += 1
            if self.satori_level_index < len(self.satori_levels):
                self._prepare_satori_level(self.satori_levels[self.satori_level_index])

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
            self.root.after(self.args.interval_ms, self._poll_terminal_input)

        print("Terminal controls: l=toggle lock, m=new map, n=next scene, p=next palette, q=quit", file=sys.stderr)

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
        self.root.after(self.args.interval_ms, self._poll_terminal_input)

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
                if self._handle_command_char(char):
                    return

    def _handle_command_char(self, char: str) -> bool:
        lowered = char.lower()
        if lowered == "l":
            self.toggle_scene_lock()
            return False
        if lowered == "m":
            self.new_map()
            return False
        if lowered == "n":
            self.next_scene()
            return False
        if lowered == "p":
            self.next_palette()
            return False
        if lowered == "q":
            self.quit()
            return True
        return False

    def render_frame(self) -> None:
        if self.closed:
            return
        if self.args.satori:
            self._advance_satori()
        red_table, green_table, blue_table = self._current_channel_tables()
        self.rgb_buffer[0::3] = self.image_data.translate(red_table)
        self.rgb_buffer[1::3] = self.image_data.translate(green_table)
        self.rgb_buffer[2::3] = self.image_data.translate(blue_table)

        self.photo = self.tk.PhotoImage(data=bytes(self.ppm_buffer), format="PPM")
        self.image_label.configure(image=self.photo)

        self.palette_offset = (self.palette_offset + PALETTE_CYCLE_STEP) % PALETTE_SIZE
        if not self.scene_locked:
            self.frame_counter += 1
            if self.frame_counter >= self.frames_per_image:
                self.new_scene()

        self.root.after(self.args.interval_ms, self.render_frame)

    def next_scene(self, _event=None) -> None:
        self.new_scene()

    def new_map(self, _event=None) -> None:
        self._build_scene_image()
        self.frame_counter = 0

    def next_palette(self, _event=None) -> None:
        self.color_table = self._build_color_table(prefer_new_palette=True)
        self.palette_offset = 0

    def toggle_scene_lock(self, _event=None) -> None:
        self.scene_locked = not self.scene_locked
        state = "locked" if self.scene_locked else "unlocked"
        print(f"Scene lock {state}", file=sys.stderr)

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
        description="Display the Fluere palette-cycling animation in a 512x128 panel window.",
    )
    parser.add_argument("--x", type=int, default=DEFAULT_X, help="Window X position in pixels.")
    parser.add_argument("--y", type=int, default=DEFAULT_Y, help="Window Y position in pixels.")
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=DEFAULT_INTERVAL_MS,
        help="Milliseconds between animation updates.",
    )
    parser.add_argument(
        "--image-duration-ms",
        type=int,
        default=DEFAULT_IMAGE_DURATION_MS,
        help="How long to show each generated image before rebuilding it. Default: 120000 (2 minutes).",
    )
    parser.add_argument(
        "--knots",
        type=int,
        default=DEFAULT_NUM_KNOTS,
        help="Number of Fluere knots to use when generating an image.",
    )
    parser.add_argument(
        "--palette",
        help="Optional embedded palette name. Defaults to random.",
    )
    parser.add_argument(
        "--palette-file",
        type=Path,
        help="Optional external Fluere palettes.txt file to override the embedded palettes.",
    )
    parser.add_argument(
        "--satori",
        action="store_true",
        help="Progressively render from low resolution to full resolution while the palette cycles.",
    )
    parser.add_argument(
        "--style1",
        choices=STYLE_NAMES,
        help="Optional first Fluere drawing style. Defaults to random.",
    )
    parser.add_argument(
        "--style2",
        choices=STYLE_NAMES,
        help="Optional second Fluere drawing style. Defaults to random.",
    )
    parser.add_argument(
        "--no-checkerboard",
        dest="checkerboard",
        action="store_false",
        default=True,
        help="Disable the alternating-pixel mix and render only style2 across the whole image.",
    )
    parser.add_argument(
        "--randomize-palette",
        dest="randomize_palette",
        action="store_true",
        default=None,
        help="Randomly choose a subset/order of palette colors when building the color table.",
    )
    parser.add_argument(
        "--no-randomize-palette",
        dest="randomize_palette",
        action="store_false",
        help="Use the palette colors in file order.",
    )
    parser.add_argument(
        "--stripes",
        dest="stripes",
        action="store_true",
        default=None,
        help="Alternate the chosen palette colors with black.",
    )
    parser.add_argument(
        "--no-stripes",
        dest="stripes",
        action="store_false",
        help="Disable black stripe insertion in the color table.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional RNG seed for repeatable output.",
    )
    parser.add_argument(
        "--title",
        default="Fluere Panel",
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
    args = parser.parse_args()
    if args.palette_file is not None:
        args.palette_file = args.palette_file.expanduser()

    if args.knots < 1:
        raise SystemExit("--knots must be at least 1")
    if args.interval_ms < 1:
        raise SystemExit("--interval-ms must be at least 1")
    if args.image_duration_ms < 1:
        raise SystemExit("--image-duration-ms must be at least 1")
    if args.palette_file is not None and not args.palette_file.is_file():
        raise SystemExit(f"palette file not found: {args.palette_file}")

    return args


def main() -> None:
    args = parse_args()

    try:
        import tkinter as tk
    except ModuleNotFoundError as exc:
        raise SystemExit("tkinter is not installed for this Python interpreter.") from exc

    FluerePanelApp(tk, args).run()


if __name__ == "__main__":
    main()
