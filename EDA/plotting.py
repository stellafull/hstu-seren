from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

FONT = ImageFont.load_default()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, *, fill: str = 'black') -> None:
    draw.text(xy, text, fill=fill, font=FONT)


def save_line_chart(points: Iterable[tuple[float, float]], output_path: Path, *, title: str, x_label: str, y_label: str) -> None:
    points = list(points)
    width, height = 900, 520
    margin_left, margin_right, margin_top, margin_bottom = 80, 40, 60, 80
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    _ensure_parent(output_path)

    if not points:
        _text(draw, (20, 20), f"{title}\\nNo data available.")
        image.save(output_path)
        return

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if min_x == max_x:
        max_x += 1.0
    if min_y == max_y:
        max_y += 1.0

    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom

    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline='black', width=2)
    _text(draw, (width // 2 - 120, 18), title)
    _text(draw, (width // 2 - 40, height - 40), x_label)
    _text(draw, (12, height // 2), y_label)

    def project(x: float, y: float) -> tuple[int, int]:
        px = plot_left + int((x - min_x) / (max_x - min_x) * (plot_right - plot_left))
        py = plot_bottom - int((y - min_y) / (max_y - min_y) * (plot_bottom - plot_top))
        return px, py

    projected = [project(x, y) for x, y in points]
    for idx in range(len(projected) - 1):
        draw.line([projected[idx], projected[idx + 1]], fill='steelblue', width=3)
    for px, py in projected:
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill='tomato', outline='tomato')

    for tick_idx in range(5):
        frac = tick_idx / 4
        x_value = min_x + frac * (max_x - min_x)
        y_value = min_y + frac * (max_y - min_y)
        px = plot_left + int(frac * (plot_right - plot_left))
        py = plot_bottom - int(frac * (plot_bottom - plot_top))
        draw.line((px, plot_bottom, px, plot_bottom + 6), fill='black', width=1)
        draw.line((plot_left - 6, py, plot_left, py), fill='black', width=1)
        _text(draw, (px - 20, plot_bottom + 10), f'{x_value:.2f}')
        _text(draw, (5, py - 5), f'{y_value:.2f}')

    image.save(output_path)


def save_heatmap(values: list[list[float]], labels_x: list[str], labels_y: list[str], output_path: Path, *, title: str, annotations: list[list[str]] | None = None) -> None:
    width, height = 840, 640
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    _ensure_parent(output_path)

    rows = len(values)
    cols = len(values[0]) if rows else 0
    if rows == 0 or cols == 0:
        _text(draw, (20, 20), f"{title}\\nNo data available.")
        image.save(output_path)
        return

    plot_left, plot_top = 180, 120
    cell_w, cell_h = 220, 180
    flat = [value for row in values for value in row]
    min_v, max_v = min(flat), max(flat)
    if min_v == max_v:
        max_v += 1.0

    _text(draw, (width // 2 - 120, 32), title)
    for row_idx, row_label in enumerate(labels_y):
        _text(draw, (20, plot_top + row_idx * cell_h + cell_h // 2), row_label)
    for col_idx, col_label in enumerate(labels_x):
        _text(draw, (plot_left + col_idx * cell_w + 40, 80), col_label)

    for row_idx, row in enumerate(values):
        for col_idx, value in enumerate(row):
            frac = (value - min_v) / (max_v - min_v)
            red = int(255 * frac)
            blue = int(255 * (1 - frac))
            color = (255 - blue // 2, 230 - red // 5, 255 - red // 2)
            x0 = plot_left + col_idx * cell_w
            y0 = plot_top + row_idx * cell_h
            rect = (x0, y0, x0 + cell_w - 8, y0 + cell_h - 8)
            draw.rectangle(rect, fill=color, outline='black', width=2)
            text = f'{value:.3f}'
            if annotations is not None:
                text = annotations[row_idx][col_idx]
            _text(draw, (x0 + 16, y0 + 24), text)

    image.save(output_path)
