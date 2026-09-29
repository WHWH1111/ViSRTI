#!/usr/bin/env python3
"""Compose the regenerated Figure 2 panels on the author's original canvas."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageChops


CANVAS_SIZE = (4000, 2598)
TOP_BOX = (90, 35, 2240, 1535)
BOTTOM_BOX = (90, 1615, 2240, 2425)


def _trim_white(image: Image.Image, padding: int = 12) -> Image.Image:
    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, "white")
    difference = ImageChops.difference(rgb, background).convert("L")
    difference = difference.point(lambda value: 255 if value > 8 else 0)
    bounds = difference.getbbox()
    if bounds is None:
        raise ValueError("Input figure is blank")
    left, top, right, bottom = bounds
    return rgb.crop(
        (
            max(0, left - padding),
            max(0, top - padding),
            min(rgb.width, right + padding),
            min(rgb.height, bottom + padding),
        )
    )


def _fit(image: Image.Image, box: Tuple[int, int, int, int]) -> Tuple[Image.Image, int, int]:
    left, top, right, bottom = box
    target_width = right - left
    target_height = bottom - top
    scale = min(target_width / image.width, target_height / image.height)
    size = (round(image.width * scale), round(image.height * scale))
    resized = image.resize(size, Image.Resampling.LANCZOS)
    x = left + (target_width - resized.width) // 2
    y = top + (target_height - resized.height) // 2
    return resized, x, y


def compose(top_path: Path, bottom_path: Path, output_stem: Path) -> None:
    canvas = Image.new("RGB", CANVAS_SIZE, "white")
    for source, box in ((top_path, TOP_BOX), (bottom_path, BOTTOM_BOX)):
        with Image.open(source) as handle:
            panel = _trim_white(handle)
        panel, x, y = _fit(panel, box)
        canvas.paste(panel, (x, y))

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_stem.with_suffix(".png"), dpi=(300, 300))
    canvas.save(
        output_stem.with_suffix(".tiff"),
        dpi=(300, 300),
        compression="tiff_lzw",
    )
    canvas.save(output_stem.with_suffix(".pdf"), resolution=300)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=Path, required=True, help="Figure 2a-d image")
    parser.add_argument("--bottom", type=Path, required=True, help="Figure 2e-f image")
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    compose(args.top.resolve(), args.bottom.resolve(), args.output_stem.resolve())


if __name__ == "__main__":
    main()
