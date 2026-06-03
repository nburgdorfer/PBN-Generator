from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from ruamel.yaml import YAML


_logger = logging.getLogger(__name__)


def load_palette(palette_file: str | Path) -> tuple[list[str], list[str], np.ndarray]:
    yaml = YAML(typ="safe", pure=True)
    with open(palette_file, "r") as palette_handle:
        palette = yaml.load(palette_handle)

    names = palette["names"]
    codes = palette["codes"]
    if len(names) != len(codes):
        raise ValueError("Palette must have the same number of names and codes.")

    colors = np.array([hex_to_rgb(code) for code in codes], dtype=np.uint8)
    return names, codes, colors


def hex_to_rgb(code: str) -> tuple[int, int, int]:
    value = str(code).strip().lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    if len(value) != 6:
        raise ValueError(f"Invalid hex color code: {code}")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def palette_lab(colors_rgb: np.ndarray) -> np.ndarray:
    colors = colors_rgb.reshape(1, -1, 3)
    return cv2.cvtColor(colors, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)


def format_hex_code(code: str) -> str:
    value = str(code).strip().lstrip("#").upper()
    return f"#{value}"


def build_palette_legend_image(
    names: list[str],
    codes: list[str],
    colors_rgb: np.ndarray,
) -> np.ndarray:
    row_height = 64
    swatch_size = 42
    padding = 20
    gap = 18
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.62
    thickness = 1

    labels = [
        f"{index}. {name}  {format_hex_code(code)}"
        for index, (name, code) in enumerate(zip(names, codes), start=1)
    ]
    text_widths = [
        cv2.getTextSize(label, font, font_scale, thickness)[0][0]
        for label in labels
    ]
    width = padding * 2 + swatch_size + gap + max(text_widths, default=240)
    height = padding * 2 + row_height * len(labels)
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    for row, (label, color) in enumerate(zip(labels, colors_rgb)):
        y = padding + row * row_height
        swatch_left = padding
        swatch_top = y + (row_height - swatch_size) // 2
        swatch_right = swatch_left + swatch_size
        swatch_bottom = swatch_top + swatch_size

        image[swatch_top:swatch_bottom, swatch_left:swatch_right] = color
        cv2.rectangle(
            image,
            (swatch_left, swatch_top),
            (swatch_right, swatch_bottom),
            (0, 0, 0),
            1,
        )
        cv2.putText(
            image,
            label,
            (swatch_right + gap, y + 40),
            font,
            font_scale,
            (0, 0, 0),
            thickness,
            lineType=cv2.LINE_AA,
        )

    return image


def print_palette_legend(names: list[str], codes: list[str]) -> None:
    _logger.info("Palette numbers:")
    for index, (name, code) in enumerate(zip(names, codes), start=1):
        _logger.info("%s: %s (%s)", index, name, code)
