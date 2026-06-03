from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pbn_generator.config import PBNConfig
from pbn_generator.palette import (
    build_palette_legend_image,
    generate_palette,
    load_palette,
    print_palette_legend,
)
from pbn_generator.registry import (
    build_cluster_labeler,
    build_image_preprocessor,
    build_label_processor,
)


_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PBNResult:
    colored_path: Path
    outline_path: Path
    palette_path: Path
    intermediate_path: Path | None


class PBNGenerator:
    def __init__(self, config: PBNConfig, base_dir: str | Path | None = None):
        self.config = config
        self.base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        self.image_preprocessor = build_image_preprocessor(
            self.config.image_preprocessor
        )
        self.cluster_labeler = build_cluster_labeler(self.config.cluster_labeler)
        self.label_processor = build_label_processor(
            self.config.label_processor,
            dpi=self.config.canvas.dpi,
        )

    def run(self) -> PBNResult:
        cfg = self.config
        self._validate_config()

        border_thickness = cfg.output.border_thickness
        if border_thickness <= 0:
            border_thickness = max(1, int(round(cfg.canvas.dpi * 0.008)))

        _logger.info("Loading canvas image...")
        image = load_canvas_image(
            self._resolve_path(cfg.input.image),
            cfg.canvas.size,
            cfg.canvas.dpi,
        )
        image = self.image_preprocessor.preprocess(image)

        intermediate_path = None
        if cfg.output.write_intermediate:
            intermediate_path = self._output_path(self.image_preprocessor.output_suffix)
            save_rgb(intermediate_path, image)

        if cfg.input.palette is None:
            _logger.info(
                "Generating palette from image with %s colors...",
                cfg.palette_generator.num_colors,
            )
            names, codes, colors_rgb = generate_palette(
                image,
                cfg.palette_generator.num_colors,
            )
        else:
            names, codes, colors_rgb = load_palette(
                self._resolve_path(cfg.input.palette)
            )
        print_palette_legend(names, codes)
        _logger.info("Assigning pixels to palette colors with K-means clustering...")
        labels = self.cluster_labeler.label(image, colors_rgb)

        labels = self.label_processor.process(labels, colors_rgb)

        _logger.info("Drawing outputs...")
        border_mask = build_border_mask(labels, border_thickness)
        colored = colorize_labels(labels, colors_rgb)
        outline = build_numbered_outline(
            labels,
            border_mask,
            cfg.canvas.dpi,
            len(colors_rgb),
        )

        colored_path = self._output_path("_colored.png")
        outline_path = self._output_path("_outline.png")
        palette_path = self._output_path("_palette.png")
        save_rgb(colored_path, colored)
        save_rgb(outline_path, outline)
        save_rgb(palette_path, build_palette_legend_image(names, codes, colors_rgb))
        _logger.info("Wrote %s", colored_path)
        _logger.info("Wrote %s", outline_path)
        _logger.info("Wrote %s", palette_path)

        return PBNResult(
            colored_path=colored_path,
            outline_path=outline_path,
            palette_path=palette_path,
            intermediate_path=intermediate_path,
        )

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.label_processor.min_area < 0:
            raise ValueError("Minimum region area must be non-negative.")
        if cfg.canvas.dpi <= 0:
            raise ValueError("DPI must be positive.")
        if cfg.input.palette is not None and str(cfg.input.palette).strip() == "":
            raise ValueError("input.palette cannot be blank.")

    def _resolve_path(self, path: str | Path | None) -> Path:
        if path is None:
            raise ValueError("Path cannot be None.")
        path = Path(path)
        if path.is_absolute():
            return path
        return self.base_dir / path

    def _output_path(self, suffix: str) -> Path:
        output_directory = Path(self.config.output.directory)
        if not output_directory.is_absolute():
            output_directory = self.base_dir / output_directory

        output_stem = Path(self.config.input.image).stem
        return output_directory / f"{output_stem}{suffix}"


def load_canvas_image(image_file: str | Path, canvas_size: tuple[float, float], dpi: int) -> np.ndarray:
    width_in, height_in = canvas_size
    max_width_px = int(round(width_in * dpi))
    max_height_px = int(round(height_in * dpi))
    if max_width_px <= 0 or max_height_px <= 0:
        raise ValueError("Canvas dimensions must be positive.")

    try:
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        resample_filter = Image.LANCZOS

    image = Image.open(image_file).convert("RGB")
    source_width_px, source_height_px = image.size
    scale = min(
        max_width_px / source_width_px,
        max_height_px / source_height_px,
    )
    width_px = max(1, int(round(source_width_px * scale)))
    height_px = max(1, int(round(source_height_px * scale)))

    _logger.info(
        "Resizing to %sx%s px (%.2fx%.2f in)",
        width_px,
        height_px,
        width_px / dpi,
        height_px / dpi,
    )
    image = image.resize((width_px, height_px), resample_filter)
    return np.asarray(image)


def build_border_mask(labels: np.ndarray, thickness: int) -> np.ndarray:
    border = np.zeros(labels.shape, dtype=bool)
    border[1:, :] |= labels[1:, :] != labels[:-1, :]
    border[:-1, :] |= labels[:-1, :] != labels[1:, :]
    border[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    border[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if thickness > 1:
        kernel = np.ones((3, 3), dtype=np.uint8)
        border = cv2.dilate(
            border.astype(np.uint8),
            kernel,
            iterations=thickness - 1,
        ).astype(bool)

    return border


def colorize_labels(
    labels: np.ndarray,
    colors_rgb: np.ndarray,
) -> np.ndarray:
    return colors_rgb[labels].copy()


def build_numbered_outline(
    labels: np.ndarray,
    border_mask: np.ndarray,
    dpi: int,
    num_colors: int,
) -> np.ndarray:
    outline = np.full((*labels.shape, 3), 255, dtype=np.uint8)
    outline[border_mask] = np.array([0, 0, 0], dtype=np.uint8)
    draw_region_numbers(outline, labels, dpi, num_colors)
    return outline


def draw_region_numbers(
    image: np.ndarray,
    labels: np.ndarray,
    dpi: int,
    num_colors: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    base_scale = max(0.35, dpi / 250.0)

    for color_index in range(num_colors):
        mask = (labels == color_index).astype(np.uint8)
        component_count, components, stats, _ = cv2.connectedComponentsWithStats(
            mask,
            connectivity=8,
        )

        for component_index in range(1, component_count):
            text = str(color_index + 1)
            x = stats[component_index, cv2.CC_STAT_LEFT]
            y = stats[component_index, cv2.CC_STAT_TOP]
            w = stats[component_index, cv2.CC_STAT_WIDTH]
            h = stats[component_index, cv2.CC_STAT_HEIGHT]
            roi = (components[y : y + h, x : x + w] == component_index).astype(np.uint8)
            center_x, center_y, radius = best_label_position(roi, x, y)

            thickness = 1
            (unit_width, unit_height), unit_baseline = cv2.getTextSize(
                text,
                font,
                1.0,
                thickness,
            )
            available = max(6.0, radius * 1.45)
            scale = min(
                base_scale,
                available / max(unit_width, unit_height + unit_baseline),
            )
            scale = max(0.25, scale)

            if scale > 0.75:
                thickness = 2

            (text_width, text_height), _ = cv2.getTextSize(
                text,
                font,
                scale,
                thickness,
            )
            origin = (
                int(round(center_x - text_width / 2)),
                int(round(center_y + text_height / 2)),
            )
            cv2.putText(
                image,
                text,
                origin,
                font,
                scale,
                (0, 0, 0),
                thickness,
                lineType=cv2.LINE_AA,
            )


def best_label_position(component_roi: np.ndarray, offset_x: int, offset_y: int) -> tuple[int, int, float]:
    padded = np.pad(component_roi, 1, mode="constant", constant_values=0)
    distances = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    _, radius, _, max_location = cv2.minMaxLoc(distances)
    center_x = offset_x + max_location[0] - 1
    center_y = offset_y + max_location[1] - 1
    return center_x, center_y, radius


def save_rgb(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)
