from __future__ import annotations

from abc import ABC, abstractmethod
import logging

import cv2
import numpy as np

from pbn_generator.palette import palette_lab


_logger = logging.getLogger(__name__)


class LabelProcessor(ABC):
    @abstractmethod
    def process(self, labels: np.ndarray, colors_rgb: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SmoothMergeProcessor(LabelProcessor):
    def __init__(
        self,
        min_area: float,
        dpi: int,
        merge_passes: int,
        label_smooth_size: int,
        label_smooth_passes: int,
    ):
        self.min_area = min_area
        self.dpi = dpi
        self.merge_passes = merge_passes
        self.label_smooth_size = label_smooth_size
        self.label_smooth_passes = label_smooth_passes

    def process(self, labels: np.ndarray, colors_rgb: np.ndarray) -> np.ndarray:
        min_area_px = int(round(self.min_area * self.dpi * self.dpi))

        _logger.info("Smoothing region borders...")
        labels = self._smooth_label_edges(
            labels,
            len(colors_rgb),
            self.label_smooth_size,
            self.label_smooth_passes,
        )

        _logger.info(
            "Merging sections smaller than %s square inches (%s px)...",
            self.min_area,
            min_area_px,
        )
        labels = self._merge_small_regions(
            labels,
            colors_rgb,
            min_area_px,
            self.merge_passes,
        )
        return labels
        

    def _smooth_label_edges(
        self,
        labels: np.ndarray,
        num_colors: int,
        window_size: int,
        passes: int,
    ) -> np.ndarray:
        if window_size <= 1 or passes <= 0:
            return labels
        if window_size % 2 == 0:
            window_size += 1

        labels = labels.copy()
        for _ in range(passes):
            best_count = np.full(labels.shape, -1.0, dtype=np.float32)
            smoothed = labels.copy()

            for color_index in range(num_colors):
                mask = (labels == color_index).astype(np.float32)
                counts = cv2.boxFilter(
                    mask,
                    ddepth=-1,
                    ksize=(window_size, window_size),
                    normalize=False,
                    borderType=cv2.BORDER_REPLICATE,
                )
                counts[labels == color_index] += 0.01
                replace = counts > best_count
                smoothed[replace] = color_index
                best_count[replace] = counts[replace]

            labels = smoothed

        return labels

    def _merge_small_regions(
        self,
        labels: np.ndarray,
        colors_rgb: np.ndarray,
        min_area_px: int,
        max_passes: int,
    ) -> np.ndarray:
        if min_area_px <= 1:
            return labels

        colors_lab = palette_lab(colors_rgb)
        kernel = np.ones((3, 3), dtype=np.uint8)
        labels = labels.copy()
        num_colors = len(colors_rgb)

        for pass_index in range(max_passes):
            changed = 0
            for color_index in range(num_colors):
                mask = (labels == color_index).astype(np.uint8)
                component_count, components, stats, _ = cv2.connectedComponentsWithStats(
                    mask,
                    connectivity=8,
                )

                for component_index in range(1, component_count):
                    area = stats[component_index, cv2.CC_STAT_AREA]
                    if area >= min_area_px:
                        continue

                    component_mask = components == component_index
                    neighbor_mask = cv2.dilate(
                        component_mask.astype(np.uint8),
                        kernel,
                        iterations=1,
                    ).astype(bool)
                    neighbor_mask &= ~component_mask

                    neighbor_labels = labels[neighbor_mask]
                    neighbor_labels = neighbor_labels[neighbor_labels != color_index]
                    if neighbor_labels.size == 0:
                        continue

                    next_label = self._choose_merge_label(
                        color_index,
                        neighbor_labels,
                        colors_lab,
                        num_colors,
                    )
                    labels[component_mask] = next_label
                    changed += 1

            _logger.info(
                "Merge pass %s: merged %s undersized sections",
                pass_index + 1,
                changed,
            )
            if changed == 0:
                break

        return labels

    def _choose_merge_label(
        self,
        current_label: int,
        neighbor_labels: np.ndarray,
        colors_lab: np.ndarray,
        num_colors: int,
    ) -> int:
        counts = np.bincount(neighbor_labels, minlength=num_colors).astype(np.float32)
        color_distances = np.linalg.norm(colors_lab - colors_lab[current_label], axis=1)
        scores = counts / (1.0 + color_distances)
        scores[current_label] = -1.0
        return int(np.argmax(scores))
