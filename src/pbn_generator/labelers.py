from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from pbn_generator.palette import palette_lab


class ClusterLabeler(ABC):
    @abstractmethod
    def label(self, image_rgb: np.ndarray, colors_rgb: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class KMeansLabeler(ClusterLabeler):
    def __init__(self, allow_unused_colors: bool = True):
        self.allow_unused_colors = allow_unused_colors

    def label(self, image_rgb: np.ndarray, colors_rgb: np.ndarray) -> np.ndarray:
        cluster_labels, centers_lab = self._cluster_image_lab(image_rgb, len(colors_rgb))
        center_to_palette = self._map_cluster_centers_to_palette(
            centers_lab,
            colors_rgb,
        )
        palette_labels = center_to_palette[cluster_labels.flatten()]
        return palette_labels.reshape(image_rgb.shape[:2])

    def _cluster_image_lab(
        self,
        image_rgb: np.ndarray,
        num_colors: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if num_colors <= 0:
            raise ValueError("Palette must contain at least one color.")

        image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        pixels = image_lab.reshape(-1, 3).astype(np.float32)
        k = min(num_colors, pixels.shape[0])

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
        _, cluster_labels, centers_lab = cv2.kmeans(
            pixels,
            k,
            None,
            criteria,
            10,
            cv2.KMEANS_RANDOM_CENTERS,
        )
        return cluster_labels.reshape(image_rgb.shape[:2]), centers_lab

    def _map_cluster_centers_to_palette(
        self,
        centers_lab: np.ndarray,
        colors_rgb: np.ndarray,
    ) -> np.ndarray:
        colors_lab = palette_lab(colors_rgb)
        distances = np.linalg.norm(
            centers_lab[:, None, :] - colors_lab[None, :, :],
            axis=2,
        )

        if self.allow_unused_colors:
            return np.argmin(distances, axis=1).astype(np.int32)

        assignments = np.full(centers_lab.shape[0], -1, dtype=np.int32)
        used_palette_colors = set()
        center_count, palette_count = distances.shape
        ranked_pairs = np.argsort(distances, axis=None)

        for pair_index in ranked_pairs:
            center_index, palette_index = np.unravel_index(pair_index, distances.shape)
            if assignments[center_index] != -1:
                continue
            if palette_index in used_palette_colors:
                continue
            assignments[center_index] = palette_index
            used_palette_colors.add(palette_index)
            if len(used_palette_colors) == min(center_count, palette_count):
                break

        for center_index in range(center_count):
            if assignments[center_index] == -1:
                assignments[center_index] = int(np.argmin(distances[center_index]))

        return assignments
