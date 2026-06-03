from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np


class ImagePreprocessor(ABC):
    output_suffix = "_preprocessed.png"

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class BilateralPreprocessor(ImagePreprocessor):
    output_suffix = "_bilat.png"

    def __init__(self, filter_size: int):
        self.filter_size = filter_size

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if self.filter_size <= 1:
            return image

        filter_size = self.filter_size
        if filter_size % 2 == 0:
            filter_size += 1

        return cv2.bilateralFilter(
            image,
            filter_size,
            sigmaColor=55,
            sigmaSpace=55,
        )
