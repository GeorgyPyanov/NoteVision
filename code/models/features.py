"""Feature extraction shared by the training pipeline and prediction API."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.feature import hog

from code.datasets.preprocess import IMAGE_SIZE, preprocess_image

DEFAULT_HOG_PARAMS: dict[str, Any] = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
}


def extract_hog_features(image_or_path: Image.Image | str | Path,
                         hog_params: dict[str, Any] | None = None) -> np.ndarray:
    """Apply canonical image preprocessing and extract one HOG feature vector."""
    params = DEFAULT_HOG_PARAMS if hog_params is None else hog_params
    image = preprocess_image(image_or_path, image_size=IMAGE_SIZE)
    return hog(image, **params, feature_vector=True).astype(np.float32)


def extract_batch(paths: list[str | Path], hog_params: dict[str, Any] | None = None) -> np.ndarray:
    """Extract HOG features in deterministic manifest order."""
    return np.vstack([extract_hog_features(path, hog_params) for path in paths])
