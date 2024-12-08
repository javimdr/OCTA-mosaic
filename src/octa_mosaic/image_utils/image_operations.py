import cv2
import numpy as np


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    img_copy = image.copy()
    mask_copy = mask.copy()
    if mask_copy.dtype != bool:
        mask_copy = mask_copy.astype(bool)

    img_copy[~mask_copy] = 0
    return img_copy


def erode_mask(mask, pixels=10):
    k_size = pixels * 2 + 1
    kernel = np.ones((k_size, k_size))

    return cv2.erode(mask.astype("float32"), kernel, iterations=1).astype(bool)


def dilate_mask(mask, pixels=10):
    k_size = pixels * 2 + 1
    kernel = np.ones((k_size, k_size))

    return cv2.dilate(mask.astype("float32"), kernel, iterations=1).astype(bool)
