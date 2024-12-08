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


def crop_image(image: np.ndarray, px: int) -> np.ndarray:
    return image[px:-px, px:-px]


def add_gaussian_noise(image: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image


def add_salt_pepper_noise(image: np.ndarray, prob: float) -> np.ndarray:
    noisy_image = np.copy(image)
    salt_mask = np.random.rand(*image.shape) < (prob / 2)
    pepper_mask = np.random.rand(*image.shape) < (prob / 2)
    noisy_image[salt_mask] = 255
    noisy_image[pepper_mask] = 0
    return noisy_image
