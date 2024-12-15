from typing import List, Tuple

import cv2
import numpy as np

from octa_mosaic.image_utils.image_operations import dilate_mask
from octa_mosaic.mosaic.mosaic import Mosaic


def get_images_and_masks(
    mosaic: Mosaic,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Retrieves individual images and corresponding binary masks of each transformed image
    within a mosaic.

    Args:
        mosaic (Mosaic): The Mosaic object containing image data and transforms.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: A tuple containing:
            - images_list (List[np.ndarray]): List with all transformed images.
            - masks_list (np.ndarray): List of binary masks for each transformed image.
    """
    n_images = mosaic.n_images()
    mosaic_size = mosaic.mosaic_size

    masks_list = np.zeros((n_images, *mosaic_size), bool)
    images_list = np.zeros((n_images, *mosaic_size))

    for idx in range(n_images):
        image, _, _ = mosaic.get_image_data(idx)
        tf = mosaic.image_centered_transform(idx)

        mask_tf = cv2.warpPerspective(np.ones_like(image), tf.params, mosaic_size[::-1])
        image_tf = cv2.warpPerspective(image, tf.params, mosaic_size[::-1])

        masks_list[idx] = mask_tf.astype(bool)
        images_list[idx] = image_tf

    return images_list, masks_list


def compute_seamline(fg_mask: np.ndarray, bg_mask: np.ndarray, width: int = 10):
    """
    Compute the seamline zone between the foreground and background masks, with a
        specified width.

    Args:
        fg_mask (np.ndarray): The binary mask representing the foreground image's
            position in the mosaic.
        bg_mask (np.ndarray): The binary mask representing the background image's
            position in the mosaic.
        width (int, optional): The width (in pixels) of the seamline.
            Default is 10.

    Returns:
        np.ndarray: A binary mask representing the seamline.
    """

    overlap = np.logical_and(fg_mask, bg_mask)
    bg_non_overlaped_zone = np.logical_xor(bg_mask, overlap)
    bg_non_overlaped_zone_dilated = dilate_mask(bg_non_overlaped_zone, width)

    return np.logical_and(overlap, bg_non_overlaped_zone_dilated)


def compute_mosaic_seamlines(masks: np.ndarray, width: int) -> List[np.ndarray]:
    """
    Compute the overlap along the seamlines between the foreground and background images.

    Args:
        masks (np.ndarray): A list of binary masks representing the foreground
            (first mask) and background (subsequent masks) for each image in the mosaic.
        width (int): The width of the seamline (in pixels).

    Returns:
        List[np.ndarray]: A list of binary masks representing the seamline overlaps
            between consecutive images.
    """
    assert len(masks) > 0

    fg_mask = masks[0]

    overlaps_list = []
    for idx in range(1, len(masks)):
        bg_mask = masks[idx]
        seamline = compute_seamline(fg_mask, bg_mask, width)
        fg_mask = np.logical_or(fg_mask, bg_mask)

        overlaps_list.append(seamline)
    return overlaps_list
