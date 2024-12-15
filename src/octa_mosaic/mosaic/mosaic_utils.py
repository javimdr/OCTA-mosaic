from typing import List, Tuple

import cv2
import numpy as np

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
