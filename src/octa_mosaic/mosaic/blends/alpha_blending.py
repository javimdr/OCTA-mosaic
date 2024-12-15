import numpy as np

from octa_mosaic.mosaic import mosaic_utils
from octa_mosaic.mosaic.mosaic import Mosaic


def _blender_two_images(
    fg_image: np.ndarray,
    bg_image: np.ndarray,
    fg_mask: np.ndarray,
    bg_mask: np.ndarray,
    anchor_px: int = 10,
    stripes: int = 10,
) -> np.ndarray:
    """Blends two images with overlapping regions using gradient transitions.

    Args:
        fg_image (np.ndarray): Foreground image.
        bg_image (np.ndarray): Background image.
        fg_mask (np.ndarray): Mask indicating the foreground region in the mosaic.
        bg_mask (np.ndarray): Mask indicating the background region in the mosaic.
        anchor_px (int, optional): Width of the blending transition in pixels.
            Defaults to 10.
        stripes (int, optional): Number of gradient transition stripes.
            Defaults to 10.

    Returns:
        np.ndarray: The blended image as a NumPy array.
    """
    # Calculate the intersection between images
    intersection = np.logical_and(fg_mask, bg_mask)

    # Compute gradient transition stripes
    stripe_fractions = [(anchor_px / stripes) * f for f in range(1, stripes + 2)]
    stripes_list = np.array(
        [
            mosaic_utils.calc_border_of_overlap(fg_mask, bg_mask, int(stripe)).astype(
                "float32"
            )
            for stripe in stripe_fractions
        ]
    )

    border = stripes_list[-1].astype("bool")
    to_remove = np.logical_xor(border, intersection)
    stripes_list[-1] = 0  # Clear last stripe to avoid adding it to the blend
    stripes_list[:-1] *= 1 / anchor_px

    # Aplpy gradient blending to background
    bg_mask[to_remove] = 0
    bg_image[to_remove] = 0
    bg_gradient = np.where(border, np.sum(stripes_list, axis=0), bg_mask)
    bg_gradient = np.clip(bg_gradient, 0, 1)
    bg_image = bg_image.astype("float32") * bg_gradient

    # Aplpy gradient blending to foreground
    fg_gradient = np.where(border, abs(1 - bg_gradient), fg_mask)
    fg_gradient = np.clip(fg_gradient, 0, 1)
    fg_image = fg_image.astype("float32") * fg_gradient

    # Combine both images
    updated_fg_image = np.clip(fg_image + bg_image, 0, 255).astype("uint8")
    return updated_fg_image


def alpha_blending(mosaic: Mosaic, anchor_px: int = 10, strides: int = 10) -> np.ndarray:
    """Blends a sequence of images in a mosaic using gradient transitions.

    Args:
        mosaic (Mosaic): Mosaic to blend its images.
        anchor_px (int, optional): Width of the blending transition in pixels.
            Defaults to 10.
        strides (int, optional): Number of gradient transition stripes.
            Defaults to 10.

    Returns:
        np.ndarray: The fully blended image as a NumPy array.
    """
    images_list, masks_list = mosaic_utils.get_images_and_masks(mosaic)

    fg_mask = masks_list[0]
    fg_image = images_list[0]

    for idx in range(1, mosaic.n_images()):
        bg_mask = masks_list[idx]
        bg_image = images_list[idx]

        fg_image = _blender_two_images(
            fg_image, bg_image, fg_mask, bg_mask, anchor_px, strides
        )
        fg_mask = np.logical_or(fg_mask, bg_mask)

    return fg_image
