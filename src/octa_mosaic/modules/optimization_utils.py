from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from octa_mosaic.image_utils import image_similarity
from octa_mosaic.mosaic.mosaic import Mosaic
from octa_mosaic.mosaic.mosaic_utils import (
    compute_mosaic_seamlines,
    get_images_and_masks,
)
from octa_mosaic.mosaic.transforms.tf_utils import individual_to_mosaic


def as_individual_objective_function(
    tf_individual: np.ndarray,
    function: Callable[[Mosaic, Any], float],
    initial_mosaic: Mosaic,
    *function_args: Optional[Tuple[Any]],
    **function_kwargs: Optional[Dict[str, Any]],
) -> float:
    """
    Evaluates the objective function for an individual solution in the context of
    a mosaic optimization problem.

    This function converts the given individual representation (optimization vector)
    into a `Mosaic` object, then applies the provided evaluation function to compute its
    fitness.

    Args:
        tf_individual (np.ndarray): An array representing the individual solution.
        function (Callable[[Mosaic, Any], float]): The objective function used
            to evaluate the mosaic. It takes a `Mosaic` object and additional
            parameters as inputs and returns a float value representing the fitness.
        initial_mosaic (Mosaic): The initial mosaic configuration, used as a
            starting point for constructing the current mosaic.
        *function_args (Optional[Tuple[Any]]): Positional arguments to pass to
            the objective function.
        **function_kwargs (Optional[Dict[str, Any]]): Keyword arguments to pass
            to the objective function.

    Returns:
        float: The fitness value of the individual, as computed by the objective function.
    """
    current_mosaic = individual_to_mosaic(tf_individual, initial_mosaic)
    return function(current_mosaic, *function_args, **function_kwargs)


def calc_zncc_on_seamlines(
    seamlines: List[np.ndarray], images: np.ndarray, masks: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the ZNCC (Zero-Mean Normalized Cross-Correlation) on the seamlines regions
    between the mosaic images.

    The correation values are weighted by the seamline area size.

    Args:
        seamlines (List[np.ndarray]): A list of binary masks representing the
            seamlines regions between consecutive images.
        images (np.ndarray): Mosaic images list.
        masks (np.ndarray): Mosaic images mask list.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - An array with the ZNCC values for each seamline region.
            - An array with corresponding seamline area size.
    """

    fg_mask = masks[0]
    fg_image = images[0]

    zncc_values = np.zeros(len(seamlines))
    seamline_areas = np.zeros(len(seamlines))

    for idx in range(1, len(images)):
        curr_seamline = seamlines[idx - 1]
        bg_mask = masks[idx]
        bg_image = images[idx]

        # Calculate CC
        curr_seamline_area = np.count_nonzero(curr_seamline)
        seamline_areas[idx - 1] = curr_seamline_area

        if curr_seamline_area > 0:
            zncc_values[idx - 1] = image_similarity.zncc(
                fg_image[curr_seamline].ravel(), bg_image[curr_seamline].ravel()
            )
        else:
            zncc_values[idx - 1] = 0

        fg_image = np.where((~fg_mask) & (bg_mask), bg_image, fg_image)

    return zncc_values, seamline_areas


def calc_zncc_on_multiple_seamlines(
    mosaic: Mosaic, widths: List[int], weights: List[float]
) -> float:
    """
    Compute the ZNCC values for seamlines of multiple widths in the mosaic, considering
    their respective areas. The ZNCC values are weighted to control to which seamline
    should have more importance.

    For each seamline width, the function calculates the ZNCC values for the overlap
    between the foreground and background images along the seamline. The values are then
    weighted by the areas of the overlap, and the final weighted sum is computed using
    the provided weights.

    Args:
        mosaic (Mosaic): The mosaic to evaluate.
        widths (list): A list of seamline widths (in pixels) to evaluate.
        weights (list): A list of weights to apply to each seamline's ZNCC value.

    Returns:
        float: The final weighted value.
    """

    assert len(widths) == len(weights)
    images_list, masks_list = get_images_and_masks(mosaic)

    zncc_in_seamlines = []
    for width in widths:
        mosaic_seamlines = compute_mosaic_seamlines(masks_list, width)
        zncc_values, area_sizes = calc_zncc_on_seamlines(
            mosaic_seamlines, images_list, masks_list
        )

        total_area = area_sizes.sum()
        if total_area == 0:
            weights_list = list(np.zeros(len(area_sizes)))
        else:
            weights_list = area_sizes / total_area

        value = np.sum(zncc_values * weights_list)
        zncc_in_seamlines.append(value)

    return np.sum(np.multiply(weights, zncc_in_seamlines))
