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


def calc_zncc_on_overlaps(
    overlaps_list: List[np.ndarray],
    images_list: np.ndarray,
    masks_list: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """

    The area must be greater than `min_area` to evaluate the correlation,
    by default 0.

    """

    fg_mask = masks_list[0]
    fg_image = images_list[0]

    CC_list = np.zeros(len(overlaps_list))
    border_size_list = np.zeros(len(overlaps_list))

    for idx in range(1, len(images_list)):
        overlap = overlaps_list[idx - 1]
        bg_mask = masks_list[idx]
        bg_image = images_list[idx]

        # Calculate CC
        overlap_size = np.count_nonzero(overlap)
        border_size_list[idx - 1] = overlap_size

        if overlap_size > 0:
            CC_list[idx - 1] = image_similarity.zncc(
                fg_image[overlap].ravel(), bg_image[overlap].ravel()
            )
        else:  # border_size == 0:
            CC_list[idx - 1] = 0

        # plots.plot_mult([plots.impair(fg_image, bg_image), border_iou], [f"CC: {CC_list[idx-1]:.4f}", f"Area: {border_size_list[idx-1]}"],cols=2, base_size=10)
        fg_image = np.where((~fg_mask) & (bg_mask), bg_image, fg_image)

    # sum_borders = border_size_list.sum()
    # weights_list = np.array(
    #     [border_size / sum_borders for border_size in border_size_list]
    # )

    return CC_list, border_size_list


def multi_edge_optimized(
    mosaic: Mosaic, borders_width: list, borders_weight: list
) -> float:

    assert len(borders_width) == len(borders_weight)
    images_list, masks_list = get_images_and_masks(mosaic)

    zncc_in_borders = []
    for border_px in borders_width:
        overlaps_list = compute_mosaic_seamlines(masks_list, border_px)
        zncc_list, area_list = calc_zncc_on_overlaps(
            overlaps_list, images_list, masks_list
        )

        total_area = area_list.sum()
        if total_area == 0:
            weights_list = list(np.zeros(len(area_list)))
        else:
            weights_list = area_list / total_area

        value = np.sum(zncc_list * weights_list)
        zncc_in_borders.append(value)

    return np.sum(np.multiply(borders_weight, zncc_in_borders))
