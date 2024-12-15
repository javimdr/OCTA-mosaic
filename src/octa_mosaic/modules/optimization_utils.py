from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from octa_mosaic.image_utils import image_similarity
from octa_mosaic.image_utils.image_operations import dilate_mask
from octa_mosaic.mosaic.mosaic import Mosaic
from octa_mosaic.mosaic.transforms.tf_utils import individual_to_mosaic


def get_images_and_masks_list(mosaic: Mosaic):
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


def calc_border_of_overlap(fg, bg, border_px=10):
    """
    fg : foreground image
    bg : background image
    """

    overlap = np.logical_and(fg, bg)
    bg_non_overlaped_zone = np.logical_xor(bg, overlap)
    bg_non_overlaped_zone_dilated = dilate_mask(bg_non_overlaped_zone, border_px)

    return np.logical_and(overlap, bg_non_overlaped_zone_dilated)


# ======== Funciones objetivo ========
def calc_CC_on_overlaps_and_areas(
    mosaic: Mosaic, border_px: int = 10, min_area: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """_summary_

    Parameters
    ----------
    mosaic : mosaic
        Mosaic to evaluate.
    border_px : int, optional
        Anchor of the edge overlap, by default 10
    min_area : int, optional
        The area must be greater than `min_area` to evaluate the correlation,
        by default 0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Correlation list and area size of each overlap.
    """
    images_list, masks_list = get_images_and_masks_list(mosaic)

    fg_mask = masks_list[0]
    fg_image = images_list[0]
    CC_list = np.zeros(mosaic.n_images() - 1)
    border_size_list = np.zeros(mosaic.n_images() - 1)

    for idx in range(1, mosaic.n_images()):
        bg_mask = masks_list[idx]
        bg_image = images_list[idx]

        # Calculate CC
        border_iou = calc_border_of_overlap(fg_mask, bg_mask, border_px)
        border_size = np.count_nonzero(border_iou)
        border_size_list[idx - 1] = border_size

        if border_size > min_area:
            CC_list[idx - 1] = image_similarity.zncc(
                fg_image[border_iou].ravel(), bg_image[border_iou].ravel()
            )
        else:  # border_size == 0:
            CC_list[idx - 1] = 0

        # plots.plot_mult([plots.impair(fg_image, bg_image), border_iou], [f"CC: {CC_list[idx-1]:.4f}", f"Area: {border_size_list[idx-1]}"],cols=2, base_size=10)
        fg_image = np.where((~fg_mask) & (bg_mask), bg_image, fg_image)
        fg_mask = np.logical_or(fg_mask, bg_mask)

    sum_borders = border_size_list.sum()
    weights_list = np.array(
        [border_size / sum_borders for border_size in border_size_list]
    )

    return CC_list, weights_list


def func_objetivo_borde(individual, mosaic, border_px=10, boder_size_th=0):
    """Función base. NO USAR. Devuelve una lista de correlaciones y sus pesos asociados"""
    curr_mosaic = individual_to_mosaic(individual, mosaic)
    return calc_CC_on_overlaps_and_areas(curr_mosaic, border_px, boder_size_th)


def func_objetivo_borde_sum(individual, mosaic, border_px=10):
    CC_list, weights_list = func_objetivo_borde(individual, mosaic, border_px)
    return np.sum(CC_list)


def func_objetivo_borde_prod(individual, mosaic, border_px=10):
    CC_list, weights_list = func_objetivo_borde(individual, mosaic, border_px)
    return np.prod(CC_list)


def func_objetivo_borde_weighted_sum(individual, mosaic, border_px=10):
    CC_list, weights_list = func_objetivo_borde(individual, mosaic, border_px)
    return np.sum(CC_list * weights_list)


def objective_function_multi_edge(
    individual: np.ndarray, mosaic: Mosaic, borders_width: list, borders_weight: list
) -> float:
    """
    Calcula el valor de ZNCC.
    """
    assert len(borders_width) == len(borders_weight)

    CC_border_list = [
        func_objetivo_borde_weighted_sum(individual, mosaic, border_px=border)
        for border in borders_width
    ]

    return np.sum(np.multiply(borders_weight, CC_border_list))


### Geometric Mean


def objective_function_edge_geometric_mean(individual, mosaic, border_px=10):
    CC_list, _ = func_objetivo_borde(individual, mosaic, border_px)
    CC_list_corrected = np.where(CC_list > 0, CC_list, 1e-5)

    n = len(CC_list_corrected)
    assert n > 0, "No hay ningún valor de correlación"
    geometric_mean = np.prod(CC_list_corrected) ** (1 / n)
    return geometric_mean


def objective_function_multi_edge_geometric_mean(
    individual: np.ndarray, mosaic: Mosaic, borders_width: list, borders_weight: list
) -> float:
    """
    Calcula el valor de ZNCC.
    """
    assert len(borders_width) == len(borders_weight)

    CC_border_list = [
        objective_function_edge_geometric_mean(individual, mosaic, border_px=border)
        for border in borders_width
    ]

    return np.sum(np.multiply(borders_weight, CC_border_list))


# Geometric mean v2


def objective_function_edge_geometric_mean_v2(individual, mosaic, border_px=10):
    CC_list, _ = func_objetivo_borde(individual, mosaic, border_px)
    n = len(CC_list)
    assert n > 0, "No hay ningún valor de correlación."

    CC_list = np.array(CC_list)
    if np.any(CC_list < 0):
        return -1

    geometric_mean = np.prod(CC_list) ** (1 / n)
    return geometric_mean


def objective_function_multi_edge_geometric_mean_v2(
    individual: np.ndarray, mosaic: Mosaic, borders_width: list, borders_weight: list
) -> float:
    """
    Calcula el valor de ZNCC.
    """
    assert len(borders_width) == len(borders_weight)

    CC_border_list = [
        objective_function_edge_geometric_mean_v2(individual, mosaic, border_px=border)
        for border in borders_width
    ]

    return np.sum(np.multiply(borders_weight, CC_border_list))


# =========== DICE ===================
def calc_dice_and_areas(mosaic, border_px=10, boder_size_th=0):
    images_list, masks_list = get_images_and_masks_list(mosaic)

    fg_mask = masks_list[0]
    fg_image = images_list[0]
    CC_list = np.zeros(mosaic.n_images() - 1)
    border_size_list = np.zeros(mosaic.n_images() - 1)

    for idx in range(1, mosaic.n_images()):
        bg_mask = masks_list[idx]
        bg_image = images_list[idx]

        # Calculate CC
        border_iou = calc_border_of_overlap(fg_mask, bg_mask, border_px)
        boder_size = np.count_nonzero(border_iou)
        border_size_list[idx - 1] = boder_size

        if boder_size > boder_size_th:
            CC_list[idx - 1] = image_similarity.zncc(
                fg_image[border_iou].ravel(), bg_image[border_iou].ravel()
            )

        # plots.plot_mult([plots.impair(fg_image, bg_image), border_iou], [f"CC: {CC_list[idx-1]:.4f}", f"Area: {border_size_list[idx-1]}"],cols=2, base_size=10)
        fg_image = np.where((~fg_mask) & (bg_mask), bg_image, fg_image)
        fg_mask = np.logical_or(fg_mask, bg_mask)

    sum_borders = border_size_list.sum()
    weights_list = np.array(
        [border_size / sum_borders for border_size in border_size_list]
    )

    return CC_list, weights_list


def func_objetivo_edge_dice(individual, base_mosaic, border_px=10):
    individual_mosaic = individual_to_mosaic(individual, base_mosaic)
    DICE_list, weights_list = calc_dice_and_areas(individual_mosaic, border_px)
    return np.sum(DICE_list * weights_list)


# ===========


def calc_metric_edge(
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    mosaic: Mosaic,
    border_px: int = 10,
    boder_size_th: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:

    images_list, masks_list = get_images_and_masks_list(mosaic)

    fg_mask = masks_list[0]
    fg_image = images_list[0]
    metric_list = np.zeros(mosaic.n_images() - 1)
    border_size_list = np.zeros(mosaic.n_images() - 1)

    for idx in range(1, mosaic.n_images()):
        bg_mask = masks_list[idx]
        bg_image = images_list[idx]

        # Calculate CC
        border_iou = calc_border_of_overlap(fg_mask, bg_mask, border_px)
        boder_size = np.count_nonzero(border_iou)
        border_size_list[idx - 1] = boder_size

        if boder_size > boder_size_th:
            metric_list[idx - 1] = metric_func(
                fg_image[border_iou].ravel(), bg_image[border_iou].ravel()
            )

        # plots.plot_mult([plots.impair(fg_image, bg_image), border_iou], [f"CC: {CC_list[idx-1]:.4f}", f"Area: {border_size_list[idx-1]}"],cols=2, base_size=10)
        fg_image = np.where((~fg_mask) & (bg_mask), bg_image, fg_image)
        fg_mask = np.logical_or(fg_mask, bg_mask)

    sum_borders = border_size_list.sum()
    weights_list = np.array(
        [border_size / sum_borders for border_size in border_size_list]
    )

    return metric_list, weights_list


def calc_metric_multiples_edges(
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    mosaic: Mosaic,
    borders_width: List[int],
    borders_weight: List[float],
) -> float:
    """Suma ponderada (borders_weight) de la suma ponderada (del área) de cada borde."""

    assert len(borders_width) == len(borders_weight)

    CC_border_list = []
    for border in borders_width:
        CC_list, weights_list = calc_metric_edge(metric_func, mosaic, border_px=border)
        CC_border = np.sum(CC_list * weights_list)
        CC_border_list.append(CC_border)

    return np.sum(np.multiply(borders_weight, CC_border_list))


# --------------------------
def calc_overlaps(masks_list: np.ndarray, seamline_px: int) -> List[np.ndarray]:
    assert len(masks_list) > 0

    fg_mask = masks_list[0]

    overlaps_list = []
    for idx in range(1, len(masks_list)):
        bg_mask = masks_list[idx]
        seamline = calc_border_of_overlap(fg_mask, bg_mask, seamline_px)
        fg_mask = np.logical_or(fg_mask, bg_mask)

        overlaps_list.append(seamline)
    return overlaps_list


def calc_zncc_on_overlaps(
    overlaps_list: List[np.ndarray],
    images_list: np.ndarray,
    masks_list: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """_summary_

    Parameters
    ----------
    mosaic : mosaic
        Mosaic to evaluate.
    border_px : int, optional
        Anchor of the edge overlap, by default 10
    min_area : int, optional
        The area must be greater than `min_area` to evaluate the correlation,
        by default 0.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Correlation list and area size of each overlap.
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


def objective_function_multi_edge_optimized(
    individual: np.ndarray, mosaic: Mosaic, borders_width: list, borders_weight: list
) -> float:
    """
    Calcula el valor de ZNCC.
    """
    current_mosaic = individual_to_mosaic(individual, mosaic)
    return multi_edge_optimized(current_mosaic, borders_width, borders_weight)


def multi_edge_optimized(
    mosaic: Mosaic, borders_width: list, borders_weight: list
) -> float:

    assert len(borders_width) == len(borders_weight)
    images_list, masks_list = get_images_and_masks_list(mosaic)

    CC_border_list = []
    for border_px in borders_width:
        overlaps_list = calc_overlaps(masks_list, border_px)
        CC_list, area_list = calc_zncc_on_overlaps(
            overlaps_list, images_list, masks_list
        )

        total_area = area_list.sum()
        if total_area == 0:
            weights_list = list(np.zeros(len(area_list)))
        else:
            weights_list = area_list / total_area

        value = np.sum(CC_list * weights_list)
        CC_border_list.append(value)

    return np.sum(np.multiply(borders_weight, CC_border_list))


def objective_function_seamline_zncc(
    individual: np.ndarray, mosaic: Mosaic, border_width: int
) -> float:
    """
    Calcula el valor de ZNCC.
    """
    current_mosaic = individual_to_mosaic(individual, mosaic)
    return seamline_zncc(current_mosaic, border_width)


def seamline_zncc(mosaic: Mosaic, border_width: int) -> float:
    images_list, masks_list = get_images_and_masks_list(mosaic)

    overlaps_list = calc_overlaps(masks_list, border_width)
    CC_list, area_list = calc_zncc_on_overlaps(overlaps_list, images_list, masks_list)

    total_area = area_list.sum()
    if total_area == 0:
        weights_list = list(np.zeros(len(area_list)))
    else:
        weights_list = area_list / total_area

    value = np.sum(CC_list * weights_list)
    return value
