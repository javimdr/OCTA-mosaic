import copy
from datetime import date
from typing import Callable, List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage.transform import AffineTransform

from octa_mosaic.modules.evolutionary import init_population_lhs
from octa_mosaic.modules.mosaic import Mosaic
from octa_mosaic.modules.utils import metrics


class ExperimentResult:
    def __init__(self, title_test, config_dict, results_dict, info={}):
        self.title = title_test
        self.configuration = copy.deepcopy(config_dict)
        self.date = date.today().strftime("%d/%m/%Y")
        self.results_dict = copy.deepcopy(results_dict)
        self.info = info

    def __repr__(self):
        return f"ExperimentResult({self.title})"

    def __str__(self):
        return self.__repr__()


def plot_sol(sol, ax=None, y_min=0, y_max=1):
    with sns.axes_style("whitegrid"):

        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(12, 6))

        x = np.arange(len(sol.fitness_record["pop_mean"]))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        ax.plot(x, sol.fitness_record["best"], label="Best", color=colors[0], zorder=3)

        mean = np.array(sol.fitness_record["pop_mean"])
        std = np.array(sol.fitness_record["pop_std"])
        ax.fill_between(x, mean - std, mean + std, alpha=0.3, color=colors[1], zorder=1)  # type: ignore
        ax.plot(x, mean, label="Mean", color=colors[1], zorder=2)
        ax.legend()

        ax.set_ylim(ymin=y_min, ymax=y_max)
        ax.set_xlabel("Generations")
        ax.set_ylabel("Fitness")
        ax.set_title(
            f'Best: {sol.fitness_record["best"][-1] : 0.4f}. Pop: {sol.fitness_record["pop_mean"][-1] : 0.4f} ± {sol.fitness_record["pop_std"][-1] : 0.4f}'
        )
        return ax


# ======== Funciones ayuda de las funciones objetivo ========


def affine_bounds(
    base_loc: Sequence,  # base_scale: Sequence=(1,1),
    trans_bound: float = 30,
    scale_bound: float = 0.05,
    rot_bound: float = 15.0,
    shear_bound: float = 5.0,
):
    """Bounds of the individuals

    Parameters
    ----------
    base_loc : Sequence
        The translation bounds are applied around this location.
    trans_bound : int, optional
        Translation bounds, by default $\\pm$30 pixels.
    scale_bound : float, optional
        Scale bounds, by default $\\pm$0.05 (5%).
    rot_bound : float, optional
        Rotation bounds, by default $\\pm$15 degrees.
    shear_bound : float, optional
        Shear bounds, by default $\\pm$5 degrees.

    Returns
    -------
    Sequence 6x2
        Upper and lower bounds for eache freedom degree:
        (tx, ty, sx, sy, rot, shear)
    """
    x, y = base_loc
    sx, sy = (1, 1)  # base_scale

    translation_x_bounds = (x - trans_bound, x + trans_bound)
    translation_y_bounds = (y - trans_bound, y + trans_bound)
    scale_x_bounds = (sx - scale_bound, sx + scale_bound)
    scale_y_bounds = (sy - scale_bound, sy + scale_bound)
    rotation_bounds = (np.deg2rad(-rot_bound), np.deg2rad(rot_bound))
    shear_bounds = (np.deg2rad(-shear_bound), np.deg2rad(shear_bound))

    return np.array(
        [
            translation_x_bounds,
            translation_y_bounds,
            scale_x_bounds,
            scale_y_bounds,
            rotation_bounds,
            shear_bounds,
        ]
    ).astype(float)


def initialize_population(n_images, popsize, bounds, seed=None):
    tm_individual_norm = [0.5 for _ in range(6 * n_images)]
    pop_norm_lhc = init_population_lhs(popsize, 6 * n_images, seed)
    pop_norm = np.append(pop_norm_lhc, tm_individual_norm)
    pop_norm = pop_norm.reshape(popsize + 1, 6 * n_images)
    return pop_norm


def tf_repr_to_tf_matrix(tf_repr) -> AffineTransform:
    assert len(tf_repr) == 6, f"{tf_repr}"
    T = AffineTransform(
        translation=tf_repr[:2],
        scale=tf_repr[2:4],
        rotation=tf_repr[4],
        shear=tf_repr[5],
    )
    # print(np.round(T.params, 2))
    return T


def tf_matrix_to_repr(matrix: AffineTransform) -> np.ndarray:
    tx, ty = matrix.translation
    sx, sy = matrix.scale
    rot = matrix.rotation
    shear = matrix.shear

    return np.array([tx, ty, sx, sy, rot, shear])


def tf_matrix_to_tf_repr(tf_matrix: AffineTransform) -> np.ndarray:
    """Angles expressed in radians."""
    if not isinstance(tf_matrix, AffineTransform):
        raise ValueError()

    tx, ty = tf_matrix.translation
    sx, sy = tf_matrix.scale
    rot = tf_matrix.rotation
    shear = tf_matrix.shear

    return np.array([tx, ty, sx, sy, rot, shear], "float32")


def individual_to_mosaico(individual: np.ndarray, mosaico: Mosaic) -> Mosaic:
    n_images = len(mosaico.images_list)
    new_mosaico = mosaico.copy()

    tfs_list = [
        tf_repr_to_tf_matrix(individual[idx * 6 : (idx * 6) + 6])
        for idx in range(n_images)
    ]
    new_mosaico.set_transforms_list(tfs_list)

    return new_mosaico


def get_images_and_masks_list(mosaico: Mosaic):
    n_images = mosaico.n_images()
    mosaico_size = mosaico.mosaic_size

    masks_list = np.zeros((n_images, *mosaico_size), bool)
    images_list = np.zeros((n_images, *mosaico_size))

    for idx in range(n_images):
        image, _, _ = mosaico.get_image_data(idx)
        tf = mosaico.image_centered_transform(idx)

        mask_tf = cv2.warpPerspective(np.ones_like(image), tf.params, mosaico_size[::-1])
        image_tf = cv2.warpPerspective(image, tf.params, mosaico_size[::-1])

        masks_list[idx] = mask_tf.astype(bool)
        images_list[idx] = image_tf

    return images_list, masks_list


def erode_mask(mask, pixels=10):
    k_size = pixels * 2 + 1
    kernel = np.ones((k_size, k_size))

    return cv2.erode(mask.astype("float32"), kernel, iterations=1).astype(bool)


def dilate_mask(mask, pixels=10):
    k_size = pixels * 2 + 1
    kernel = np.ones((k_size, k_size))

    return cv2.dilate(mask.astype("float32"), kernel, iterations=1).astype(bool)


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
    mosaico: Mosaic, border_px: int = 10, min_area: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """_summary_

    Parameters
    ----------
    mosaico : Mosaico
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
    images_list, masks_list = get_images_and_masks_list(mosaico)

    fg_mask = masks_list[0]
    fg_image = images_list[0]
    CC_list = np.zeros(mosaico.n_images() - 1)
    border_size_list = np.zeros(mosaico.n_images() - 1)

    for idx in range(1, mosaico.n_images()):
        bg_mask = masks_list[idx]
        bg_image = images_list[idx]

        # Calculate CC
        border_iou = calc_border_of_overlap(fg_mask, bg_mask, border_px)
        border_size = np.count_nonzero(border_iou)
        border_size_list[idx - 1] = border_size

        if border_size > min_area:
            CC_list[idx - 1] = metrics.zncc(
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


def func_objetivo_borde(individual, mosaico, border_px=10, boder_size_th=0):
    """Función base. NO USAR. Devuelve una lista de correlaciones y sus pesos asociados"""
    curr_mosaico = individual_to_mosaico(individual, mosaico)
    return calc_CC_on_overlaps_and_areas(curr_mosaico, border_px, boder_size_th)


def func_objetivo_borde_sum(individual, mosaico, border_px=10):
    CC_list, weights_list = func_objetivo_borde(individual, mosaico, border_px)
    return np.sum(CC_list)


def func_objetivo_borde_prod(individual, mosaico, border_px=10):
    CC_list, weights_list = func_objetivo_borde(individual, mosaico, border_px)
    return np.prod(CC_list)


def func_objetivo_borde_weighted_sum(individual, mosaico, border_px=10):
    CC_list, weights_list = func_objetivo_borde(individual, mosaico, border_px)
    return np.sum(CC_list * weights_list)


def objective_function_multi_edge(
    individual: np.ndarray, mosaico: Mosaic, borders_width: list, borders_weight: list
) -> float:
    """
    Calcula el valor de ZNCC.
    """
    assert len(borders_width) == len(borders_weight)

    CC_border_list = [
        func_objetivo_borde_weighted_sum(individual, mosaico, border_px=border)
        for border in borders_width
    ]

    return np.sum(np.multiply(borders_weight, CC_border_list))


### Geometric Mean


def objective_function_edge_geometric_mean(individual, mosaico, border_px=10):
    CC_list, _ = func_objetivo_borde(individual, mosaico, border_px)
    CC_list_corrected = np.where(CC_list > 0, CC_list, 1e-5)

    n = len(CC_list_corrected)
    assert n > 0, "No hay ningún valor de correlación"
    geometric_mean = np.prod(CC_list_corrected) ** (1 / n)
    return geometric_mean


def objective_function_multi_edge_geometric_mean(
    individual: np.ndarray, mosaico: Mosaic, borders_width: list, borders_weight: list
) -> float:
    """
    Calcula el valor de ZNCC.
    """
    assert len(borders_width) == len(borders_weight)

    CC_border_list = [
        objective_function_edge_geometric_mean(individual, mosaico, border_px=border)
        for border in borders_width
    ]

    return np.sum(np.multiply(borders_weight, CC_border_list))


# Geometric mean v2


def objective_function_edge_geometric_mean_v2(individual, mosaico, border_px=10):
    CC_list, _ = func_objetivo_borde(individual, mosaico, border_px)
    n = len(CC_list)
    assert n > 0, "No hay ningún valor de correlación."

    CC_list = np.array(CC_list)
    if np.any(CC_list < 0):
        return -1

    geometric_mean = np.prod(CC_list) ** (1 / n)
    return geometric_mean


def objective_function_multi_edge_geometric_mean_v2(
    individual: np.ndarray, mosaico: Mosaic, borders_width: list, borders_weight: list
) -> float:
    """
    Calcula el valor de ZNCC.
    """
    assert len(borders_width) == len(borders_weight)

    CC_border_list = [
        objective_function_edge_geometric_mean_v2(individual, mosaico, border_px=border)
        for border in borders_width
    ]

    return np.sum(np.multiply(borders_weight, CC_border_list))


# =========== DICE ===================
def calc_dice_and_areas(mosaico, border_px=10, boder_size_th=0):
    images_list, masks_list = get_images_and_masks_list(mosaico)

    fg_mask = masks_list[0]
    fg_image = images_list[0]
    CC_list = np.zeros(mosaico.n_images() - 1)
    border_size_list = np.zeros(mosaico.n_images() - 1)

    for idx in range(1, mosaico.n_images()):
        bg_mask = masks_list[idx]
        bg_image = images_list[idx]

        # Calculate CC
        border_iou = calc_border_of_overlap(fg_mask, bg_mask, border_px)
        boder_size = np.count_nonzero(border_iou)
        border_size_list[idx - 1] = boder_size

        if boder_size > boder_size_th:
            CC_list[idx - 1] = metrics.zncc(
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


def func_objetivo_edge_dice(individual, base_mosaico, border_px=10):
    individual_mosaico = individual_to_mosaico(individual, base_mosaico)
    DICE_list, weights_list = calc_dice_and_areas(individual_mosaico, border_px)
    return np.sum(DICE_list * weights_list)


# ===========


def calc_metric_edge(
    metric_func: Callable[[np.ndarray, np.ndarray], float],
    mosaico: Mosaic,
    border_px: int = 10,
    boder_size_th: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:

    images_list, masks_list = get_images_and_masks_list(mosaico)

    fg_mask = masks_list[0]
    fg_image = images_list[0]
    metric_list = np.zeros(mosaico.n_images() - 1)
    border_size_list = np.zeros(mosaico.n_images() - 1)

    for idx in range(1, mosaico.n_images()):
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
    mosaico: Mosaic,
    borders_width: List[int],
    borders_weight: List[float],
    boder_size_th: int = 0,
) -> float:
    """Suma ponderada (borders_weight) de la suma ponderada (del área) de cada borde."""

    assert len(borders_width) == len(borders_weight)

    CC_border_list = []
    for border in borders_width:
        CC_list, weights_list = calc_metric_edge(metric_func, mosaico, border_px=border)
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
    mosaico : Mosaico
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
            CC_list[idx - 1] = metrics.zncc(
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


# TODO: Revisar
def objective_function_multi_edge_optimized(
    individual: np.ndarray, mosaico: Mosaic, borders_width: list, borders_weight: list
) -> float:
    """
    Calcula el valor de ZNCC.
    """
    current_mosaico = individual_to_mosaico(individual, mosaico)
    return multi_edge_optimized(current_mosaico, borders_width, borders_weight)


def multi_edge_optimized(
    mosaico: Mosaic, borders_width: list, borders_weight: list
) -> float:

    assert len(borders_width) == len(borders_weight)
    images_list, masks_list = get_images_and_masks_list(mosaico)

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
    individual: np.ndarray, mosaico: Mosaic, border_width: int
) -> float:
    """
    Calcula el valor de ZNCC.
    """
    current_mosaico = individual_to_mosaico(individual, mosaico)
    return seamline_zncc(current_mosaico, border_width)


def seamline_zncc(mosaico: Mosaic, border_width: int) -> float:
    images_list, masks_list = get_images_and_masks_list(mosaico)

    overlaps_list = calc_overlaps(masks_list, border_width)
    CC_list, area_list = calc_zncc_on_overlaps(overlaps_list, images_list, masks_list)

    total_area = area_list.sum()
    if total_area == 0:
        weights_list = list(np.zeros(len(area_list)))
    else:
        weights_list = area_list / total_area

    value = np.sum(CC_list * weights_list)
    return value


# import numpy as np
# import cv2
# from skimage.transform import AffineTransform
# from .mosaico import Mosaico, RigidMosaico
# from .evolutionary import init_population_lhs
# from mosaicking.modules.utils import metrics, plots
# import copy
# from datetime import date
# import matplotlib.pyplot as plt
# import seaborn as sns
# from typing import Sequence
# from typing import List, Tuple, Callable

# class ExperimentResult:
#     def __init__(self, title_test, config_dict, results_dict, info={}):
#         self.title = title_test
#         self.configuration = copy.deepcopy(config_dict)
#         self.date = date.today().strftime("%d/%m/%Y")
#         self.results_dict = copy.deepcopy(results_dict)
#         self.info = info

#     def __repr__(self):
#         return f"ExperimentResult({self.title})"

#     def __str__(self):
#         return self.__repr__()


# def plot_sol(sol, ax=None, y_min=0, y_max=1):
#     with sns.axes_style("whitegrid"):

#         if ax is None:
#             f, ax = plt.subplots(1,1, figsize=(12, 6))

#         x = np.arange(len(sol.fitness_record['pop_mean']))
#         colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

#         ax.plot(x, sol.fitness_record['best'], label="Best", color=colors[0], zorder=3)

#         mean = np.array(sol.fitness_record['pop_mean'])
#         std = np.array(sol.fitness_record['pop_std'])
#         ax.fill_between(x, mean - std, mean + std, alpha=0.3, color=colors[1], zorder=1)
#         ax.plot(x, mean, label="Mean", color=colors[1], zorder=2)
#         ax.legend()

#         ax.set_ylim(ymin=y_min, ymax=y_max)
#         ax.set_xlabel("Generations")
#         ax.set_ylabel("Fitness")
#         return ax

# # ======== Funciones ayuda de las funciones objetivo ========

# def affine_bounds(base_loc: Sequence, #base_scale: Sequence=(1,1),
#                   trans_bound: int=30, scale_bound: float=0.05,
#                   rot_bound: float=15., shear_bound: float=5.):
#     """Bounds of the individuals

#     Parameters
#     ----------
#     base_loc : Sequence
#         The translation bounds are applied around this location.
#     trans_bound : int, optional
#         Translation bounds, by default $\pm$30 pixels.
#     scale_bound : float, optional
#         Scale bounds, by default $\pm$0.05 (5%).
#     rot_bound : float, optional
#         Rotation bounds, by default $\pm$15 degrees.
#     shear_bound : float, optional
#         Shear bounds, by default $\pm$5 degrees.

#     Returns
#     -------
#     Sequence 6x2
#         Upper and lower bounds for eache freedom degree:
#         (tx, ty, sx, sy, rot, shear)
#     """
#     x, y = base_loc
#     sx, sy = (1, 1) #base_scale

#     translation_x_bounds = (x -trans_bound, x +trans_bound)
#     translation_y_bounds = (y -trans_bound, y +trans_bound)
#     scale_x_bounds =       (sx - scale_bound, sx + scale_bound)
#     scale_y_bounds =       (sy - scale_bound, sy + scale_bound)
#     rotation_bounds =      (np.deg2rad(-rot_bound),    np.deg2rad(rot_bound))
#     shear_bounds =         (np.deg2rad(-shear_bound),  np.deg2rad(shear_bound))

#     return np.array([translation_x_bounds, translation_y_bounds,
#                      scale_x_bounds, scale_y_bounds,
#                      rotation_bounds,
#                      shear_bounds
#                     ]).astype(float)

# def initialize_population(n_images, popsize, bounds, seed=None):
#     tm_individual_norm = [0.5 for _ in range(6*n_images)]
#     pop_norm_lhc = init_population_lhs(popsize, 6*n_images, seed)
#     pop_norm = np.append(pop_norm_lhc, tm_individual_norm)
#     pop_norm = pop_norm.reshape(popsize + 1, 6*n_images)
#     return pop_norm

# def tf_repr_to_tf_matrix(tf_repr):
#     assert len(tf_repr) == 6, f"{tf_repr}"
#     T = AffineTransform(translation=tf_repr[:2],
#                         scale=      tf_repr[2:4],
#                         rotation=   tf_repr[4],
#                         shear=      tf_repr[5])
#     #print(np.round(T.params, 2))
#     return T

# def individual_to_mosaico(individual, mosaico):
#     n_images = len( mosaico.images_list ) ##
#     new_mosaico = RigidMosaico()

#     for idx in range(n_images):
#         theta = tf_repr_to_tf_matrix(individual[idx*6:(idx*6)+6])
#         image, translation = mosaico.get_image_data(idx)
#         new_mosaico.add(image, translation, theta)

#     return new_mosaico

# def get_images_and_masks_list(mosaico):
#     n_images = mosaico.n_images()
#     mosaico_size = mosaico.mosaic_size

#     masks_list = np.zeros((n_images, *mosaico_size), bool)
#     images_list = np.zeros((n_images, *mosaico_size))

#     for idx in range(n_images):
#         if isinstance(mosaico, RigidMosaico):
#             image, _, _ = mosaico.get_image_data(idx)
#             tf = mosaico.image_centered_transform(idx)
#         else:
#             image, tf = mosaico.get_image_data(idx)

#         mask_tf = cv2.warpPerspective(np.ones_like(image), tf.params, mosaico_size[::-1])
#         image_tf = cv2.warpPerspective(image, tf.params, mosaico_size[::-1])

#         masks_list[idx] = mask_tf.astype(bool)
#         images_list[idx] = image_tf

#     return images_list, masks_list

# def erode_mask(mask, pixels=10):
#     k_size = pixels*2+1
#     kernel = np.ones((k_size, k_size))

#     return cv2.erode(mask.astype("float32"), kernel, iterations=1).astype(bool)

# def dilate_mask(mask, pixels=10):
#     k_size = pixels*2+1
#     kernel = np.ones((k_size, k_size))

#     return cv2.dilate(mask.astype("float32"), kernel, iterations=1).astype(bool)

# def calc_boder_of_overlap(fg, bg, border_px=10):
#     """
#     fg : foreground image
#     bg : background image
#     """

#     overlap = np.logical_and(fg, bg)
#     bg_non_overlaped_zone = np.logical_xor(bg, overlap)
#     bg_non_overlaped_zone_dilated = dilate_mask(bg_non_overlaped_zone, border_px)

#     return np.logical_and(overlap, bg_non_overlaped_zone_dilated)


# # ======== Funciones objetivo ========

# def func_objetivo_borde(individual, mosaico, border_px=10, boder_size_th=0):
#     """ Función base. NO USAR. Devuelve una lista de correlaciones y sus pesos asociados"""
#     curr_mosaico = individual_to_mosaico(individual, mosaico)
#     images_list, masks_list = get_images_and_masks_list(curr_mosaico)

#     fg_mask = masks_list[0]
#     fg_image = images_list[0]
#     CC_list = np.zeros(mosaico.n_images() - 1)
#     border_size_list = np.zeros(mosaico.n_images() - 1)

#     for idx in range(1, mosaico.n_images()):
#         bg_mask = masks_list[idx]
#         bg_image = images_list[idx]


#         # Calculate CC
#         border_iou = calc_boder_of_overlap(fg_mask, bg_mask, border_px)
#         boder_size = np.count_nonzero(border_iou)
#         border_size_list[idx-1] = boder_size

#         if boder_size > boder_size_th:
#             CC_list[idx-1] = metrics.zncc(fg_image[border_iou].ravel(), bg_image[border_iou].ravel())
#         else:  # boder_size == 0:
#             CC_list[idx-1] = 0

#         #plots.plot_mult([plots.impair(fg_image, bg_image), border_iou], [f"CC: {CC_list[idx-1]:.4f}", f"Area: {border_size_list[idx-1]}"],cols=2, base_size=10)
#         fg_image = np.where((~fg_mask) & (bg_mask), bg_image, fg_image)
#         fg_mask = np.logical_or(fg_mask, bg_mask)

#     sum_borders = border_size_list.sum()
#     weights_list = np.array([border_size / sum_borders for border_size in border_size_list])

#     return CC_list, weights_list

# def func_objetivo_borde_sum(individual, mosaico, border_px=10):
#     CC_list, weights_list = func_objetivo_borde(individual, mosaico, border_px)
#     return np.sum(CC_list)

# def func_objetivo_borde_prod(individual, mosaico, border_px=10):
#     CC_list, weights_list = func_objetivo_borde(individual, mosaico, border_px)
#     return np.prod(CC_list)

# def func_objetivo_borde_weighted_sum(individual, mosaico, border_px=10):
#     CC_list, weights_list = func_objetivo_borde(individual, mosaico, border_px)
#     return np.sum(CC_list * weights_list)


# def objective_function_multi_edge(
#         individual: np.ndarray,
#         mosaico: Mosaico,
#         borders_width: list,
#         borders_weight: list
#     ) -> float:
#     """
#     Calcula el valor de ZNCC.
#     """
#     assert len(borders_width) == len(borders_weight)

#     CC_border_list = [func_objetivo_borde_weighted_sum(
#                         individual, mosaico, border_px=border
#                       ) for border in borders_width]

#     return np.sum(np.multiply(borders_weight, CC_border_list))


# ### Geometric Mean v1.

# def objective_function_edge_geometric_mean(individual, mosaico, border_px=10):
#     CC_list, _ = func_objetivo_borde(individual, mosaico, border_px)
#     CC_list_corrected = np.where(CC_list > 0, CC_list, 1e-5)

#     n = len(CC_list_corrected)
#     assert n > 0, "No hay ningún valor de correlación"
#     geometric_mean = np.prod(CC_list_corrected) ** (1/n)
#     return geometric_mean

# def objective_function_multi_edge_geometric_mean(
#         individual: np.ndarray,
#         mosaico: Mosaico,
#         borders_width: list,
#         borders_weight: list
#     ) -> float:
#     """
#     Calcula el valor de ZNCC.
#     """
#     assert len(borders_width) == len(borders_weight)

#     CC_border_list = [
#         objective_function_edge_geometric_mean(individual, mosaico, border_px=border)
#         for border in borders_width
#     ]

#     return np.sum(np.multiply(borders_weight, CC_border_list))

# # Geometric mean v2

# def objective_function_edge_geometric_mean_v2(individual, mosaico, border_px=10):
#     CC_list, _ = func_objetivo_borde(individual, mosaico, border_px)
#     n = len(CC_list)
#     assert n > 0, "No hay ningún valor de correlación."

#     CC_list = np.array(CC_list)
#     if np.any(CC_list < 0):
#         return -1

#     geometric_mean = np.prod(CC_list) ** (1/n)
#     return geometric_mean

# def objective_function_multi_edge_geometric_mean_v2(
#         individual: np.ndarray,
#         mosaico: Mosaico,
#         borders_width: list,
#         borders_weight: list
#     ) -> float:
#     """
#     Calcula el valor de ZNCC.
#     """
#     assert len(borders_width) == len(borders_weight)

#     CC_border_list = [
#         objective_function_edge_geometric_mean_v2(individual, mosaico, border_px=border)
#         for border in borders_width
#     ]

#     return np.sum(np.multiply(borders_weight, CC_border_list))

# # =========== DICE ===================
# def calc_dice_and_areas(mosaico, border_px=10, boder_size_th=0):
#     images_list, masks_list = get_images_and_masks_list(mosaico)

#     fg_mask = masks_list[0]
#     fg_image = images_list[0]
#     CC_list = np.zeros(mosaico.n_images() - 1)
#     border_size_list = np.zeros(mosaico.n_images() - 1)

#     for idx in range(1, mosaico.n_images()):
#         bg_mask = masks_list[idx]
#         bg_image = images_list[idx]


#         # Calculate CC
#         border_iou = calc_boder_of_overlap(fg_mask, bg_mask, border_px)
#         boder_size = np.count_nonzero(border_iou)
#         border_size_list[idx-1] = boder_size

#         if boder_size > boder_size_th:
#             CC_list[idx-1] = metrics.zncc(fg_image[border_iou].ravel(), bg_image[border_iou].ravel())

#         #plots.plot_mult([plots.impair(fg_image, bg_image), border_iou], [f"CC: {CC_list[idx-1]:.4f}", f"Area: {border_size_list[idx-1]}"],cols=2, base_size=10)
#         fg_image = np.where((~fg_mask) & (bg_mask), bg_image, fg_image)
#         fg_mask = np.logical_or(fg_mask, bg_mask)

#     sum_borders = border_size_list.sum()
#     weights_list = np.array([border_size / sum_borders for border_size in border_size_list])

#     return CC_list, weights_list

# def func_objetivo_edge_dice(individual, base_mosaico, border_px=10):
#     individual_mosaico = individual_to_mosaico(individual, base_mosaico)
#     DICE_list, weights_list = calc_dice_and_areas(individual_mosaico, border_px)
#     return np.sum(DICE_list * weights_list)


# #===========

# def calc_metric_edge(
#         metric_func: Callable[[np.ndarray, np.ndarray], float],
#         mosaico: Mosaico,
#         border_px: int = 10,
#         boder_size_th: int = 0
#     ) -> float:

#     images_list, masks_list = optimization_utils.get_images_and_masks_list(mosaico)

#     fg_mask = masks_list[0]
#     fg_image = images_list[0]
#     metric_list = np.zeros(mosaico.n_images() - 1)
#     border_size_list = np.zeros(mosaico.n_images() - 1)

#     for idx in range(1, mosaico.n_images()):
#         bg_mask = masks_list[idx]
#         bg_image = images_list[idx]

#         # Calculate CC
#         border_iou = optimization_utils.calc_boder_of_overlap(fg_mask, bg_mask, border_px)
#         boder_size = np.count_nonzero(border_iou)
#         border_size_list[idx-1] = boder_size


#         if boder_size > boder_size_th:
#             metric_list[idx-1] = metric_func(fg_image[border_iou].ravel(), bg_image[border_iou].ravel())

#         #plots.plot_mult([plots.impair(fg_image, bg_image), border_iou], [f"CC: {CC_list[idx-1]:.4f}", f"Area: {border_size_list[idx-1]}"],cols=2, base_size=10)
#         fg_image = np.where((~fg_mask) & (bg_mask), bg_image, fg_image)
#         fg_mask = np.logical_or(fg_mask, bg_mask)

#     sum_borders = border_size_list.sum()
#     weights_list = np.array([border_size / sum_borders for border_size in border_size_list])

#     return metric_list, weights_list

# def calc_metric_multiples_edges(
#         metric_func: Callable[[np.ndarray, np.ndarray], float],
#         mosaico: Mosaico,
#         borders_width: List[int],
#         borders_weight: List[float],
#         boder_size_th: int = 0
#     ) -> float:

#     assert len(borders_width) == len(borders_weight)

#     CC_border_list = []
#     for border in borders_width:
#         CC_list, weights_list = calc_metric_edge(metric_func, mosaico, border_px=border)
#         CC_border = np.sum(CC_list * weights_list)
#         CC_border_list.append(CC_border)

#     return np.sum(np.multiply(borders_weight, CC_border_list))
