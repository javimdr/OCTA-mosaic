import itertools
from typing import Callable, List, Optional, Tuple

import numpy as np

from octa_mosaic.modules import optimization_utils
from octa_mosaic.modules.mosaic import Mosaic
from octa_mosaic.modules.template_matching import template_matching
from octa_mosaic.modules.utils import metrics


def get_corner(center, image_shape, corner="top_left"):
    H, W = image_shape
    top_left = np.subtract(center, [H // 2, W // 2])

    if corner == "top_left":
        return top_left
    elif corner == "top_right":
        return np.add(top_left, [0, W])
    elif corner == "bottom_left":
        return np.add(top_left, [H, 0])
    elif corner == "bottom_right":
        return np.add(top_left, [H, W])
    else:
        raise ValueError("Unknown corner.")


class mosaic_TM_register_MB:
    def __init__(self, borders_width: List[int], borders_weight: List[float]):
        self.borders_width = borders_width
        self.borders_weight = borders_weight
        assert len(borders_width) == len(borders_weight)

    def _select_first_pair(self, images_dict, mode="full", corr_func="ZNCC"):
        loop_cc = []
        loop_loc_corners = []

        images_idx_pairs = list(itertools.combinations(list(images_dict.keys()), 2))
        for index_A, index_B in images_idx_pairs:
            image_A = images_dict[index_A]
            image_B = images_dict[index_B]

            cc_value, cc_loc, cc_matrix = template_matching(
                image_A, image_B, mode=mode, corr_func=corr_func
            )
            cc_loc = cc_loc[::-1]
            if mode == "full":
                cc_loc = get_corner(cc_loc, image_B.shape[:2])

            cc_loc = get_corner(cc_loc, image_B.shape[:2])
            loop_cc.append(cc_value)
            loop_loc_corners.append(cc_loc)

        best_case_idx = np.argmax(loop_cc)

        index_A = images_idx_pairs[best_case_idx][0]
        index_B = images_idx_pairs[best_case_idx][1]
        IB_center = loop_loc_corners[best_case_idx]

        return index_A, index_B, IB_center

    def _select_next_image(self, mosaic, images_dict, mode="full", corr_func="ZNCC"):
        loop_cc = []
        loop_loc_corners = []
        images_dict_keys = list(images_dict.keys())
        for idx_image in images_dict_keys:
            image = images_dict[idx_image]
            cc_value, cc_loc, cc_matrix = template_matching(
                mosaic.image(), image, mode=mode, corr_func=corr_func
            )
            cc_loc = cc_loc[::-1]
            if mode == "full":
                cc_loc = get_corner(cc_loc, image.shape[:2])

            cc_loc = get_corner(cc_loc, image.shape[:2])
            loop_cc.append(cc_value)
            loop_loc_corners.append(cc_loc)

        best_case_idx = np.argmax(loop_cc)

        image_idx = images_dict_keys[best_case_idx]
        location = loop_loc_corners[best_case_idx]
        return image_idx, location, np.max(loop_cc)

    def create_mosaic(
        self,
        images_list: List[np.ndarray],
        metric_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ) -> Tuple[List, List]:

        if metric_func is None:
            metric_func = metrics.zncc

        images_order = []
        images_locations = []

        images_dict = {idx: image for idx, image in enumerate(images_list)}
        index_A, index_B, loc_B = self._select_first_pair(images_dict)

        image_A = images_dict[index_A]
        image_B = images_dict[index_B]

        # Check order
        mosaic_AB_bin = Mosaic()
        mosaic_AB_bin.add(image_A)
        mosaic_AB_bin.add(image_B, loc_B)
        CC_AB = optimization_utils.calc_metric_multiples_edges(
            metric_func, mosaic_AB_bin, self.borders_width, self.borders_weight
        )

        mosaic_BA_bin = Mosaic()
        mosaic_BA_bin.add(image_B)
        mosaic_BA_bin.add(image_A, -loc_B)
        CC_BA = optimization_utils.calc_metric_multiples_edges(
            metric_func, mosaic_BA_bin, self.borders_width, self.borders_weight
        )

        mosaic = Mosaic()
        if CC_AB > CC_BA:
            mosaic.add(image_A)
            mosaic.add(image_B, loc_B)
            images_order = [index_A, index_B]
            images_locations = [(0, 0), tuple(loc_B)]
        else:
            mosaic.add(image_B)
            mosaic.add(image_A, -loc_B)
            images_order = [index_B, index_A]
            images_locations = [(0, 0), tuple(-loc_B)]
        # ----

        del images_dict[index_A]
        del images_dict[index_B]

        while len(images_dict) > 0:
            image_idx, location, cc = self._select_next_image(mosaic, images_dict)
            location = tuple(location)

            images_order.append(image_idx)
            images_locations.append(location)
            mosaic.add(images_dict[image_idx], location)

            del images_dict[image_idx]

        return images_order, images_locations

    @staticmethod
    def mosaic_from_indices(images_order, locations, images_list):
        mosaic = Mosaic()

        for image_idx, location in zip(images_order, locations):
            mosaic.add(images_list[image_idx], location)

        return mosaic
