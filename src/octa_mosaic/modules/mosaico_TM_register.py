import itertools
import warnings

import numpy as np

from octa_mosaic.modules.mosaic import Mosaic
from octa_mosaic.modules.template_matching import template_matching


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


class Mosaico_TM_register:
    def cc_order(self, images_list):
        warnings.warn(
            "El orden del primer par es 'arbitrario'. Mejor usar 'cc_order_new()'."
        )
        images_order = []
        images_locations = []

        images_dict = {idx: image for idx, image in enumerate(images_list)}
        index_A, index_B, loc_B = self._select_first_pair(images_dict)

        mosaico = Mosaic()
        mosaico.add(images_dict[index_A])
        mosaico.add(images_dict[index_B], loc_B)

        images_order = [index_A, index_B]
        images_locations = [(0, 0), tuple(loc_B)]

        del images_dict[index_A]
        del images_dict[index_B]

        while len(images_dict) > 0:
            image_idx, location, cc = self._select_next_image(mosaico, images_dict)
            location = tuple(location)

            images_order.append(image_idx)
            images_locations.append(location)
            mosaico.add(images_dict[image_idx], location)

            del images_dict[image_idx]

        #         print(out_str)
        return images_order, images_locations

    def cc_order_new(self, images_list):
        images_order = []
        images_locations = []

        images_dict = {idx: image for idx, image in enumerate(images_list)}
        index_A, index_B, loc_B = self._select_first_pair(images_dict)

        mosaico_AB = Mosaic()
        mosaico_AB.add(images_dict[index_A])
        mosaico_AB.add(images_dict[index_B], loc_B)

        mosaico_BA = Mosaic()
        mosaico_BA.add(images_dict[index_B])
        mosaico_BA.add(images_dict[index_A], -loc_B)

        #          plots.plot_mult([mosaico_AB.image(), mosaico_BA.image()], cols=2)

        del images_dict[index_A]
        del images_dict[index_B]

        image_idx_AB, location_AB, cc_AB = self._select_next_image(
            mosaico_AB, images_dict
        )
        image_idx_BA, location_BA, cc_BA = self._select_next_image(
            mosaico_BA, images_dict
        )

        mosaico_AB.add(images_dict[image_idx_AB], location_AB)
        mosaico_BA.add(images_dict[image_idx_BA], location_BA)

        # plots.plot_mult([mosaico_AB.image(), mosaico_BA.image()], [f"{cc_AB:.4f}, ({cc_AB > cc_BA})", f"{cc_BA:.4f}, ({cc_BA > cc_AB})"], cols=2, base_size=10)
        out_str = ""
        if cc_AB > cc_BA:
            out_str += "Mosaico (AB)."
            images_order = [index_A, index_B, image_idx_AB]
            images_locations = [(0, 0), tuple(loc_B), tuple(location_AB)]

            del images_dict[image_idx_AB]
            mosaico = mosaico_AB

        else:
            out_str += "Mosaico (BA)."
            images_order = [index_B, index_A, image_idx_BA]
            images_locations = [(0, 0), tuple(-loc_B), tuple(location_BA)]

            del images_dict[image_idx_BA]
            mosaico = mosaico_BA

        while len(images_dict) > 0:
            image_idx, location, cc = self._select_next_image(mosaico, images_dict)
            location = tuple(location)

            images_order.append(image_idx)
            images_locations.append(location)
            mosaico.add(images_dict[image_idx], location)

            del images_dict[image_idx]

        #         print(out_str)
        return images_order, images_locations

    def _select_rest(self, images_dict, mode="full", corr_func="ZNCC"):
        pass

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

    def mosaic_from_indices(self, images_order, locations, images_list):
        mosaico = Mosaic()

        for image_idx, location in zip(images_order, locations):
            mosaico.add(images_list[image_idx], location)

        return mosaico
