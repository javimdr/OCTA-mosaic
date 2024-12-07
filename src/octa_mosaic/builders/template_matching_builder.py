import itertools
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np

from octa_mosaic.modules import optimization_utils
from octa_mosaic.modules.mosaic import Mosaic
from octa_mosaic.modules.template_matching import template_matching
from octa_mosaic.modules.utils import metrics


class TemplateMatchingBuilder:
    def __init__(
        self,
        tm_func: Literal["ZNCC", "CV"] = "ZNCC",
        tm_mode: Literal["full", "valid"] = "full",
        first_pair_func: Callable = None,
        first_pair_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Builder for create image mosaics using template matching with an iterative
        optimization process. The algorithm aims to sequentially add images to the mosaic
        in an order that maximizes correlation, producing an optimized mosaic.

        Algorithm Overview:
            1. Calculate the correlation (`tm_func`) between all pairs of images.
            2. Select the pair with the highest correlation value.
            3. Determine the order of these two images (foreground and background)
                using `first_pair_func`. The order that maximizes the value determines
                the sequence. If `first_pair_func` is not provided, the order of the
                initial pair of images will be arbitrary, and no specific criteria
                will be applied to determine their order.
            4. Generate an initial mosaic using the selected pair of images.
            5. Iteratively, until all images are added to the mosaic:
                a. Calculate the template matching correlation (`tm_func`) between the
                    remaining images and the current mosaic (acting as the `fixed` image).
                b. Select the image with the highest correlation.
                c. Add the selected image to the mosaic background.

        Args:
            tm_func (str): The correlation function of template matching ("ZNCC" or "CV").
                Defaults to "ZNCC".
            tm_mode (str): The mode of template matching ("full" or "valid").
                Defaults to "full".
            first_pair_func (Callable, optional): function to determine the best order
                of the first image pair (higher value). The first argument to the
                function is expected to be a Mosaic. If is not provided, the order of the
                initial pair of images will be arbitrary.
                Defaults to None.
            first_pair_kwargs (dict, optional): Additional arguments for the
                first_pair_func.
                Defaults to None.
        """
        if first_pair_kwargs is None:
            first_pair_kwargs = {}

        self.tm_func = tm_func
        self.tm_mode = tm_mode
        self.first_pair_fn = first_pair_func
        self.first_pair_kwargs = first_pair_kwargs

    def _select_first_pair(
        self, images_dict: Dict[int, np.ndarray]
    ) -> Tuple[int, int, Tuple[int, int]]:
        """
        Selects the best pair of initial images based on correlation values.

        Args:
            images_dict (dict): A dictionary of image indices mapped to their respective images.

        Returns:
            Tuple[int, int, Tuple[int, int]]: The indices of the first two images and
                the top-left corner of the second image to match.
        """
        loop_cc = []
        loop_loc_corners = []

        images_idx_pairs = list(itertools.combinations(list(images_dict.keys()), 2))
        for index_A, index_B in images_idx_pairs:
            image_A = images_dict[index_A]
            image_B = images_dict[index_B]

            cc_value, cc_loc, cc_matrix = template_matching(
                image_A, image_B, mode=self.tm_mode, corr_func=self.tm_func
            )
            cc_loc = cc_loc[::-1]  # (x, y) to (i, j)
            if self.tm_mode == "full":
                cc_loc = self._get_top_left_corner(cc_loc, image_B.shape[:2])

            cc_loc = self._get_top_left_corner(cc_loc, image_B.shape[:2])
            loop_cc.append(cc_value)
            loop_loc_corners.append(cc_loc)

        best_case_idx = np.argmax(loop_cc)

        index_A = images_idx_pairs[best_case_idx][0]
        index_B = images_idx_pairs[best_case_idx][1]
        IB_center = loop_loc_corners[best_case_idx]

        return index_A, index_B, IB_center

    def _select_next_image(self, mosaic, images_dict):
        """
        Selects the next best image to add to the mosaic based on correlation.

        Args:
            mosaic (Mosaic): The current mosaic object.
            images_dict (dict): A dictionary of remaining images.

        Returns:
            Tuple[int, Tuple[int, int], float]: The index of the selected image,
                its location, and the correlation value.
        """
        loop_cc = []
        loop_loc_corners = []
        images_dict_keys = list(images_dict.keys())
        for idx_image in images_dict_keys:
            image = images_dict[idx_image]
            cc_value, cc_loc, cc_matrix = template_matching(
                mosaic.image(), image, mode=self.tm_mode, corr_func=self.tm_func
            )
            cc_loc = cc_loc[::-1]  # (x, y) to (i, j)
            if self.tm_mode == "full":
                cc_loc = self._get_top_left_corner(cc_loc, image.shape[:2])

            cc_loc = self._get_top_left_corner(cc_loc, image.shape[:2])
            loop_cc.append(cc_value)
            loop_loc_corners.append(cc_loc)

        best_case_idx = np.argmax(loop_cc)

        image_idx = images_dict_keys[best_case_idx]
        location = loop_loc_corners[best_case_idx]
        return image_idx, location, np.max(loop_cc)

    def create_mosaic(self, images_list: List[np.ndarray]) -> Mosaic:
        """
        Builds a mosaic from a list of images.

        Args:
            images_list (List[np.ndarray]): The list of images to be mosaicked.

        Returns:
            Mosaic: The resulting mosaic object.
        """
        images_order, images_locations = self.generate_mosaic_order(images_list)
        return self.mosaic_from_order(images_order, images_locations, images_list)

    def generate_mosaic_order(
        self,
        images_list: List[np.ndarray],
    ) -> Tuple[List, List]:
        """
        Generates the optimal order and locations for placing images in the mosaic.
        This function is very useful to generate reports in contrast to `create_mosaic`.

        Args:
            images_list (List[np.ndarray]): The list of images to be mosaicked
                (foreground to background).

        Returns:
            Tuple[List[int], List[Tuple[int, int]]]:
                The order of image indices and their corresponding locations.
        """
        images_order = []
        images_locations = []

        images_dict = {idx: image for idx, image in enumerate(images_list)}

        # Select first two images
        index_A, index_B, loc_B = self._select_first_pair(images_dict)
        images_order, images_locations, mosaic = self._select_order_first_pair(
            images_dict, index_A, index_B, loc_B
        )

        del images_dict[index_A]
        del images_dict[index_B]

        # Add the rest images
        while len(images_dict) > 0:
            image_idx, location, cc = self._select_next_image(mosaic, images_dict)
            location = tuple(location)

            images_order.append(image_idx)
            images_locations.append(location)
            mosaic.add(images_dict[image_idx], location)

            del images_dict[image_idx]

        return images_order, images_locations

    def _select_order_first_pair(self, images_dict, index_A, index_B, loc_B):
        """
        Determines the best order for the first pair of images. If `first_pair_func` is
        None, the order of the initial pair of images will be Image A for foreground and
        B for background.

        Args:
            images_dict (dict): A dictionary of image indices mapped to their respective images.
            index_A (int): The index of the first image.
            index_B (int): The index of the second image.
            loc_B (Tuple[int, int]): The initial location of the second image.

        Returns:
            Tuple[List[int], List[Tuple[int, int]], Mosaic]:
                The order of the first pair, their locations, and the initialized mosaic.
        """
        image_A = images_dict[index_A]
        image_B = images_dict[index_B]

        # Generates AB-Mosaic (A: foreground, B: Background)
        mosaic_AB = Mosaic()
        mosaic_AB.add(image_A)
        mosaic_AB.add(image_B, loc_B)
        images_order_AB = [index_A, index_B]
        images_locations_AB = [(0, 0), tuple(loc_B)]
        if self.first_pair_fn is None:
            # If there is no criteria, return this mosaic
            return images_order_AB, images_locations_AB, mosaic_AB

        # Generates BA-Mosaic (B: foreground, A: Background)
        mosaic_BA = Mosaic()
        mosaic_BA.add(image_B)
        mosaic_BA.add(image_A, -loc_B)
        images_order_BA = [index_B, index_A]
        images_locations_BA = [(0, 0), tuple(-loc_B)]

        # Calculates criteria value (greater is better)
        value_AB_option = self.first_pair_fn(mosaic_AB, **self.first_pair_kwargs)
        value_BA_option = self.first_pair_fn(mosaic_BA, **self.first_pair_kwargs)

        if value_AB_option > value_BA_option:
            return images_order_AB, images_locations_AB, mosaic_AB
        else:
            return images_order_BA, images_locations_BA, mosaic_BA

    @staticmethod
    def mosaic_from_order(
        images_order: List[int],
        locations: List[Tuple[int, int]],
        images_list: List[np.ndarray],
    ) -> Mosaic:
        """
        Constructs a mosaic from a given image order and their respective locations.

        Args:
            images_order (List[int]): The ordered list of image indices.
            locations (List[Tuple[int, int]]): The locations for placing the images.
            images_list (List[np.ndarray]): The list of images to be placed.

        Returns:
            Mosaic: The resulting mosaic object.
        """
        mosaic = Mosaic()

        for image_idx, location in zip(images_order, locations):
            mosaic.add(images_list[image_idx], location)

        return mosaic

    @staticmethod
    def _get_top_left_corner(
        center: Tuple[int, int], image_shape: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Converts a center location to the top-left corner based on image dimensions.

        Args:
            center (Tuple[int, int]): The center coordinates.
            image_shape (Tuple[int, int]): The height and width of the image.

        Returns:
            Tuple[int, int]: The top-left corner coordinates.
        """
        H, W = image_shape
        return np.subtract(center, [H // 2, W // 2])
