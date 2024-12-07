from typing import Callable, List, Optional, Tuple

import numpy as np

from octa_mosaic.modules import optimization_utils
from octa_mosaic.modules.mosaic import Mosaic
from octa_mosaic.modules.mosaico_TM_register import Mosaico_TM_register
from octa_mosaic.modules.utils import metrics, plots


class Mosaico_TM_register_MB(Mosaico_TM_register):
    def __init__(self, borders_width: List[int], borders_weight: List[float]):
        self.borders_width = borders_width
        self.borders_weight = borders_weight
        assert len(borders_width) == len(borders_weight)

    def edge_order(
        self,
        images_list: List[np.ndarray],
        metric_func: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        plot_first_pair: bool = False,
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
        mosaico_AB_bin = Mosaic()
        mosaico_AB_bin.add(image_A)
        mosaico_AB_bin.add(image_B, loc_B)
        CC_AB = optimization_utils.calc_metric_multiples_edges(
            metric_func, mosaico_AB_bin, self.borders_width, self.borders_weight
        )

        mosaico_BA_bin = Mosaic()
        mosaico_BA_bin.add(image_B)
        mosaico_BA_bin.add(image_A, -loc_B)
        CC_BA = optimization_utils.calc_metric_multiples_edges(
            metric_func, mosaico_BA_bin, self.borders_width, self.borders_weight
        )

        if plot_first_pair:
            print(f"Location: {loc_B}")
            plots.plot_mult(
                [
                    plots.impair(
                        mosaico_AB_bin.image_with_order([0]),
                        mosaico_AB_bin.image_with_order([1]),
                    ),
                    plots.impair(
                        mosaico_BA_bin.image_with_order([0]),
                        mosaico_BA_bin.image_with_order([1]),
                    ),
                ],
                [f"{CC_AB : 0.4f}", f"{CC_BA : 0.4f}"],
                cols=2,
                base_size=10,
            )

        mosaico = Mosaic()
        if CC_AB > CC_BA:
            mosaico.add(image_A)
            mosaico.add(image_B, loc_B)
            images_order = [index_A, index_B]
            images_locations = [(0, 0), tuple(loc_B)]
        else:
            mosaico.add(image_B)
            mosaico.add(image_A, -loc_B)
            images_order = [index_B, index_A]
            images_locations = [(0, 0), tuple(-loc_B)]
        # ----

        del images_dict[index_A]
        del images_dict[index_B]

        while len(images_dict) > 0:
            image_idx, location, cc = self._select_next_image(mosaico, images_dict)
            location = tuple(location)

            images_order.append(image_idx)
            images_locations.append(location)
            mosaico.add(images_dict[image_idx], location)

            del images_dict[image_idx]

        return images_order, images_locations
