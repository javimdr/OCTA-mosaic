from typing import Any, Dict, List, Tuple

import numpy as np

from octa_mosaic.modules.experiments.procedure import Procedure, Report
from octa_mosaic.modules.mosaico import Mosaico
from octa_mosaic.modules.mosaico_tm_register_MB import Mosaico_TM_register_MB


class TemplateMatchingEvaluatingEdges(Procedure):
    def _execution(
        self,
        images_list: List[np.ndarray],
        fobj_kwargs: Dict[str, Any],
    ) -> Tuple[Mosaico, Report]:

        # function = config["function"]
        border_width_list = fobj_kwargs["border_width_list"]
        border_weight_list = fobj_kwargs["border_weight_list"]

        register = Mosaico_TM_register_MB(border_width_list, border_weight_list)
        images_order, images_locations = register.edge_order(images_list)
        mosaic_tm = register.mosaico_from_indices(
            images_order, images_locations, images_list
        )
        report = self._generate_report(images_order, images_locations)
        return mosaic_tm, report

    def _generate_report(
        self, images_order: List[int], locations: List[Tuple[int, int]]
    ) -> Report:

        data = [
            {
                "image_index": index,
                "location": location,
            }
            for index, location in zip(images_order, locations)
        ]
        return {"images_order": data}

    @staticmethod
    def mosaic_from_report(images_list: List[np.ndarray], report: Report) -> Mosaico:
        mosaico = Mosaico()

        for image_data in report["images_order"]:
            image = images_list[image_data["index"]]
            location = image_data["location"]

            mosaico.add(image, location, "ij")

        return mosaico
