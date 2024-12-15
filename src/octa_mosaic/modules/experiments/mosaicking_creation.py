from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from octa_mosaic import Mosaic, TemplateMatchingBuilder
from octa_mosaic.modules.experiments.procedure import Procedure, Report


class TemplateMatchingEvaluatingEdges(Procedure):
    def _execution(
        self,
        images_list: List[np.ndarray],
        first_pair_func: Callable = None,
        first_pair_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Mosaic, Report]:

        register = TemplateMatchingBuilder(
            first_pair_func=first_pair_func,
            first_pair_kwargs=first_pair_kwargs,
        )
        images_order, images_locations = register.generate_mosaic_order(images_list)
        mosaic_tm = register.mosaic_from_order(
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
    def mosaic_from_report(images_list: List[np.ndarray], report: Report) -> Mosaic:
        mosaic = Mosaic()

        for image_data in report["images_order"]:
            image = images_list[image_data["index"]]
            location = image_data["location"]

            mosaic.add(image, location, "ij")

        return mosaic
