from dataclasses import dataclass

import numpy as np

from octa_mosaic.modules import optimization_utils
from octa_mosaic.modules.mosaic import Mosaic


@dataclass(frozen=True)
class TransformConfig:
    translation: float
    scale: float
    rotation: float  # in degrees
    shear: float  # in degrees


@dataclass
class MosaicProblem:
    mosaic: Mosaic
    bounds: np.ndarray

    def __init__(self, mosaic: Mosaic, transform_config: TransformConfig):
        self.mosaic = mosaic
        self.n_var = mosaic.n_images() * 6

        self.bounds = self._init_bounds(transform_config)

        self.fobj = None
        self.fobj_args = None

    def _init_bounds(self, transformation_config: TransformConfig) -> np.ndarray:
        transformation_bounds = optimization_utils.affine_bounds(
            (0, 0),
            trans_bound=transformation_config.translation,
            scale_bound=transformation_config.scale,
            rot_bound=transformation_config.rotation,
            shear_bound=transformation_config.shear,
        )

        return np.tile(transformation_bounds, (self.mosaic.n_images(), 1))
