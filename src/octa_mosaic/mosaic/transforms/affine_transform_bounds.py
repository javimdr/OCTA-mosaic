from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class AffineTransformBounds:
    """Defines and calculates bounds for affine transformation parameters.

    This class encapsulates the configuration and computation of affine transformation
    bounds, including translation, scaling, rotation, and shear.

    Attributes:
        translation (float): Translation bounds in pixels for both `x` and `y`.
            By default, ±0.
        scale (float): Scale bounds as a fraction of the base scale for both `x` and `y`.
            By default, ±0.
        rotation (float): Rotation bounds in degrees. By default, ±0.
        shear (float): Shear bounds in degrees. By default, ±0.
    """

    translation: float = 0.0
    scale: float = 0.0
    rotation: float = 0.0  # in degrees
    shear: float = 0.0  # in degrees

    def compute_bounds(
        self,
        base_translation: Optional[Tuple[float, float]] = None,
        in_radians: bool = True,
    ) -> np.ndarray:
        """Calculates the upper and lower bounds for each affine transformation parameter.

        The bounds are computed as follows:
        - Translation bounds are applied around the `base_translation`. By default (0, 0).
        - Scaling bounds are applied around a default scale of 1.
        - Rotation and shear bounds are applied in radians.

        Args:
            base_translation (Tuple[float, float], optional): The base translation
                (tx, ty) for applying the translation bounds. Defaults to (0.0, 0.0).
            in_radians (bool): If `True`, rotation and shear bounds are returned
                in radians. If `False`, they are returned in degrees. Defaults to `True`.

        Returns:
            np.ndarray: A 6x2 array of bounds for affine transformation parameters in the
                following order: (tx, ty, sx, sy, rot, shear). Each row contains the
                lower and upper bounds for the respective parameter.
        """
        if base_translation is None:
            base_translation = (0.0, 0.0)

        tx, ty = base_translation
        sx, sy = (1.0, 1.0)  # Default scale

        translation_x_bounds = (tx - self.translation, tx + self.translation)
        translation_y_bounds = (ty - self.translation, ty + self.translation)
        scale_x_bounds = (sx - self.scale, sx + self.scale)
        scale_y_bounds = (sy - self.scale, sy + self.scale)

        if in_radians:
            rotation_bounds = (np.deg2rad(-self.rotation), np.deg2rad(self.rotation))
            shear_bounds = (np.deg2rad(-self.shear), np.deg2rad(self.shear))
        else:
            rotation_bounds = (-self.rotation, self.rotation)
            shear_bounds = (-self.shear, self.shear)

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
