from typing import Literal

import cv2
import numpy as np

from octa_mosaic.image_utils import image_similarity


def template_matching(
    fixed: np.ndarray,
    template: np.ndarray,
    corr_func: Literal["ZNCC", "CV"] = "ZNCC",
    mode: Literal["same", "full"] = "full",
):
    """
    Find the location of the maximum correlation between a fixed
    image and a template.

    Parameters:
        fixed (np.ndarray): The fixed image for template matching.
        template (np.ndarray): The template to match against the fixed image.
        corr_func (str): The type of correlation function to use. Supported values
            are "CV" (uses cv2.matchTemplate) and "ZNCC" (uses metrics.normxcorr2).
        mode (str): How to handle edge pixels when matching. Supported values are
            "same" and "full".

        Returns:
            max_value (float): The highest correlation value.
            max_location (tuple): The coordinates of the maximum correlation point in
                the format (x, y).
            ccorr_matrix (np.ndarray): The 2D correlation matrix between the fixed and
                template.
    """
    _CCORR_FUNCTIONS = ["CV", "ZNCC"]

    if corr_func == "CV":
        ccorr_matrix = cv2.matchTemplate(
            fixed.astype("float32"), template.astype("float32"), cv2.TM_CCORR_NORMED
        )
    elif corr_func == "ZNCC":
        ccorr_matrix = image_similarity.normxcorr2(fixed, template, mode)
    else:
        raise ValueError(
            f"Invalid correlation function. Use one of this: {_CCORR_FUNCTIONS}"
        )

    _, max_value, _, max_location = cv2.minMaxLoc(ccorr_matrix)
    return max_value, max_location, ccorr_matrix
