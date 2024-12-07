import cv2

from octa_mosaic.modules.utils import metrics


def template_matching(fixed, template, corr_func="ZNCC", mode="same"):
    """
    Realiza el proceso de template matching empleando la función de
    correlación seleccionada.

    :param fixed: imagen estática
    :param template: plantilla
    :param corr_func: función de correlación. Funciones:
        - CV: cv2.TM_CCORR_NORMED
        - ZNCC: Zero Normalized Cross Correlation

    :return: Valor máximo de correlación, su posición (esquina superior izq.)
    en forma 'xy' y la matriz de correlación.
    """
    _CCORR_FUNCTIONS = ["CV", "ZNCC"]

    if corr_func == "CV":
        ccorr_matrix = cv2.matchTemplate(
            fixed.astype("float32"), template.astype("float32"), cv2.TM_CCORR_NORMED
        )
    elif corr_func == "ZNCC":
        ccorr_matrix = metrics.normxcorr2(fixed, template, mode)
    else:
        raise ValueError(
            f"Invalid correlation function. Use one of this: {_CCORR_FUNCTIONS}"
        )

    min_value, max_value, min_location, max_location = cv2.minMaxLoc(ccorr_matrix)
    return max_value, max_location, ccorr_matrix
