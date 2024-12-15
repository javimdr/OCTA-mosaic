import numpy as np
from skimage.transform import AffineTransform


def params_to_tf(params: np.ndarray) -> AffineTransform:
    """
    Convert an array of an affine transform parameters to an AffineTransform object.

    The input array should contain 6 elements, representing the following transformation
    parameters:
        - translation in x and y
        - scale in x and y
        - rotation angle expressed in radians
        - shear angle expressed in radians

    Args:
        params (np.ndarray): A 1D numpy array with 6 elements representing the
            transformation parameters.

    Returns:
        AffineTransform: An instance of the AffineTransform.
    """
    if len(params) != 6:
        raise ValueError("params must contains 6 values")

    T = AffineTransform(
        translation=params[:2],
        scale=params[2:4],
        rotation=params[4],
        shear=params[5],
    )
    return T


def tf_to_params(matrix: AffineTransform) -> np.ndarray:
    """
    Convert an AffineTransform instance to a 1D array with the transformation parameters.

    The output array contains the following transformation parameters:
        - translation in x and y
        - scale in x and y
        - rotation angle expressed in radians
        - shear angle expressed in radians

    Args:
        matrix (AffineTransform): An affine transform.

    Returns:
        np.ndarray: A 1D numpy array containing the transformation parameters in the
            order: `[translation_x, translation_y, scale_x, scale_y, rotation, shear]`.
    """
    if not isinstance(matrix, AffineTransform):
        raise ValueError("Expected an instance of `AffineTransform`")

    tx, ty = matrix.translation
    sx, sy = matrix.scale
    rot = matrix.rotation
    shear = matrix.shear

    return np.array([tx, ty, sx, sy, rot, shear])
