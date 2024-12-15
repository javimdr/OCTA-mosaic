import numpy as np
from skimage.transform import AffineTransform

from octa_mosaic.mosaic.mosaic import Mosaic


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


def individual_to_mosaic(individual: np.ndarray, mosaic: Mosaic) -> Mosaic:
    """
    Convert an individual to a Mosaic.

    This function takes an individual (represented as a 1D array) and applies the
    transformation encoded in the array to the provided Mosaic. The individual is
    divided into segments, each corresponding to a transformation params for each image
    in the mosaic.

    Args:
        individual (np.ndarray): A array of transformation parameters for each image in
            the mosaic.
        mosaic (Mosaic): The original Mosaic to apply the transformations to.

    Returns:
        Mosaic: A new Mosaic with the transformations applied.
    """
    n_images = mosaic.n_images()
    new_mosaic = mosaic.copy()

    tfs_list = [
        params_to_tf(individual[idx * 6 : (idx * 6) + 6]) for idx in range(n_images)
    ]
    new_mosaic.set_transforms_list(tfs_list)

    return new_mosaic
