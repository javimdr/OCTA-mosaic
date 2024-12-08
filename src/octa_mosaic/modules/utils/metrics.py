from typing import Sequence

import cv2
import numpy as np
from scipy.signal import fftconvolve

from octa_mosaic.image_utils.image_metrics import ncc, zncc


def euclidean_dist(p: Sequence, q: Sequence) -> float:
    """Compute euclidean distance between the point P and Q.

    Parameters
    ----------
    p : Sequence
        First point
    q : Sequence
        Second point

    Returns
    -------
    float
        Euclidean distance
    """
    return np.linalg.norm(np.subtract(p, q))


def normxcorr2(image, template, mode="valid", precission=10):
    """
    Fast Template Matching usign ZNCC function.
    https://github.com/Sabrewarrior/normxcorr2-python/blob/master/normxcorr2.py
    """

    # If this happens, it is probably a mistake
    if (
        np.ndim(template) > np.ndim(image)
        or len(
            [i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]
        )
        > 0
    ):
        print("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)

    image = fftconvolve(np.square(image), a1, mode=mode) - np.square(
        fftconvolve(image, a1, mode=mode)
    ) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    den = np.sqrt(image * template)

    if den.size == 1 and den == 0:
        return 0

    with np.errstate(divide="ignore", invalid="ignore"):
        out /= den

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0
    out = out.round(precission)

    if den.size == 1:
        return float(out)
    return out


def impair_ccorr(src_image, dst_image, transform, ccorr_function="ZNCC"):
    """

    Parameters
    ----------
    src_image
    dst_image
    transform
    mask
    ccorr_function: {CV, ZNCC}

    Returns
    -------

    """
    H, W = dst_image.shape[:2]
    mask = np.ones(src_image.shape[:2], np.uint8)

    src_transform = cv2.warpPerspective(src_image, transform, (W, H))
    mask_transform = cv2.warpPerspective(mask, transform, (W, H))
    indexes = mask_transform > 0

    if ccorr_function == "CV":
        ccorr = ncc(src_transform[indexes], dst_image[indexes])
    elif ccorr_function == "ZNCC":
        ccorr = zncc(src_transform[indexes], dst_image[indexes])
    else:
        raise ValueError("Invalid correlation funcion. Use 'CV' or 'ZNCC'")

    return ccorr


def dice(im1, im2, empty_score=1.0):
    """
    Computes the Dice coefficient, a measure of set similarity.
    https://gist.github.com/brunodoamaral/e130b4e97aa4ebc468225b7ce39b3137

    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """

    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:  # black images
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2.0 * intersection.sum() / im_sum


# def dice_coef(y_true, y_pred):
#     y_true_f = tf.reshape(tf.dtypes.cast(y_true, tf.float32), [-1])
#     y_pred_f = tf.reshape(tf.dtypes.cast(y_pred, tf.float32), [-1])
#     intersection = tf.reduce_sum(y_true_f * y_pred_f)
#     return (2. * intersection + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)

# def pixel_accuracy(im1, im2):
#     """
#     Computes the Pixel Accuracy, a measure of set similarity.

#     Parameters
#     ----------
#     im1 : array-like, bool
#         Any array of arbitrary size. If not boolean, will be converted.
#     im2 : array-like, bool
#         Any other array of identical size. If not boolean, will be converted.
#     Returns
#     -------
#     pixel_acc : float
#         Pixel accuracy as a float on range [0,1].
#         Maximum similarity = 1
#         No similarity = 0

#     Notes
#     -----
#     The order of inputs for `pixel_accuracy` is irrelevant. The result will be
#     identical if `im1` and `im2` are switched.
#     """

#     im1 = np.asarray(im1).astype(bool)
#     im2 = np.asarray(im2).astype(bool)

#     if im1.shape != im2.shape:
#         raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
#     if im1.ndim > 2:
#         raise ValueError("Images must be 2D or 1D")

#     n_pixels = im1.size
#     if n_pixels == 0:
#         raise ValueError("Images can not be empty")

#     true_positive = np.count_nonzero(im1==im2)
#     return true_positive / n_pixels


def pixel_accuracy(im1, im2):
    """
    Computes the Pixel Accuracy: TP / (FN + FP).

    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    pixel_acc : float
        Pixel accuracy as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `pixel_accuracy` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """

    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    if im1.ndim > 2:
        raise ValueError("Images must be 2D or 1D")

    n_pixels = im1.size
    if n_pixels == 0:
        raise ValueError("Images can not be empty")

    true_positive = np.logical_and(im1, im2)
    falses = np.logical_xor(im1, im2)
    return true_positive.sum() / falses.sum()
