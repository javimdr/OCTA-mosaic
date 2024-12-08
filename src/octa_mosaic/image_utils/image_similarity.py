from typing import Literal

import numpy as np
from scipy.signal import fftconvolve


def ncc(x: np.ndarray, y: np.ndarray) -> float:
    """Normalized Cross Correlation"""
    xf = x.astype(float)
    yf = y.astype(float)

    num = (xf * yf).sum()
    den = np.sqrt((xf**2).sum() * (yf**2).sum())
    if den == 0:
        return 0

    return num / den


def zncc(x: np.ndarray, y: np.ndarray) -> float:
    """
    Zero-Normalized Cross Correlation (ZNCC).

    This is equivalent to `normxcorr2(x, y, mode="valid")`
    https://es.mathworks.com/help/images/ref/normxcorr2.html
    """
    assert x.shape == y.shape

    xf = x.astype(float)
    yf = y.astype(float)

    x_subs_mean = xf - np.mean(xf)
    y_subs_mean = yf - np.mean(yf)
    # x_std = np.std(xf)
    # y_std = np.std(yf)

    num = np.sum(x_subs_mean * y_subs_mean)
    den = (np.sum(x_subs_mean**2) * np.sum(y_subs_mean**2)) ** 0.5

    return num / den if den != 0 else 0


def normxcorr2(
    image: np.ndarray,
    template: np.ndarray,
    mode: Literal["full", "same", "valid"] = "valid",
    precision: int = 10,
) -> np.ndarray:
    """
    Computes the ZNCC of the matrices `image` and `template` using the mode `mode`.
    The resulting matrix contains the correlation coefficients.

    Based on Octave/Matlab implementation by: Benjamin Eltzner, 2014 <b.eltzner@gmx.de>
    Based on https://github.com/Sabrewarrior/normxcorr2-python
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
    out = out.round(precision)

    if den.size == 1:
        return float(out)
    return out


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
