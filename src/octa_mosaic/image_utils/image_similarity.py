from typing import Literal

import numpy as np
from scipy.signal import fftconvolve


def ncc(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the Normalized Cross Correlation (NCC) between two 1D or 2D arrays of same
    shape.

    This function measures the similarity between two signals. It is normalized to
    ensure that the result is between -1 and 1, where 1 indicates perfect similarity,
    0 indicates no similarity, and -1 indicates perfect inverse similarity.

    Args:
        x (np.ndarray): First input array.
        y (np.ndarray): Second input array.

    Returns:
        float: Normalized cross correlation in range [-1, 1].
    """
    xf = x.astype(float)
    yf = y.astype(float)

    num = (xf * yf).sum()
    den = np.sqrt((xf**2).sum() * (yf**2).sum())
    if den == 0:
        return 0

    return num / den


def zncc(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the Zero-Normalized Cross Correlation (ZNCC) between two 1D or 2D arrays of
    same shape. This is equivalent to `normxcorr2(x, y, mode="valid")`

    ZNCC is a variant of the normalized cross-correlation that removes the mean
    of both arrays before computing the correlation. This is particularly useful
    for template matching where the signal's offset is irrelevant.

    Args:
        x (np.ndarray): First input array.
        y (np.ndarray): Second input array.

    Returns:
        float: Zero-normalized cross correlation in range [-1, 1].

    Raises:
        AssertionError: If the shapes of `x` and `y` do not match.

    Notes:
        If either array has zero variance, the function returns 0 to avoid division by
        zero. In this case, the result can be interpretated as no-correlation value.
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
    Computes the Zero-Normalized Cross Correlation (ZNCC) between a 2D image and a
    template, based on Octave/Matlab implementation
    (https://github.com/Sabrewarrior/normxcorr2-python)

    The function performs the cross-correlation of the `image` and `template`
    using the specified `mode` ("full", "same", or "valid") and returns the correlation
    coefficients for each region of the image where the template fits.

    Args:
        image (np.ndarray): The 2D array representing the image to search within.
        template (np.ndarray): The 2D array representing the template to match.
        mode (Literal["full", "same", "valid"]): The mode for convolution:
            - "full": Returns the full 2D correlation.
            - "same": Returns a correlation array with the same shape as `image`.
            - "valid": Returns only the valid correlation values where the template
                fits completely, that is equivalent to `zncc(x, y)`
        precision (int): The number of decimal places to round the output.

    Returns:
        np.ndarray: The resulting matrix of ZNCC values representing the correlation between
            `image` and `template`.

    Notes:
        The template is flipped both horizontally and vertically for the correlation process.
        If `template` is larger than `image`, an error message is printed.
    """
    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or template.size > image.size:
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


def dice(im1: np.ndarray, im2: np.ndarray, empty_score: float = 1.0) -> float:
    """
    Computes the Dice coefficient, a measure of set similarity between two binary images.

    The Dice coefficient is commonly used in image segmentation to measure the similarity
    between two binary sets (e.g., predicted vs ground truth segmentation).

    Args:
        im1 (np.ndarray): First binary image (array-like, bool).
        im2 (np.ndarray): Second binary image (array-like, bool).
        empty_score (float, optional): The score to return when both `im1` and `im2` are
            empty (default is 1.0).

    Returns:
        float: The Dice coefficient, a float in the range [0, 1], where 1 indicates
            perfect overlap and 0 indicates no overlap.
            If both images are empty, `empty_score` is returned.

    Raises:
        ValueError: If the shapes of `im1` and `im2` do not match.

    Notes:
        The order of the input images does not affect the result. If both images are empty (i.e., all pixels are 0),
        the function returns the `empty_score` value.
    """

    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:  # Full black images
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2.0 * intersection.sum() / im_sum
