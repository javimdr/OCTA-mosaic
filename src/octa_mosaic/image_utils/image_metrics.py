import numpy as np


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
    Zero-Normalized Cross Correlation (ZNCC)
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
