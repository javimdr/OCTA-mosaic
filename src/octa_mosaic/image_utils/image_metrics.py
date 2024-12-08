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
