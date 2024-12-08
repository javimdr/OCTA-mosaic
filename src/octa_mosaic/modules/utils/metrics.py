from typing import Sequence

import numpy as np


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
