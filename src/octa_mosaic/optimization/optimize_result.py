from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class OptimizeResult:
    """Result of an optimization algorithm.

    Args:
        x (ndarray): The vector solution of the optimizer.
        fitness (float): The fitness value of the vector solution.
        message (str, optional): Message with the cause of the optimizer termination.
        nit (int): Number of iterations performed by the optimizer.
    """

    x: np.ndarray
    fitness: float
    message: Optional[str] = None
    nits: Optional[int] = None
