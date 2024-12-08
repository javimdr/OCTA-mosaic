from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class OptimizeResult:
    """Result of an optimization algorithm.
    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    fitness : float
        Value of objective function
    message : str
        Description of the cause of the termination.
    nit : int
        Number of iterations performed by the optimizer.

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
