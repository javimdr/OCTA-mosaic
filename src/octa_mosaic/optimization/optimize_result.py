from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class OptimizeResult:
    """
    Represents the result of an optimization algorithm.

    Attributes:
        x (np.ndarray): The best solution vector found by the optimizer.
        fitness (float): The fitness value of the solution vector.
        execution_time (float, optional): The time taken to complete the optimization
            process, in seconds.
        nits (int, optional): The number of iterations performed by the optimizer.
        last_population (np.ndarray, optional): The last population of solutions
            evaluated by the optimizer.
        message (str, optional): A message describing the reason for the optimizer's
            termination or other details.
    """

    x: np.ndarray
    fitness: float
    execution_time: Optional[float] = None
    nits: Optional[int] = None
    last_population: Optional[np.ndarray] = None
    message: Optional[str] = None

    def __repr__(self):
        return f"{self.__class__.__name__}(fitness={self.fitness:.4f})"
