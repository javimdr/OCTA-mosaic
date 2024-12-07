import numpy as np


class OptimizeResult(dict):
    """Represents the optimization result.
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
    """

    def __init__(self, x: np.ndarray, fitness: float, message: str, nits: int):
        super().__init__()
        self.x = x
        self.fitness = fitness
        self.message = message
        self.nits = nits

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__  # type: ignore
    __delattr__ = dict.__delitem__  # type: ignore

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return "\n".join(
                [k.rjust(m) + ": " + repr(v) for k, v in sorted(self.items())]
            )
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())
