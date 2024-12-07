from concurrent import futures
from typing import Any, Callable, Sequence, Tuple

import numpy as np

OptimizationFunction = Callable[[np.ndarray, Sequence[Any]], float]


def evaluate_population(
    population: np.ndarray,
    f_obj: OptimizationFunction,
    f_obj_args: Sequence[Any] = (),
    n_workers: int = 1,
) -> np.ndarray:
    """Calculate the fitness value of the each solution in a population

    Args:
        population (np.ndarray): set of possibles solutions
        f_obj (Callable): _description_
        f_obj_args (Tuple, optional): _description_. Defaults to ().
        n_workers (int, optional): _description_. Defaults to 1.

    Returns:
        np.ndarray: _description_
    """

    if n_workers == 1:
        return np.array([f_obj(x, *f_obj_args) for x in population], float)

    # Otherwise:
    fitness_list = np.zeros(len(population), float)

    with futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_fitness = {
            executor.submit(f_obj, x, *f_obj_args): idx
            for idx, x in enumerate(population)
        }

        for future in futures.as_completed(future_to_fitness):
            idx = future_to_fitness[future]
            if future.exception() is None:
                fitness_list[idx] = future.result()
            else:
                raise ValueError(f"Error on individual {idx}: {future.exception()}")

    return fitness_list


def denormalize(population: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    bl, bu = np.array(bounds).T
    diff = np.fabs(bl - bu)
    pop_denorm = bl + population * diff
    return pop_denorm


def normalize(population, bounds):
    min_b, max_b = np.array(bounds).T
    pop_norm = (population - min_b) / (max_b - min_b)
    pop_norm = np.clip(pop_norm, 0, 1)
    return pop_norm
