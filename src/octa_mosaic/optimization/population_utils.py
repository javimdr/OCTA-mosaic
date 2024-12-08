from concurrent import futures
from typing import Any, Callable, Sequence, Tuple

import numpy as np

OptimizationFunction = Callable[[np.ndarray, Sequence[Any]], float]


def denormalize(population: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """
    Converts a normalized population to its original scale using specified bounds.

    Args:
        population (np.ndarray): Normalized population where each value is in the
            range [0, 1].
        bounds (np.ndarray): Array of shape (n_dimensions, 2), where each row contains
            the lower and upper bounds for a dimension.

    Returns:
        np.ndarray: Denormalized population with values scaled according to the given
            bounds.
    """
    lower_bounds, upper_bounds = np.array(bounds).T
    diff = np.fabs(lower_bounds - upper_bounds)
    denormalized_population = lower_bounds + population * diff
    return denormalized_population


def normalize(population: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """
    Normalizes a population to the range [0, 1] using specified bounds.

    Args:
        population (np.ndarray): Population to normalize, with values within the range
            defined by bounds.
        bounds (np.ndarray): Array of shape (n_dimensions, 2), where each row contains
            the lower and upper bounds for a dimension.

    Returns:
        np.ndarray: Normalized population with values scaled to [0, 1].
    """
    lower_bounds, upper_bounds = np.array(bounds).T
    normalized_population = (population - lower_bounds) / (upper_bounds - lower_bounds)
    normalized_population = np.clip(normalized_population, 0, 1)
    return normalized_population


def evaluate_population(
    population: np.ndarray,
    f_obj: OptimizationFunction,
    f_obj_args: Sequence[Any] = (),
    n_workers: int = -1,
) -> np.ndarray:
    """
    Evaluates the fitness of each individual in a population using the objective function.

    Args:
        population (np.ndarray): Array of individuals to evaluate.
        f_obj (OptimizationFunction): Objective function to compute the fitness of
            each individual.
        f_obj_args (Sequence[Any], optional): Additional arguments to pass to the
            objective function. Defaults to ().
        n_workers (int, optional): Number of workers for parallel processing.
            Defaults to -1 (all available processors).

    Returns:
        np.ndarray: Array of fitness values corresponding to each individual in the population.

    """

    if n_workers == 1:
        return np.array([f_obj(x, *f_obj_args) for x in population], float)

    fitness_values = np.zeros(len(population), float)

    with futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_fitness = {
            executor.submit(f_obj, x, *f_obj_args): idx
            for idx, x in enumerate(population)
        }

        for future in futures.as_completed(future_to_fitness):
            idx = future_to_fitness[future]
            if future.exception() is None:
                fitness_values[idx] = future.result()
            else:
                raise ValueError(f"Error on individual {idx}: {future.exception()}")

    return fitness_values


def evaluate_and_select_best_individuals(
    indvs_to_select: int,
    population: np.ndarray,
    bounds: np.ndarray,
    f_obj: Callable,
    args: Tuple = (),
    n_workers: int = -1,
) -> np.ndarray:
    """
    Evaluate and selects the best individuals from a population based on their fitness
    values (higher fitness is better).

    Args:
        indvs_to_select (int): Number of individuals to select.
        population (np.ndarray): Normalized population to evaluate and select from.
        bounds (np.ndarray): Array of shape (n_dimensions, 2) containing the lower and
            upper bounds for each dimension.
        f_obj (Callable): Objective function to evaluate the fitness of individuals.
        args (Tuple, optional): Additional arguments to pass to the objective function.
            Defaults to ().
        n_workers (int, optional): Number of workers for parallel processing.
            Defaults to -1 (all available processors).

    Returns:
        np.ndarray: Array of the best individuals, normalized, in descending order of fitness.
    """
    pop_denorm = denormalize(population, bounds)
    fitness_values = evaluate_population(pop_denorm, f_obj, args, n_workers)

    fitness_argsort = np.argsort(fitness_values)[::-1]  # max to min
    sorted_initial_pop = population[fitness_argsort]

    return sorted_initial_pop[:indvs_to_select]
