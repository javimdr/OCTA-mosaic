import multiprocessing as mp
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple
from warnings import warn

import numpy as np
from tqdm import tqdm

from octa_mosaic.modules.optimization import evaluate
from octa_mosaic.modules.optimization.iteration_state import IterationState
from octa_mosaic.modules.optimization.optimize_result import OptimizeResult

CallbackFunction = Callable[[IterationState], bool]


@dataclass
class DifferentialEvolutionParams:
    """
    F: Mutation probability
    C: Crossover probability value
    popsize: Population size. Not necessary if initial population is used.
    """

    F: float = 0.8
    C: float = 0.7
    generations: int = 1000
    strategy: str = "best1bin"
    popsize: Optional[int] = None
    idv_elements: int = 1
    use_bounce_back: bool = True

    # Stop when the Euclidean distance of all individuals from the best
    # individual is less than `population_similarity`
    population_similarity: Optional[float] = None

    # Stop when not occurred an improvement of the best fitness value
    # in a `gens_without_improve` generations.
    gens_without_improve: Optional[int] = None

    # Stop when the std of all fitness values is less than `fitness_population_std_tol`.
    fitness_population_std_tol: Optional[float] = None

    seed: Optional[int] = None
    callback: Optional[CallbackFunction] = None
    cores: int = -1
    display_progress_bar: bool = False


def differential_evolution_from_params(
    params: DifferentialEvolutionParams,
    bounds: np.ndarray,
    fobj: Callable,
    fobj_args: Sequence[Any] = (),
    initial_population: Optional[np.ndarray] = None,
) -> OptimizeResult:

    solution = differential_evolution(
        fobj=fobj,
        args=fobj_args,
        bounds=bounds,
        initial_population=initial_population,
        mutation=params.F,
        recombination=params.C,
        generations=params.generations,
        strategy=params.strategy,
        popsize=params.popsize,
        n_elements_in_individual=params.idv_elements,
        use_bounce_back=params.use_bounce_back,
        max_dist=params.population_similarity,
        no_acc=params.gens_without_improve,
        pop_convergence_tol=params.fitness_population_std_tol,
        seed=params.seed,
        callback=params.callback,
        cores=params.cores,
        display_progress_bar=params.display_progress_bar,
    )
    return solution


def differential_evolution(
    fobj: Callable,
    bounds: np.ndarray,
    args: Sequence[Any] = (),
    mutation: float = 0.8,
    recombination: float = 0.7,
    generations: int = 1000,
    strategy: str = "best1bin",
    popsize: Optional[int] = 20,
    initial_population: Optional[np.ndarray] = None,
    n_elements_in_individual: int = 1,
    use_bounce_back: bool = True,
    max_dist: Optional[float] = None,
    no_acc: Optional[int] = None,
    pop_convergence_tol: Optional[float] = None,
    seed: Optional[int] = None,
    callback: Optional[Callable] = None,
    cores: int = -1,
    display_progress_bar: bool = False,
):
    """Finds the global maximum of a multivariate function.

    Parameters
    ----------
    fobj : Callable
        The objective function to be maximized. Must be in the form
        f(x, *args), where x is the argument in the form of a 1-D array
        and args is a tuple of any additional fixed parameters needed to
        completely specify the function.

    bounds : Sequence
        Bounds for variables. Sequence with upper and lower bounds, with
        size Nx2, where N is the dimmension of idividuals.

    args : Sequence
        Any additional fixed parameters needed to completely specify the
        objective function.

    mutation : float, optional
        The mutation constant, by default 0.8

    recombination : float, optional
        The crossover constant, by default 0.7

    generations : int, optional
        The maximum number of generations over which the entire population
        is evolved, by default 1000

    strategy : str, optional
        The differential evolution strategy to use. Should be one of:

        - 'best1bin'
        - 'rand1bin'

        The default is 'best1bin'

    popsize : int, optional
        Population size, by default 20

    initial_population : Sequence, optional
        Array specifying the initial population. The array should have
        shape (M, N), where M is the total population size and N is the
        the dimmension of idividuals. Must be normalized. By default is
        None, which means that the population is initialized with Latin
        Hypercube Sampling algorithm.

    n_elements_in_individual : int, optional
        If the individual is made up of several equal elements, it is
        interesting to carry out the crossover respecting these subsets.
        By default it is 1, indicating that the entire individual
        represents a single element.

    max_dist : float, optional
        Stop criteria [1]_: When the average euclidean distance of each
        individual with respect to the best individual is less than a
        threshold th (max_dist), the execution stops.
        By default it is None, which means that this criterion will not
        be used.

    no_acc : int, optional
        Stop criteria [1]_: If not occurred an improvement of the objective
        function in a specified number of generations g (no_acc), the
        execution will stop.
        By default it is None, which means that this criterion will not
        be used.

    seed : int, optional
        Seed used in the process, by default None.

    callback : Callable, optional
        A function to follow the progress of the process, by default None.
        This function will receive a single argument of type
        OptimizeSolution. If callback returns *True*, then the process is halted.

    cores : int, optional
        Number of cores to use, by default use all cores (-1).

    display_progress_bar : bool, optional
        Display progress bar of process, by default False. Requieres tqdm
        package.

    References
    ----------
    .. [1] Zielinski, K., Weitkemper, P., Laur, R., & Kammeyer, K. D.
            (2006, May). Examination of stopping criteria for differential
            evolution based on a power allocation problem. In Proceedings
            of the 10th International Conference on Optimization of
            Electrical and Electronic Equipment (Vol. 3, pp. 149-156).
    """
    solver = DifferentialEvolution(
        fobj,
        bounds,
        args=args,
        F=mutation,
        C=recombination,
        generations=generations,
        strategy=strategy,
        popsize=popsize,
        initial_population=initial_population,
        idv_elements=n_elements_in_individual,
        use_bounce_back=use_bounce_back,
        max_dist=max_dist,
        no_acc=no_acc,
        pop_convergence_tol=pop_convergence_tol,
        seed=seed,
        callback=callback,
    )  # as solver:

    sol = solver.solve(cores, display_progress_bar)
    return sol


def normalize_population(population, bounds):
    min_b, max_b = np.array(bounds, "float32").T
    pop_norm = (population - min_b) / (max_b - min_b)
    pop_norm = np.clip(pop_norm, 0, 1)
    return pop_norm


def init_population_lhs(pop_size, idv_dimm, seed):
    """
    Initializes the population with Latin Hypercube Sampling.
    Latin Hypercube Sampling ensures that each parameter is uniformly
    sampled over its range.
    """
    rng = np.random.RandomState(seed)

    segsize = 1.0 / pop_size
    samples = (
        segsize * rng.uniform(size=(pop_size, idv_dimm))
        + np.linspace(0.0, 1.0, pop_size, endpoint=False)[:, np.newaxis]
    )

    # Create an array for population of candidate solutions.
    population = np.zeros_like(samples)

    for j in range(idv_dimm):
        order = rng.permutation(range(pop_size))
        population[:, j] = samples[order, j]

    return population


class DifferentialEvolution:
    _STRATEGIES = ["rand1bin", "best1bin"]

    def __init__(
        self,
        fobj: Callable,
        bounds: np.ndarray,
        args: Sequence[Any] = (),
        F: float = 0.8,
        C: float = 0.7,
        generations: int = 1000,
        strategy: str = "best1bin",
        popsize: Optional[int] = 20,
        initial_population: Optional[np.ndarray] = None,
        idv_elements: int = 1,
        use_bounce_back: bool = True,
        max_dist: Optional[float] = None,
        no_acc: Optional[int] = None,
        pop_convergence_tol: Optional[float] = None,
        seed: Optional[int] = None,
        callback: Optional[Callable] = None,
    ):
        """Finds the global maximum of a multivariate function.

        Parameters
        ----------
        fobj : Callable
            The objective function to be maximized. Must be in the form
            f(x, *args), where x is the argument in the form of a 1-D array
            and args is a tuple of any additional fixed parameters needed to
            completely specify the function.

        bounds : Sequence
            Bounds for variables. Sequence with upper and lower bounds, with
            size Nx2, where N is the dimmension of idividuals.

        args : Sequence
            Any additional fixed parameters needed to completely specify the
            objective function.

        F : float, optional
            The mutation constant, by default 0.8

        C : float, optional
            The recombination constant, by default 0.7

        generations : int, optional
            The maximum number of generations over which the entire population
            is evolved, by default 1000

        strategy : str, optional
            The differential evolution strategy to use. Should be one of:

            - 'best1bin'
            - 'rand1bin'

            The default is 'best1bin'

        popsize : int, optional
            Population size, by default 20

        initial_population : Sequence, optional
            Array specifying the initial population. The array should have
            shape (M, N), where M is the total population size and N is the
            the dimmension of idividuals. Must be normalized. By default is
            None, which means that the population is initialized with Latin
            Hypercube Sampling algorithm.

        max_dist : float, optional
            Stop criteria [1]_: When the average euclidean distance of each
            individual with respect to the best individual is less than a
            threshold th (max_dist), the execution stops.
            By default it is None, which means that this criterion will not
            be used.

        no_acc : int, optional
            Stop criteria [1]_: If not occurred an improvement of the objective
            function in a specified number of generations g (no_acc), the
            execution will stop.
            By default it is None, which means that this criterion will not
            be used.

        seed : int, optional
            Seed used in the process, by default None.

        callback : Callable, optional
            A function to follow the progress of the process, by default None.
            This function will receive a single argument of type
            OptimizeSolution. If callback returns *True*, then the process is halted.

        References
        ----------
        .. [1] Zielinski, K., Weitkemper, P., Laur, R., & Kammeyer, K. D.
               (2006, May). Examination of stopping criteria for differential
               evolution based on a power allocation problem. In Proceedings
               of the 10th International Conference on Optimization of
               Electrical and Electronic Equipment (Vol. 3, pp. 149-156).
        """

        self.fobj = fobj
        self.fobj_args = args
        self.bounds = np.array(bounds, "float32")
        self._dimm = len(bounds)
        if idv_elements == 1:
            idv_elements = self._dimm
        self.idv_elements = idv_elements
        assert self._dimm % self.idv_elements == 0

        self.F = F
        self.C = C
        self.iterations = generations
        self.strategy = strategy
        self.use_bounce_back = use_bounce_back
        self.random_state = np.random.RandomState(seed)

        self.max_dist = max_dist
        self.no_acc_iters = no_acc
        self.pop_convergence_tol = pop_convergence_tol
        self.callback = callback

        assert self.max_dist is None or self.max_dist > 0
        assert self.no_acc_iters is None or self.no_acc_iters > 0

        if initial_population is not None:
            self.pop_size = len(initial_population)
            self.population = np.array(initial_population, "float32")
            if not (self.population.min() >= 0.0 and self.population.max() <= 1.0):
                warn(
                    "Las componentes de los individuos deben estar normalizadas en el intervalo [0, 1]. Normalizando..."
                )
                self.population = normalize_population(initial_population, bounds)
        elif popsize is None:
            raise ValueError(
                "Select a `popsize` value if `initial_population` is not provided."
            )
        else:
            self.pop_size = popsize
            self.population = init_population_lhs(
                self.pop_size, np.size(bounds, 0), seed
            )

        self.min_b, self.max_b = np.array(bounds).T
        self.diff = np.fabs(self.min_b - self.max_b)
        self.pop_denorm = self._denorm(self.population)

        # **
        self.fitness_values = np.full(self.pop_size, -np.inf, "float32")
        self.best_idx = 0

        self.its_since_last_improvement = 0
        self.n_cores = mp.cpu_count()

    def _denorm(self, x: np.ndarray) -> np.ndarray:
        return self.min_b + x * self.diff

    def _apply_mutation(self, individual_idx: int) -> np.ndarray:
        candidates_idxs = [idx for idx in range(self.pop_size) if idx != individual_idx]

        if self.strategy == "rand1bin":
            x1, x2, x3 = self.population[
                self.random_state.choice(candidates_idxs, 3, replace=False)
            ]
        elif self.strategy == "best1bin":
            x1 = self.population[self.best_idx]
            x2, x3 = self.population[
                self.random_state.choice(candidates_idxs, 2, replace=False)
            ]
        else:
            raise ValueError(
                f"Strategy '{self.strategy}' not valid. Use rand1bin or best1bin."
            )

        individual_mutated = x1 + self.F * (x2 - x3)

        # Ensure limits
        if self.use_bounce_back:
            individual_mutated = self._bounce_back(
                individual_mutated, self.population[individual_idx]
            )

        individual_mutated = np.clip(individual_mutated, 0, 1)
        return individual_mutated

    def _bounce_back(self, descendant: np.ndarray, original: np.ndarray) -> np.ndarray:
        assert len(descendant) == len(original)

        for idx in range(len(descendant)):
            if descendant[idx] < 0:
                descendant[idx] = original[idx] + self.random_state.rand() * (
                    0 - original[idx]
                )

            if descendant[idx] > 1:
                descendant[idx] = original[idx] + self.random_state.rand() * (
                    1 - original[idx]
                )

        return descendant

    def _recombination(self, descendant, original):
        cross_points = self.random_state.rand(self.idv_elements) < self.C

        if not np.any(cross_points):  # Forzar al menos una caracteristica activada
            cross_points[self.random_state.randint(0, self.idv_elements)] = True
        if self._dimm != self.idv_elements:
            cross_points = np.repeat(cross_points, self._dimm // self.idv_elements)

        return np.where(cross_points, descendant, original)

    def _evaluate_population(
        self, population: np.ndarray, n_workers: int = -1
    ) -> np.ndarray:
        """Calculate the fitness value of the each individual in a population."""

        population_denorm = self._denorm(population)

        fitness_list = evaluate.evaluate_population(
            population_denorm, self.fobj, self.fobj_args, n_workers
        )

        return fitness_list

    def create_new_individual(self, individual_idx: int) -> np.ndarray:
        mutated = self._apply_mutation(individual_idx)
        descendant = self._recombination(mutated, self.population[individual_idx])
        return descendant

    def _update(
        self, trial_population: np.ndarray, trial_fitness_values: np.ndarray
    ) -> None:
        """
        Update the population and its fitness values based on the trial
        population and its trial.
        """
        if np.max(trial_fitness_values) > np.max(self.fitness_values):
            self.best_idx = np.argmax(trial_fitness_values)
            self.its_since_last_improvement = 0

        loc = trial_fitness_values > self.fitness_values
        self.population = np.where(loc[:, np.newaxis], trial_population, self.population)
        self.fitness_values = np.where(loc, trial_fitness_values, self.fitness_values)

    def _evaluate_generation(self):
        pass

    def _check_stops_critereias(self) -> Tuple[bool, Optional[str]]:
        status_message = None
        if self.max_dist is not None:
            best_individual = self.population[self.best_idx]
            if np.all(
                [
                    euclidean_dist(best_individual, individual) < self.max_dist
                    for individual in self.population
                ]
            ):
                status_message = "Ended because the distance criterion was met."

        if (
            self.no_acc_iters is not None
            and self.its_since_last_improvement >= self.no_acc_iters
        ):
            status_message = f"Ended because the best individual did not improve in {self.no_acc_iters} generations."

        if (
            self.pop_convergence_tol is not None
            and np.std(self.fitness_values) <= self.pop_convergence_tol
        ):
            status_message = f"Ended because the std of fitness population values is lower than {self.pop_convergence_tol} (value: {np.std(self.fitness_values) : 0.4f})."

        return (status_message is not None, status_message)

    def end_of_generation(self) -> None:
        pass

    def start_of_generation(self) -> None:
        pass

    def start_optimization(self) -> None:
        pass

    def end_optimization(self) -> None:
        pass

    def solve(self, cores=-1, display_progress_bar=False):
        start_time = time.perf_counter()
        status_message = None

        n_cores = mp.cpu_count() if cores < 1 else min(cores, mp.cpu_count())
        progress_bar = tqdm(
            range(1, self.iterations + 1),
            desc=f"Current fitness value: {self.fitness_values[self.best_idx]:.4f}.",
            disable=not display_progress_bar,
        )
        self.start_optimization()
        nit = 0
        for gen in progress_bar:
            progress_bar.set_description(
                f"Current fitness value: {self.fitness_values[self.best_idx]:.4f}."
            )
            nit += 1
            self.its_since_last_improvement += 1

            self.start_of_generation()

            trial_population = np.array(
                [self.create_new_individual(idx) for idx in range(self.pop_size)],
                # "float32",
            )
            trial_fitness = self._evaluate_population(trial_population, n_cores)
            self._update(trial_population, trial_fitness)

            generation_state = IterationState(
                nit,
                self._denorm(self.population),
                self.fitness_values,
                int(self.best_idx),
            )

            if self.callback is not None:
                if self.callback(generation_state):
                    status_message = f"Ended by callback function request."
                    break

            # Stop criterias
            stop_order, status_message = self._check_stops_critereias()
            if stop_order:
                break
            self.end_of_generation()
        self.end_optimization()
        end_time = time.perf_counter()

        if status_message is None:
            status_message = "Ended because the total number of generations was made."

        result = OptimizeResult(
            x=self._denorm(self.population[self.best_idx]),
            fitness=self.fitness_values[self.best_idx],
            message=status_message,
            nits=nit,
        )

        result.execution_time = end_time - start_time
        result.last_population = self._denorm(self.population)

        return result


def euclidean_dist(p: Iterable[float], q: Iterable[float]) -> float:
    """Compute euclidean distance between P and Q.

    Args:
        p (Iterable[float]): 1D array
        q (Iterable[float]): 1D array

    Returns:
        float: euclidean distance between P and Q

    """
    return np.linalg.norm(np.subtract(p, q))
