from typing import Callable, Optional, Tuple

import numpy as np

from octa_mosaic.modules.optimization import evaluate


class PopulationBasedOnLocalSearch:
    def __init__(self, seed: Optional[int] = None, transformation_len: int = 6):
        self.seed = seed
        self.transformation_len = transformation_len

    def tf_level_single_gen(
        self, n_transformations: int, popsize: int, x: float, sigma: float = 0.125
    ) -> np.ndarray:
        """
        Genera la población modificando un único gen de algunas transformaciones del individuo.

        Aplica P_mut para determinar qué transformaciones van a ser mutadas (AL MENOS una debe serlo).
        Las transformaciones mutadas SOLO alteran un gen de las mismas.

        La mutación se realiza sumando un valor extraído de una gaussiana de media cero y desviación
        estándar `sigma`.

        La probabilidad de mutación es definida como: P_mut = x / #numero_transformaciones

        Args:
            n_transformations (int): numero de transformaciones de las que consta cada individuo.
            popsize (int): number of individuals to create.
            x (float): mutation probability expressed like p_mut = x / #n_transformations
            sigma (float, optional): sigma of normal distribution with 0 mean. Defaults to 0.125.

        Returns:
            np.ndarray: population of size `popsize`x`n_transformations*TRANSFORMATION_LEN`
        """

        rng = np.random.RandomState(self.seed)

        n_gens = n_transformations * self.transformation_len
        individual_identity = np.ones(n_gens) * 0.5
        prob_mut = x / n_transformations

        output = np.tile(individual_identity, (popsize, 1)).astype(
            "float32"
        )  # Replicate `x` n times
        assert output.shape == (popsize, n_gens)

        for individual_i in range(popsize):
            tfs_to_mutate = rng.rand(n_transformations) <= prob_mut

            # Assert at least 1 tf is going to be mutated
            if not any(tfs_to_mutate):
                tf_i = rng.randint(0, n_transformations)
                tfs_to_mutate[tf_i] = True

            for idx, need_to_be_mutated in enumerate(tfs_to_mutate):
                if not need_to_be_mutated:
                    continue

                # Mutate ONLY one gene
                tf_start_idx = idx * self.transformation_len
                tf_end_idx = tf_start_idx + self.transformation_len
                gene_to_mute = rng.randint(tf_start_idx, tf_end_idx)

                output[individual_i][gene_to_mute] += rng.normal(0, sigma)

        return np.clip(output, 0, 1)

    def tf_level(
        self,
        n_transformations: int,
        popsize: int,
        x: float,
        sigma: float = 0.125,
    ) -> np.ndarray:
        """
        Genera la población modificando de forma homogénea TODAS las transformaciones del individuo.
        Aplica P_mut para determinar qué genes de cada transformacion son mutados (AL MENOS uno debe serlo).

        La mutación se realiza sumando un valor extraído de una gaussiana de media cero y desviación
        estándar `sigma`.

        La probabilidad de mutación es definida como:
        P_mut = x / #TRANSFORMATION_LEN, donde #TRANSFORMATION_LEN = 6

        Args:
            n_transformations (int): numero de transformaciones de las que consta cada individuo.
            popsize (int): number of individuals to create.
            x (float): mutation probability expressed like p_mut = x / #n_transformations
            sigma (float, optional): sigma of normal distribution with 0 mean. Defaults to 0.125.
            seed (Optional[int], optional): Defaults to None.

        Returns:
            np.ndarray: population of size `popsize`x`n_transformations*TRANSFORMATION_LEN`
        """
        rng = np.random.RandomState(self.seed)

        n_gens = n_transformations * self.transformation_len
        prob_mut = x / self.transformation_len

        tf_identity = np.ones(self.transformation_len) * 0.5
        output = np.zeros((popsize, n_gens), "float32")

        for individual_i in range(popsize):
            for transformation_i in range(n_transformations):
                gens_to_mutated = rng.rand(self.transformation_len) <= prob_mut

                # Assert almost 1 gen is going to be muted
                if not any(gens_to_mutated):
                    gen_to_mute = rng.randint(0, self.transformation_len)
                    gens_to_mutated[gen_to_mute] = True

                tf_start = transformation_i * 6
                tf_end = tf_start + 6

                tf_mutated = tf_identity + rng.normal(0, sigma, self.transformation_len)

                output[individual_i][tf_start:tf_end] = np.where(
                    gens_to_mutated, tf_mutated, tf_identity
                )

        return np.clip(output, 0, 1)


def select_best_individuals(
    indvs_to_select: int,
    population: np.ndarray,
    bounds: np.ndarray,
    f_obj: Callable,
    args: Tuple = (),
    n_workers: int = -1,
) -> np.ndarray:

    pop_denorm = evaluate.denormalize(population, bounds)
    fitness_values = evaluate.evaluate_population(pop_denorm, f_obj, args, n_workers)

    fitness_argsort = np.argsort(fitness_values)[::-1]  # max to min
    sorted_initial_pop = population[fitness_argsort]

    return sorted_initial_pop[:indvs_to_select]
