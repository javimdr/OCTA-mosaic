from typing import Optional

import numpy as np


class PopulationBasedOnLocalSearch:
    def __init__(self, seed: Optional[int] = None, transformation_len: int = 6):
        """
        Local search population initializers.

        This class generates populations of individuals normalized in the range [0, 1].
        Each individual is initialized as a vector of `0.5`, which corresponds to
        keeping the original transformation values (no modification). This class apply
        different local search methods around this initial vector, applying mutations
        to generate diverse populations.

        Mutation is performed by adding values sampled from a Gaussian distribution with
        mean `0` and standard deviation `sigma`.

        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to None.
            transformation_len (int): The length of each transformation in the
                individual. Defaults to 6.
        """
        self.seed = seed
        self.transformation_len = transformation_len

    def tf_level_single_gen(
        self, n_transformations: int, popsize: int, x: float, sigma: float = 0.125
    ) -> np.ndarray:
        """
        Generates the population by modifying a single gene of certain transformations
        within each individual.

        Applies a mutation probability `P_mut` to determine which transformations will
        be mutated. At least one transformation must be mutated. Mutated transformations
        alter ONLY one gene.

        Mutation probability is defined as:
            P_mut = x / n_transformations

        Args:
            n_transformations (int): Number of transformations per individual.
            popsize (int): Number of individuals to create.
            x (float): Mutation probability, expressed as P_mut = x / n_transformations.
            sigma (float, optional): Standard deviation of the normal distribution.
                Defaults to 0.125.

        Returns:
            np.ndarray: Population of shape (popsize, n_transformations * transformation_len).
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
        Generates the population by homogeneously modifying all transformations
        within each individual.

        Applies a mutation probability (P_mut) to determine which genes of each
        transformation are mutated. At least one gene in each transformation must be
        mutated.

        Mutation is performed by adding a value sampled from a Gaussian distribution
        with zero mean and standard deviation `sigma`.

        Mutation probability is defined as:
            P_mut = x / transformation_len, where transformation_len = 6

        Args:
            n_transformations (int): Number of transformations per individual.
            popsize (int): Number of individuals to create.
            x (float): Mutation probability, expressed as P_mut = x / transformation_len.
            sigma (float, optional): Standard deviation of the normal distribution.
                Defaults to 0.125.

        Returns:
            np.ndarray: Population of shape (popsize, n_transformations * transformation_len).
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
