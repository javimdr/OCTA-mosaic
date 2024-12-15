from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class PopulationInitializerType(Enum):
    MUTATE_SINGLE_TF_GEN = "mutate_single_tf_gen"
    MUTATE_ALL_TFS = "mutate_all_tfs"


@dataclass
class PopulationInitializerConfig:
    """
    Local search population initializers config.

    Attributes:
        popsize (int): Number of individuals to create. By default, 100.
        transformation_len (int): The length of each transformation in the
            individual. Defaults to 6.
        mutation_factor (float): Mutation probability factor.
        sigma (float, optional): Standard deviation of the normal distribution.
            Defaults to 0.125.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
    """

    popsize: int = 100
    transformation_len: int = 6
    mutation_factor: float = 1.5
    sigma: float = 0.125
    seed: Optional[int] = None

    def build_initializer(self) -> "TFPopulationInitializer":
        return TFPopulationInitializer(
            popsize=self.popsize,
            transformation_len=self.transformation_len,
            mutation_factor=self.mutation_factor,
            sigma=self.sigma,
            seed=self.seed,
        )


@dataclass
class TFPopulationInitializer:
    """
    Local search population initializers.

    These methods generates populations of individuals normalized in the range [0, 1].
    Each individual is initialized as a vector of `0.5`, which corresponds to
    keeping the original transformation values (no modification). This class apply
    different local search methods around this initial vector, applying mutations
    to generate diverse populations.

    Mutation is performed by adding values sampled from a Gaussian distribution with
    mean `0` and standard deviation `sigma`.

    Attributes:
        popsize (int): Number of individuals to create. By default, 100.
        transformation_len (int): The length of each transformation in the
            individual. Defaults to 6.
        mutation_factor (float): Mutation probability factor.
        sigma (float, optional): Standard deviation of the normal distribution.
            Defaults to 0.125.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
    """

    popsize: int = 100
    transformation_len: int = 6
    mutation_factor: float = 1.5
    sigma: float = 0.125
    seed: Optional[int] = None  # RandomState should be created in init

    def apply(
        self, initializer_type: PopulationInitializerType, n_transformations: int
    ) -> np.ndarray:
        if initializer_type == PopulationInitializerType.MUTATE_SINGLE_TF_GEN:
            return self.mutate_single_tf_gen(n_transformations)

        elif initializer_type == PopulationInitializerType.MUTATE_ALL_TFS:
            return self.mutate_all_tfs(n_transformations)

        raise ValueError(f"Unknown initializer type `{initializer_type}`")

    def mutate_single_tf_gen(self, n_transformations: int) -> np.ndarray:
        """
        Generates the population by modifying a single gene of certain transformations
        within each individual.

        Applies a mutation probability `P_mut` to determine which transformations will
        be mutated. At least one transformation must be mutated. Mutated transformations
        alter ONLY one gene.

        Mutation probability is defined as:
            P_mut = mutation_factor / n_transformations

        Args:
            n_transformations (int): Number of transformations per individual.

        Returns:
            np.ndarray: Population of shape (popsize, n_transformations * transformation_len).
        """
        rng = np.random.RandomState(self.seed)

        n_gens = n_transformations * self.transformation_len
        individual_identity = np.ones(n_gens) * 0.5
        prob_mut = self.mutation_factor / n_transformations

        # Replicate `x` n times
        output = np.tile(individual_identity, (self.popsize, 1)).astype("float32")
        assert output.shape == (self.popsize, n_gens)

        for individual_i in range(self.popsize):
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

                output[individual_i][gene_to_mute] += rng.normal(0, self.sigma)

        return np.clip(output, 0, 1)

    def mutate_all_tfs(self, n_transformations: int) -> np.ndarray:
        """
        Generates the population by homogeneously modifying all transformations
        within each individual.

        Applies a mutation probability (P_mut) to determine which genes of each
        transformation are mutated. At least one gene in each transformation must be
        mutated.

        Mutation is performed by adding a value sampled from a Gaussian distribution
        with zero mean and standard deviation `sigma`.

        Mutation probability is defined as:
            P_mut = mutation_factor / transformation_len

        Args:
            n_transformations (int): Number of transformations per individual.

        Returns:
            np.ndarray: Population of shape (popsize, n_transformations * transformation_len).
        """
        rng = np.random.RandomState(self.seed)

        n_gens = n_transformations * self.transformation_len
        prob_mut = self.mutation_factor / self.transformation_len

        tf_identity = np.ones(self.transformation_len) * 0.5
        output = np.zeros((self.popsize, n_gens), "float32")

        for individual_i in range(self.popsize):
            for transformation_i in range(n_transformations):
                gens_to_mutated = rng.rand(self.transformation_len) <= prob_mut

                # Assert almost 1 gen is going to be muted
                if not any(gens_to_mutated):
                    gen_to_mute = rng.randint(0, self.transformation_len)
                    gens_to_mutated[gen_to_mute] = True

                tf_start = transformation_i * 6
                tf_end = tf_start + 6

                tf_mutated = tf_identity + rng.normal(
                    0, self.sigma, self.transformation_len
                )

                output[individual_i][tf_start:tf_end] = np.where(
                    gens_to_mutated, tf_mutated, tf_identity
                )

        return np.clip(output, 0, 1)
