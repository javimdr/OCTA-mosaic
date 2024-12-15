from typing import Any, Callable, Dict, Optional

import numpy as np

import octa_mosaic.optimization.algorithms.differential_evolution as DE
from octa_mosaic import Mosaic, TemplateMatchingBuilder, mosaic_metrics
from octa_mosaic.mosaic.blends import alpha_blending
from octa_mosaic.mosaic.transforms import (
    PopulationInitializerConfig,
    TFLimits,
    tf_utils,
)
from octa_mosaic.optimization.population_utils import (
    evaluate_and_select_best_individuals,
)

# Default configuration
DEFAULT_FUNC = mosaic_metrics.calc_zncc_on_multiple_seamlines
DEFAULT_FUNC_KWARGS = {
    "widths": [5, 10, 15],
    "weights": [0.33, 0.33, 0.33],
}

DEFAULT_TF_LIMITS = TFLimits(
    translation=20,
    scale=0.1,
    rotation=10,
    shear=5,
)

DEFAULT_INITIAL_POP_CONFIG = PopulationInitializerConfig(
    mutation_factor=1.5,
    popsize=1000,
    sigma=0.125,
    seed=16,
)

DEFAULT_DE_CONFIG = DE.DifferentialEvolutionParams(
    F=0.5,
    C=0.9,
    generations=100,
    strategy="rand1bin",
    popsize=200,
    seed=16,
    fitness_population_std_tol=0.005,
    cores=8,
    display_progress_bar=True,
)


class MosaicProcessor:
    def __init__(
        self,
        func: Optional[Callable[[Mosaic, Any], float]] = None,
        func_kwargs: Optional[Dict[str, Any]] = None,
        tf_limits_config: Optional[TFLimits] = None,
        initial_pop_config: Optional[PopulationInitializerConfig] = None,
        de_config: Optional[DE.DifferentialEvolutionParams] = None,
    ):
        """
        Create mosaics using the three-stage process.

        Processor for generating wide-field OCTA mosaics from overlapping scan images
        with a fully automatic method. The approach consists of a three-stage pipeline:
            1. Build an Initial Mosaic: constructs an initial mosaic using
                correlation-based template matching.
            2. Optimize Mosaic: Refines the mosaic with an evolutionary algorithm to
                optimize vascular continuity at seams.
            3. Blend Seamlines: Blend the images together for a smooth transition

        The optimization process involves using a customizable mosaic evluation function
        and configuration for various parameters of the process such as transformation
        limits, initial population, and differential evolution settings.

        Args:
            func (Callable[[Mosaic, Any], float], optional): A callable function to
                evaluate and optimize the mosaic.
                If None, a default function is used. Default is None.
            func_kwargs (Dict[str, Any], optional): Arguments to pass to the
                objective function.
                If None, default arguments are used. Default is None.
            tf_limits_config (TFLimits, optional): Transformation limits.
                If None, the default config is used. Default is None.
            initial_pop_config (PopulationInitializerConfig, optional): Configuration
                for generate the initial population.
                If None, the default config is used. Default is None.
            de_config (DE.DifferentialEvolutionParams, optional): Configuration for the
                differential evolution algorithm.
                If None, the default config is used. Default is None.
        """
        # Set default values if not provided
        if func is None:
            func = DEFAULT_FUNC

        if func_kwargs is None:
            if func == DEFAULT_FUNC:
                func_kwargs = DEFAULT_FUNC_KWARGS
            else:
                func_kwargs = {}

        if tf_limits_config is None:
            tf_limits_config = DEFAULT_TF_LIMITS

        if initial_pop_config is None:
            initial_pop_config = DEFAULT_INITIAL_POP_CONFIG

        if de_config is None:
            de_config = DEFAULT_DE_CONFIG

        # Attributes
        self.func = func
        self.func_kwargs = func_kwargs
        self.tf_limits_config = tf_limits_config
        self.initial_population_config = initial_pop_config
        self.de_config = de_config

    def build_initial_mosaic(self, images: np.ndarray) -> Mosaic:
        """
        Create an initial mosaic using template matching.

        Args:
            images (np.ndarray): A list of images to use in the mosaic construction.

        Returns:
            Mosaic: The generated initial mosaic.
        """
        mosaic_builder = TemplateMatchingBuilder(
            first_pair_func=self.func,
            first_pair_kwargs=self.func_kwargs,
        )
        return mosaic_builder.create_mosaic(images)

    def optimize_mosaic(self, mosaic: Mosaic) -> Mosaic:
        """
        Optimize the mosaic using a differential evolution algorithm.

        This method applies differential evolution optimization on the initial mosaic,
        using the evaluation function and transformation limits. The optimization
        procedure adjusts the image transformations to best align the images.

        Args:
            mosaic (Mosaic): The initial mosaic to be optimized.

        Returns:
            Mosaic: The optimized mosaic after applying differential evolution.
        """
        objective_func = tf_utils.as_objective_function
        objective_args = (
            self.func,
            mosaic,
            *list(self.func_kwargs.values()),
        )

        # Compute bounds
        individual_bounds = self.tf_limits_config.compute_bounds()
        bounds = np.tile(individual_bounds, (mosaic.n_images(), 1))

        # Initialize and filter population
        initial_population = self._initialize_population(mosaic)
        initial_population_filtered = evaluate_and_select_best_individuals(
            indvs_to_select=self.de_config.popsize,
            population=initial_population,
            bounds=bounds,
            f_obj=objective_func,
            args=objective_args,
            n_workers=self.de_config.cores,
        )

        # Run optimization
        solution = DE.differential_evolution_from_params(
            self.de_config,
            bounds,
            objective_func,
            objective_args,
            initial_population_filtered,
        )
        return tf_utils.individual_to_mosaic(solution.x, mosaic)

    def _initialize_population(self, tm_mosaic):
        """
        Generate and initialize the population for the differential evolution
        optimization.

        Args:
            tm_mosaic (Mosaic): The initial mosaic used for population initialization.

        Returns:
            np.ndarray: The initialized population.
        """
        local_search = self.initial_population_config.build_initializer()
        initial_population = local_search.mutate_single_tf_gen(tm_mosaic.n_images())
        x0 = np.ones(tm_mosaic.n_images() * 6) * 0.5
        return np.vstack((x0, initial_population))

    def blend_mosaic(self, optimized_mosaic: Mosaic) -> np.ndarray:
        """
        Blend the seamlines for a smooth mosaic.

        This method applies an alpha blending technique to the optimized mosaic to
        produce a smooth transition between the images.

        Args:
            optimized_mosaic (Mosaic): The optimized mosaic to be blended.

        Returns:
            np.ndarray: The blended mosaic as a seamless image.
        """
        return alpha_blending(optimized_mosaic)

    def run(self, images: np.ndarray) -> np.ndarray:
        """
        Execute the complete mosaic processing pipeline.

        This method runs the entire process from building the initial mosaic using
        template matching, optimizing the mosaic with differential evolution, and
        blending the optimized mosaic to create the final seamless result.

        Args:
            images (np.ndarray): A list of images to be processed into a mosaic.

        Returns:
            np.ndarray: The final blended mosaic after optimization.
        """
        tm_mosaic = self.build_initial_mosaic(images)
        optimized_mosaic = self.optimize_mosaic(tm_mosaic)
        return self.blend_mosaic(optimized_mosaic)
