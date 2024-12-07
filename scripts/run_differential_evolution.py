# import useful packages
import datetime
import json
import pickle as pkl
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# user packages
from octa_mosaic.data import Dataset, DatasetCase
from octa_mosaic.modules import optimization_utils
from octa_mosaic.modules.experiments import initialize_population
from octa_mosaic.modules.experiments.encoders import NumpyEncoder
from octa_mosaic.modules.experiments.mosaicking_creation import (
    TemplateMatchingEvaluatingEdges,
)
from octa_mosaic.modules.experiments.mosaicking_optimization import DEProcess
from octa_mosaic.modules.optimization.differential_evolution import (
    DifferentialEvolutionParams,
)
from octa_mosaic.modules.optimization.problem import TransformConfig
from octa_mosaic.utils.constants import DATASET_PATH, EXPERIMENTS_PATH


# TODO: Move to a module
class ImagePreprocess:
    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.RandomState(seed)

    def crop(self, image: np.ndarray, px: int) -> np.ndarray:
        return image[px:-px, px:-px]

    def add_gaussian_noise(self, image: np.ndarray, sigma: float) -> np.ndarray:
        noise = self._rng.normal(0, sigma, image.shape)
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy_image

    def add_salt_pepper_noise(self, image: np.ndarray, prob: float) -> np.ndarray:
        noisy_image = np.copy(image)
        salt_mask = self._rng.rand(*image.shape) < (prob / 2)
        pepper_mask = self._rng.rand(*image.shape) < (prob / 2)
        noisy_image[salt_mask] = 255
        noisy_image[pepper_mask] = 0
        return noisy_image


# TODO: Move to a module
@dataclass
class InitialPopulationConfig:
    function: str = "tf_level_single_gen"  # tf_level_single_gen, tf_level
    x: float = 1.5
    popsize: int = 1000
    sigma: float = 0.125

    def fun_params(self) -> Dict[str, Any]:
        return {"x": self.x, "sigma": self.sigma, "popsize": self.popsize}


def run_test(
    cases_list: List[DatasetCase],
    objective_function_data: Dict[str, Any],
    transformation_config: TransformConfig,
    initial_population_config: InitialPopulationConfig,
    de_params: DifferentialEvolutionParams,
    image_preprocess_config: Dict,
):
    solutions_dict = {}
    reports_dict = {}

    image_preprocessing = ImagePreprocess(seed=de_params.seed)
    tm_procedure = TemplateMatchingEvaluatingEdges("template_matching_register")
    de_procedure = DEProcess("differential_evolution")
    local_search = initialize_population.PopulationBasedOnLocalSearch(
        seed=de_params.seed
    )
    for case in cases_list:
        # 0) Get images from case and preprocess
        images_list = case.get_images()
        if image_preprocess_config["gaussian_sigma"] > 0:
            images_list = [
                image_preprocessing.add_gaussian_noise(
                    img, image_preprocess_config["gaussian_sigma"]
                )
                for img in images_list
            ]

        if image_preprocess_config["salt_paper_prob"] > 0:
            images_list = [
                image_preprocessing.add_salt_pepper_noise(
                    img, image_preprocess_config["salt_paper_prob"]
                )
                for img in images_list
            ]

        if image_preprocess_config["crop_px"] > 0:
            images_list = [
                image_preprocessing.crop(img, image_preprocess_config["crop_px"])
                for img in images_list
            ]

        # 1) TM: First aproximation
        tm_mosaic, tm_report = tm_procedure.run(
            images_list, objective_function_data["args"]
        )

        # 2) DE: Optimization
        args = objective_function_data["args"]
        fobj_args = (tm_mosaic, args["border_width_list"], args["border_weight_list"])

        # 2.1) Bounds
        transformation_bounds = optimization_utils.affine_bounds(
            (0, 0),
            trans_bound=transformation_config.translation,
            scale_bound=transformation_config.scale,
            rot_bound=transformation_config.rotation,
            shear_bound=transformation_config.shear,
        )

        bounds = np.tile(transformation_bounds, (tm_mosaic.n_images(), 1))

        # 2.2) Initial population
        x0 = np.ones(tm_mosaic.n_images() * 6) * 0.5

        if initial_population_config.function == "tf_level_single_gen":
            initial_population = local_search.tf_level_single_gen(
                tm_mosaic.n_images(), **initial_population_config.fun_params()
            )
        elif initial_population_config.function == "tf_level":
            initial_population = local_search.tf_level(
                tm_mosaic.n_images(), **initial_population_config.fun_params()
            )
        else:
            raise ValueError("Uknown population initializer function.")

        initial_population = np.vstack((x0, initial_population))  # Add TM vector
        initial_population_filtered = initialize_population.select_best_individuals(
            indvs_to_select=de_params.popsize,
            population=initial_population,
            bounds=bounds,
            f_obj=objective_function_data["fobj"],
            args=fobj_args,
            n_workers=de_params.cores,
        )

        # 2.3) DE Params
        de_mosaic, de_report = de_procedure.run(
            tm_mosaic,
            objective_function_data["fobj"],
            fobj_args,
            bounds,
            de_params,
            initial_population_filtered,
        )

        # 3) Save data
        report = {**tm_report, **de_report}
        solutions_dict[case.get_ID()] = de_mosaic
        reports_dict[case.get_ID()] = report

    return solutions_dict, reports_dict


def main():
    # TODO: Move configuration to a file
    # Configuration
    PREPROCESS_CONFIG = {
        "crop_px": 1,
        "gaussian_sigma": 0,
        "salt_paper_prob": 0,
    }

    OBJECTIVE_FUNCTION = {
        "fobj": optimization_utils.objective_function_multi_edge_optimized,
        "args": {
            "border_width_list": [5, 10, 15],
            "border_weight_list": [0.33, 0.33, 0.33],
        },
    }

    TRANSFORMATION_CONFIG = TransformConfig(
        translation=20, scale=0.1, rotation=10, shear=5
    )

    INITIAL_POPULATION_CONFIG = InitialPopulationConfig(
        function="tf_level_single_gen",  # tf_level_single_gen, tf_level
        x=1.5,
        popsize=1000,
        sigma=0.125,
    )

    DE_HPARAMS = DifferentialEvolutionParams(
        F=0.2,
        C=0.75,
        generations=100,
        strategy="rand1bin",
        popsize=200,
        seed=16,
        fitness_population_std_tol=0.005,
        cores=8,
        display_progress_bar=True,
    )

    # Load data
    dataset = Dataset(DATASET_PATH)

    # Dataset experiment
    FILENAME = "de_rand1bin_cropped_guassian"

    # Run experiments

    datetime_id = int(datetime.datetime.now().strftime(r"%Y%m%d%H%M%S"))

    current_experiment_path = EXPERIMENTS_PATH / FILENAME
    current_experiment_path.mkdir(parents=True, exist_ok=True)

    test_sol, test_report = run_test(
        dataset.get_cases()[:1],
        OBJECTIVE_FUNCTION,
        TRANSFORMATION_CONFIG,
        INITIAL_POPULATION_CONFIG,
        DE_HPARAMS,
        PREPROCESS_CONFIG,
    )

    # Save config
    config = {
        "image_preprocessing": PREPROCESS_CONFIG,
        "objective_function": OBJECTIVE_FUNCTION,
        "transforms_config": TRANSFORMATION_CONFIG.__dict__,
        "population_initializer": INITIAL_POPULATION_CONFIG.__dict__,
        "de_hparams": DE_HPARAMS.__dict__,
    }
    with open(current_experiment_path / "config.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False)

    # Save mosaics
    serialized_mosaics = {
        case_id: mosaic.to_dict() for case_id, mosaic in test_sol.items()
    }
    with open(current_experiment_path / f"mosaics.pkl", "wb") as f:
        pkl.dump(serialized_mosaics, f)

    # Save report
    with open(current_experiment_path / f"reports.json", "w") as f:
        json.dump(test_report, f, indent=2, cls=NumpyEncoder)


if __name__ == "__main__":
    main()