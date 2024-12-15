import json
import pickle as pkl
from pathlib import Path

from tqdm import tqdm

from octa_mosaic.experiments.constants import DATASET_PATH, EXPERIMENTS_PATH
from octa_mosaic.experiments.data import Dataset
from octa_mosaic.experiments.encoders import NumpyEncoder
from octa_mosaic.modules import optimization_utils
from octa_mosaic.modules.experiments.mosaicking_creation import (
    TemplateMatchingEvaluatingEdges,
)

RESULTS_FILENAME = "template_matching"
RESULTS_PATH = EXPERIMENTS_PATH / RESULTS_FILENAME

FIRST_PAIR_FUNCTION = {
    "func": optimization_utils.calc_zncc_on_multiple_seamlines,
    "kwargs": {
        "widths": [10, 20, 30],
        "weights": [0.6, 0.3, 0.1],
    },
}


def check_output_path(output_filepath: Path) -> bool:
    if not output_filepath.exists():
        output_filepath.mkdir(parents=True)

    if output_filepath.exists():
        opt = None
        while opt not in ["y", "n"]:
            opt = input("Results files alredy exists. Do you want to override? [y/n]: ")

        if opt == "n":
            return False

    return True


def main():
    dataset = Dataset(DATASET_PATH)

    if not check_output_path(RESULTS_PATH):
        return

    results_filepath = RESULTS_PATH / f"mosaics.pkl"
    reports_filepath = RESULTS_PATH / f"reports.json"

    # RUN
    tm_procedure = TemplateMatchingEvaluatingEdges("template_matching_register")
    solutions_dict = {}
    reports_dict = {}
    for case in tqdm(dataset.get_cases()):
        case_id = case.get_ID()
        images_list = case.get_images()
        tm_mosaic, tm_report = tm_procedure.run(
            images_list,
            first_pair_func=FIRST_PAIR_FUNCTION["func"],
            first_pair_kwargs=FIRST_PAIR_FUNCTION["kwargs"],
        )
        solutions_dict[case_id] = tm_mosaic.to_dict()
        reports_dict[case_id] = tm_report

    with open(results_filepath, "wb") as f:
        pkl.dump(solutions_dict, f)

    with open(reports_filepath, "w") as f:
        json.dump(reports_dict, f, indent=2, cls=NumpyEncoder)


if __name__ == "__main__":
    main()
