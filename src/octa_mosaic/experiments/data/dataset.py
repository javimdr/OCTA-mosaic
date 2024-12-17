import glob
import os
from typing import List

from octa_mosaic.experiments.data.dataset_case import DatasetCase


class Dataset:

    def __init__(self, dataset_path: str):
        assert os.path.exists(dataset_path)
        self._dataset_path = dataset_path
        self._cases_path = [
            path
            for path in glob.glob(f"{self._dataset_path}/**/", recursive=True)
            if "eye" in path
        ]

        self._cases = [DatasetCase(path) for path in sorted(self._cases_path)]

    def get_cases(self) -> List[DatasetCase]:
        return list(self._cases)

    def get_cases_paths(self) -> List[str]:
        return list(self._cases_path)

    def get_case(self, case_id: str) -> DatasetCase:
        for case in self.get_cases():
            if case.get_ID() == case_id:
                return case
        raise ValueError(f"Case {case_id!r} not found.")

    def __str__(self) -> str:
        return f"Dataset({len(self.get_cases())} test cases)"
