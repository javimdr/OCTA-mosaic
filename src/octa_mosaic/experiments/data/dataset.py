import glob
import os
import tempfile
import zipfile
from typing import List

import requests

from octa_mosaic.experiments.constants import DATASET_PATH, DATASET_URL
from octa_mosaic.experiments.data.dataset_case import DatasetCase


class Dataset:
    """
    A class to manage [OCTA-Mosaicking Dataset](https://zenodo.org/records/14333858)
    for experiments, including downloading it if it is not available locally.
    """

    def __init__(self, dataset_path: str = None, dataset_url: str = None):
        """
        Initializes the Dataset class. Downloads and extracts the dataset if it's not
        already available locally.

        Args:
            dataset_path (str, optional): Path to the dataset. Defaults to DATASET_PATH.
            dataset_url (str, optional): URL to download the dataset. Defaults to DATASET_URL.

        Raises:
            ValueError: If the dataset path does not exist and no URL is provided.
        """
        self._dataset_path = dataset_path or DATASET_PATH
        dataset_url = dataset_url or DATASET_URL

        if not os.path.exists(self._dataset_path):
            if not dataset_url:
                msg = "Dataset path does not exist and no dataset URL was provided."
                raise ValueError(msg)
            print(f"Dataset not found at {self._dataset_path}. Downloading...")
            self._download_and_extract(dataset_url, self._dataset_path)

        self._cases_path = [
            path
            for path in glob.glob(f"{self._dataset_path}/**/", recursive=True)
            if "eye" in path
        ]

        self._cases = [DatasetCase(path) for path in sorted(self._cases_path)]

    def _download_and_extract(self, url: str, extract_to: str) -> None:
        """
        Downloads a ZIP file from the given URL and extracts it to the specified location.

        Args:
            url (str): URL to download the ZIP file from.
            extract_to (str): Directory to extract the contents of the ZIP file.

        Raises:
            RuntimeError: If the download fails due to an HTTP error.
        """
        response = requests.get(url, stream=True)
        with tempfile.NamedTemporaryFile(suffix=".zip") as tmp_file:
            if response.status_code == 200:
                with open(tmp_file.name, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)

            else:
                msg = f"Failed to download dataset from {url}. HTTP status code: {response.status_code}"
                raise RuntimeError(msg)

            with zipfile.ZipFile(tmp_file.name, "r") as zip_ref:
                zip_ref.extractall(extract_to)

    def get_cases(self) -> List[DatasetCase]:
        """
        Retrieves all cases in the dataset.

        Returns:
            List[DatasetCase]: A list of `DatasetCase` objects.
        """
        return list(self._cases)

    def get_cases_paths(self) -> List[str]:
        """
        Retrieves the paths to all cases in the dataset.

        Returns:
            List[str]: A list of paths to cases.
        """
        return list(self._cases_path)

    def get_case(self, case_id: str) -> DatasetCase:
        """
        Retrieves a specific case by its ID.

        Args:
            case_id (str): The ID of the case to retrieve.

        Returns:
            DatasetCase: The case corresponding to the provided ID.
        """
        for case in self.get_cases():
            if case.get_ID() == case_id:
                return case
        raise ValueError(f"Case {case_id!r} not found.")

    def __str__(self) -> str:
        return f"Dataset({len(self.get_cases())} test cases)"


dataset = Dataset()
print(dataset._cases_path)
print([case.get_ID() for case in dataset.get_cases()])
print(len(dataset.get_cases()[0].get_images()))
print(dataset.get_cases()[0].get_images()[0].shape)
