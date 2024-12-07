import glob
import os
from pathlib import Path
from typing import List

import cv2
import numpy as np

IMAGE_EXTENSION = ".jpeg"


class DatasetCase:
    def __init__(self, case_path: str):
        self.case_path = case_path

    def get_filenames(self, only_filename: bool = False) -> List[str]:
        filepath_list = sorted(glob.glob(f"{self.case_path}/*{IMAGE_EXTENSION}"))

        if not only_filename:
            return filepath_list

        filenames_list = [filepath.split("/")[-1] for filepath in filepath_list]
        return filenames_list

    def load_image(self, filename: str) -> np.ndarray:
        filepath = f"{self.case_path}/{filename}"
        if not filepath.endswith(IMAGE_EXTENSION):
            filepath += IMAGE_EXTENSION

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found.")

        return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    def get_images(self) -> List[np.ndarray]:
        filenames = self.get_filenames()
        images_list = [
            cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in filenames
        ]
        return images_list

    def get_ID(self) -> str:
        splited = Path(self.case_path).parts
        # ..., '8', 'ojo derecho', 'sup', ''
        patient = splited[-3]
        eye = splited[-2]
        return f"{patient} ({eye})"

    def __repr__(self) -> str:
        return f"Case({self.get_ID()})"
