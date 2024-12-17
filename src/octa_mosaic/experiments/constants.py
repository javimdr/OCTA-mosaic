from pathlib import Path

# Folder constants, useful for jupyter notebooks
PROJECT_ROOT = Path(__file__).parents[3]

ARTIFACTS_PATH = PROJECT_ROOT / "artifacts"
DATA_PATH = PROJECT_ROOT / "data"
PAPER_PATH = PROJECT_ROOT / "paper"

EXPERIMENTS_PATH = ARTIFACTS_PATH / "experiments"
DATASET_PATH = DATA_PATH / "octa_mosaicking_dataset"
DATASET_URL = "https://zenodo.org/records/14333858/files/OCTA-MosaickingDataset.zip"
