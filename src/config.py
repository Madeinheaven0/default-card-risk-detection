from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_TRAIN_DATA_DIR = RAW_DATA_DIR / "loan_data" / "data" / "train_data"
TARGET_PATH = RAW_DATA_DIR / "loan_data" / "target.csv"
USED_DATASET_PATH = RAW_DATA_DIR / "loan_data" / "data" / "loan_data_2M.parquet"

CONFIG_DIR = ROOT_DIR / "config"
CONFIG_PATH = CONFIG_DIR / "config.yaml"
LOGGING_PATH = CONFIG_DIR / "logging.conf"

MODEL_DIR = ROOT_DIR / "models"
