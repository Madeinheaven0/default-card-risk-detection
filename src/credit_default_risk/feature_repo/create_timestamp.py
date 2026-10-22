from datetime import datetime
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import polars as pl
from src.config import USED_DATASET_PATH


def create_timestamp():
    df = pl.read_parquet(USED_DATASET_PATH)
    df = df.with_columns(
        pl.lit(datetime(2026, 1, 1, 0, 0, 0)).alias("event_timestamp")
    )

    df.write_parquet(USED_DATASET_PATH)


if __name__ == "__main__":
    create_timestamp()
