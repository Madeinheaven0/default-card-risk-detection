import sys
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[2]))

from feast import FileSource
from config import USED_DATASET_PATH

customer_params_source = FileSource(
    name="customer source params",
    path=str(USED_DATASET_PATH),
    timestamp_field="event_timestamp",
)