from pathlib import Path

import pytest
import polars as pl


TESTS_DIR = Path(__file__).parent

@pytest.fixture
def dataset():
    file_path = TESTS_DIR / "conftest_data.parquet"
    return pl.read_parquet(file_path)