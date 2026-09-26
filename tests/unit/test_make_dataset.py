import polars as pl
import pytest

from src.data.make_dataset import *


def test_del_useless_features(dataset):
    useless_features = [
        "pre_fterm",
        "pre_till_fclose",
        "fclose_flag",
        "pre_loans_total_overdue",
        "id"
    ]

    df = del_useless_features(dataset)

    for features in useless_features:
        assert features not in df.columns

def test_create_paym_incidents_data(dataset):
    df = del_useless_features(dataset)
    df = create_paym_incidents_data(df)

    assert "paym_max_status" in df.columns
    assert "paym_mean_status" in df.columns
    assert "paym_count_incidents" in df.columns
    assert "paym_latest_status" in df.columns
    assert "paym_incidents_over_mean" in df.columns


def test_create_paym_features():
    pass