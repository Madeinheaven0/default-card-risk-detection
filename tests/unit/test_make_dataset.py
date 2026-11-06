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

def test_create_paym_features(dataset):
    df = del_useless_features(dataset)
    df = create_paym_features(df)

    assert "count_paym_status_0_24m" in df.columns
    assert "count_paym_status_2_24m" in df.columns
    assert "count_paym_status_3_24m" in df.columns
    assert "paym_hist_mean_1_23" in df.columns
    assert "paym_hist_max_1_23" in df.columns
    assert "paym_trend_3m_vs_12m" in df.columns

def test_return_final_dataset(dataset):
    df = return_final_dataset(dataset)

    features = [
        "rn",
        "pre_since_opened",
        "pre_since_confirmed",
        "pre_pterm",
        "pre_till_pclose",
        "pre_loans_credit_limit",
        "pre_loans_credit_cost_rate",
        "paym_mean_status",
        "paym_count_incidents",
        "paym_incidents_over_mean",
        "total_zero_loans_count",
        "paym_trend_3m_vs_12m",
        "paym_hist_mean_1_23",
        "total_delinquency_score",
        "count_paym_status_0_24m",
        "count_paym_status_2_24m",
        "count_paym_status_3_24m",
        "enc_loans_account_holder_type",
        "enc_loans_credit_status",
        "enc_loans_credit_type",
        "enc_loans_account_cur",
        "pclose_flag",
        "absolute_zero_defect",
        "paym_latest_status",
        "paym_max_status",
        "is_zero_loans5",
        "is_zero_loans530",
        "is_zero_loans3060",
        "is_zero_loans6090",
        "is_zero_loans90",
        "is_zero_util",
        "is_zero_over2limit",
        "is_zero_maxover2limit",
        "pre_loans_next_pay_summ",
        "pre_loans_outstanding",
        "pre_loans_max_overdue_sum",
    ]

    assert features == df.columns