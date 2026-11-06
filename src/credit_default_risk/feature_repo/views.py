from datetime import timedelta
from feast import FeatureView, Field
from feast.types import Float32, Int64, Int32, String

from entities import customer
from features import customer_params_source

params_customer = FeatureView(
    name="customer_params",
    entities=[customer],
    ttl=timedelta(1),
    schema=[
        Field(name="rn", dtype=Int32),
        Field(name="pre_since_opened", dtype=Int32),
        Field(name="pre_since_confirmed", dtype=Int32),
        Field(name="pre_loans_credit_limit", dtype=Int32),
        Field(name="pre_loans_credit_cost_rate", dtype=Int32),
        Field(name="paym_mean_status", dtype=Float32),
        Field(name="paym_count_incidents", dtype=Int32),
        Field(name="paym_incidents_over_mean", dtype=Float32),
        Field(name="total_zero_loans_count", dtype=Int32),
        Field(name="paym_trend_3m_vs_12m", dtype=Float32),
        Field(name="paym_hist_mean_1_23", dtype=Float32),
        Field(name="total_delinquency_score", dtype=Int32),
        Field(name="count_paym_status_0_24m", dtype=Int32),
        Field(name="count_paym_status_2_24m", dtype=Int32),
        Field(name="count_paym_status_3_24m", dtype=Int32),
        Field(name="enc_loans_account_holder_type", dtype=Int32),
        Field(name="enc_loans_credit_status", dtype=Int32),
        Field(name="enc_loans_credit_type", dtype=Int32),
        Field(name="enc_loans_account_cur", dtype=Int32),
        Field(name="pclose_flag", dtype=Int32),
        Field(name="absolute_zero_defect", dtype=Int32),
        Field(name="paym_latest_status", dtype=Int32),
        Field(name="paym_max_status", dtype=Int32),
        Field(name="is_zero_loans5", dtype=Int32),
        Field(name="is_zero_loans530", dtype=Int32),
        Field(name="is_zero_loans3060", dtype=Int32),
        Field(name="is_zero_loans6090", dtype=Int32),
        Field(name="is_zero_loans90", dtype=Int32),
        Field(name="is_zero_util", dtype=Int32),
        Field(name="is_zero_over2limit", dtype=Int32),
        Field(name="is_zero_maxover2limit", dtype=Int32),
        Field(name="pre_loans_next_pay_summ", dtype=Int32),
        Field(name="pre_loans_outstanding", dtype=Int32)
    ],
    online=True,
    source=customer_params_source
)