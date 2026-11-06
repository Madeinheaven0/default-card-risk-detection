from pydantic import BaseModel, Field
from typing import Optional


class CreditDataInput(BaseModel):
    """Pydantic model containing all numerical and categorical features required for credit default inference."""

    # Identification
    id: Optional[int] = Field(None, description="Client or loan ID")

    # Numerical & Engineered Features
    pre_since_opened: Optional[int] = Field(None, description="Months since account opened")
    pre_since_confirmed: Optional[int] = Field(None, description="Months since confirmation")
    pre_loans_credit_limit: Optional[float] = Field(None, description="Credit limit")
    pre_loans_credit_cost_rate: Optional[float] = Field(None, description="Credit cost rate")
    paym_mean_status: Optional[float] = Field(None, description="Mean payment status")
    paym_count_incidents: Optional[int] = Field(None, description="Count of payment incidents")
    paym_incidents_over_mean: Optional[float] = Field(None, description="Ratio of incidents over mean status")
    total_zero_loans_count: Optional[int] = Field(None, description="Total zero loans count")
    paym_trend_3m_vs_12m: Optional[float] = Field(None, description="Payment trend 3m vs 12m")
    paym_hist_mean_1_23: Optional[float] = Field(None, description="Historical payment mean (months 1-23)")
    total_delinquency_score: Optional[int] = Field(None, description="Total delinquency score")
    count_paym_status_0_24m: Optional[int] = Field(None, description="Count of status 0 in 24m")
    count_paym_status_2_24m: Optional[int] = Field(None, description="Count of status 2 in 24m")
    count_paym_status_3_24m: Optional[int] = Field(None, description="Count of status 3 in 24m")

    # Categorical & Flag Features
    enc_loans_account_holder_type: Optional[int] = Field(None, description="Account holder type encoding")
    enc_loans_credit_status: Optional[int] = Field(None, description="Credit status encoding")
    enc_loans_credit_type: Optional[int] = Field(None, description="Credit type encoding")
    enc_loans_account_cur: Optional[int] = Field(None, description="Account currency encoding")
    pclose_flag: Optional[int] = Field(None, description="Close flag")
    absolute_zero_defect: Optional[int] = Field(None, description="Absolute zero defect flag")
    paym_latest_status: Optional[int] = Field(None, description="Latest payment status")
    paym_max_status: Optional[int] = Field(None, description="Max payment status")
    is_zero_loans5: Optional[int] = Field(None, description="Zero loans 5 flag")
    is_zero_loans530: Optional[int] = Field(None, description="Zero loans 5-30 flag")
    is_zero_loans3060: Optional[int] = Field(None, description="Zero loans 30-60 flag")
    is_zero_loans6090: Optional[int] = Field(None, description="Zero loans 60-90 flag")
    is_zero_loans90: Optional[int] = Field(None, description="Zero loans 90+ flag")
    is_zero_util: Optional[int] = Field(None, description="Zero utilization flag")
    is_zero_over2limit: Optional[int] = Field(None, description="Zero over-limit flag")
    is_zero_maxover2limit: Optional[int] = Field(None, description="Zero max over-limit flag")
    pre_loans_next_pay_summ: Optional[int] = Field(None, description="Next payment summary")
    pre_loans_outstanding: Optional[float] = Field(None, description="Outstanding loans amount")

    class Config:
        arbitrary_types_allowed = True


class PredictionResponse(BaseModel):
    """Pydantic model for prediction response."""
    default_probability: float = Field(..., description="Predicted probability of default")
    prediction: int = Field(..., description="Binary classification label (0 or 1)")
    risk_level: str = Field(..., description="Categorized risk level (Low, Medium, High)")