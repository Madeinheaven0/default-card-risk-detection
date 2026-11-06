from fastapi import APIRouter, HTTPException, status
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from src.app.schemas import CreditDataInput, PredictionResponse
from src.config import MODEL_PATH  

router = APIRouter(
    prefix="/predict",
    tags=["Credit Risk Prediction"]
)

# Load the trained model pipeline globally to avoid reloading on every request
model_file = Path(MODEL_PATH)
if model_file.exists():
    pipeline = joblib.load(model_file)
else:
    pipeline = None


@router.post("/", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_credit_default(payload: CreditDataInput):
    """
    Endpoint to predict credit default risk based on user financial features.
    """
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Trained model artifact not found on disk."
        )

    try:
        # 1. Convert incoming Pydantic payload to a Pandas DataFrame
        input_data = pd.DataFrame([payload.dict()])

        # 2. Apply feature engineering if required (using your preprocessing logic)
        # Note: Ensure missing columns expected by the transformer/model are filled or handled gracefully.

        # 3. Predict probability and binary class
        probabilities = pipeline.predict_proba(input_data)
        default_prob = float(probabilities[0][1])  # Probability of class 1 (default)
        pred_label = int(pipeline.predict(input_data)[0])

        # 4. Assign a qualitative risk level
        if default_prob < 0.3:
            risk_level = "Low Risk"
        elif default_prob < 0.7:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"

        return PredictionResponse(
            default_probability=default_prob,
            prediction=pred_label,
            risk_level=risk_level
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference error: {str(e)}"
        )