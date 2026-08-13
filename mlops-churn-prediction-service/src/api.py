from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .features import CONTRACTS, encode_request

MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "churn_model.joblib"
app = FastAPI(title="MLOps Churn Prediction Service", version="1.0.0")

class CustomerFeatures(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0, le=1000)
    support_tickets: int = Field(ge=0, le=50)
    contract_type: str

@app.get("/health")
def health():
    return {"model_ready": MODEL_PATH.exists()}

@app.post("/predict")
def predict(payload: CustomerFeatures):
    if payload.contract_type not in CONTRACTS:
        raise HTTPException(status_code=400, detail=f"contract_type must be one of {list(CONTRACTS)}")
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not trained. Run python src/train.py first.")
    model = joblib.load(MODEL_PATH)
    probability = float(model.predict_proba(encode_request(payload.model_dump()))[0][1])
    return {
        "churn_probability": round(probability, 3),
        "risk_level": "high" if probability >= 0.65 else "medium" if probability >= 0.35 else "low",
    }
