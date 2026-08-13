# MLOps Churn Prediction Service

An end-to-end machine-learning service that trains a customer-churn model, evaluates it, saves the model artifact, and serves probability-based predictions through FastAPI.

## Architecture

```
Synthetic data → feature pipeline → model training → evaluation report → serialized artifact → prediction API
```

## Features
- Reproducible dataset generation and train/test split
- Feature scaling and Logistic Regression pipeline
- Evaluation metrics saved as JSON
- Serialized model artifact for consistent inference
- REST prediction API with input validation
- Docker setup and unit tests

## Tech stack
Python · scikit-learn · FastAPI · joblib · Docker · pytest

## Run locally

```bash
pip install -r requirements.txt
python src/train.py
uvicorn src.api:app --reload
```

Then send a request to `POST /predict`:

```json
{"tenure_months": 4, "monthly_charges": 89.0, "support_tickets": 5, "contract_type": "month-to-month"}
```
