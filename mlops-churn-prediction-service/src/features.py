import numpy as np

CONTRACTS = {"month-to-month": 0, "one-year": 1, "two-year": 2}

def make_training_data(samples: int = 800, seed: int = 42):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, samples)
    charges = rng.uniform(25, 120, samples)
    tickets = rng.integers(0, 9, samples)
    contract = rng.integers(0, 3, samples)
    risk = (0.55 * (tenure < 12) + 0.45 * (charges > 85) +
            0.6 * (tickets > 4) + 0.7 * (contract == 0) + rng.normal(0, 0.25, samples))
    churn = (risk > 1.05).astype(int)
    return np.column_stack([tenure, charges, tickets, contract]), churn

def encode_request(payload: dict):
    return [[
        payload["tenure_months"],
        payload["monthly_charges"],
        payload["support_tickets"],
        CONTRACTS[payload["contract_type"]],
    ]]
