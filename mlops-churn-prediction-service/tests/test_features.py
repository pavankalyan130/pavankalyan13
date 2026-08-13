from src.features import CONTRACTS, encode_request

def test_request_encoding():
    values = encode_request({
        "tenure_months": 12, "monthly_charges": 55.5,
        "support_tickets": 2, "contract_type": "one-year"
    })
    assert values == [[12, 55.5, 2, CONTRACTS["one-year"]]]
