"""HTTP-level tests for FastAPI endpoints."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "lgbm" in body["predictor"] or "ebm" in body["predictor"]


def test_predict_minimal_input():
    r = client.post("/predict", json={"TransactionAmt": 250.0})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_proba"] <= 1.0
    assert body["predicted_label"] in (0, 1)
    assert body["model"] in ("lgbm", "ebm")


def test_predict_accepts_extra_fields():
    payload = {
        "TransactionAmt": 999,
        "card4": "visa",
        "id_31": "mobile safari 11.0",
        "id_30": "iOS 11.1.2",
        "C14": 1.0,
        "V258": 0.5,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200


def test_explain_returns_contributions():
    payload = {"TransactionAmt": 100, "id_30": "Android 7.0", "id_14": -300.0}
    r = client.post("/explain", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "contributions" in body
    assert len(body["contributions"]) > 0
    first = body["contributions"][0]
    assert {"feature", "value", "contribution"} <= first.keys()


def test_predict_by_id_known():
    """Lookup features for a real TransactionID and predict."""
    r = client.post("/predict_by_id", json={"transaction_id": 2987004})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_proba"] <= 1.0
    assert body["model"] in ("lgbm", "ebm")


def test_predict_by_id_unknown_returns_404():
    r = client.post("/predict_by_id", json={"transaction_id": 99_999_999})
    assert r.status_code == 404


def test_explain_by_id_known():
    r = client.post("/explain_by_id", json={"transaction_id": 2987004})
    assert r.status_code == 200
    body = r.json()
    assert len(body["contributions"]) > 0


def test_predict_rejects_negative_amount():
    """Great Expectations validation should reject TransactionAmt < 0."""
    r = client.post("/predict", json={"TransactionAmt": -50})
    assert r.status_code == 422
    body = r.json()
    assert "validation_errors" in body["detail"]


def test_predict_rejects_invalid_device_type():
    r = client.post("/predict", json={"DeviceType": "tablet"})
    assert r.status_code == 422


def test_predict_rejects_out_of_range_timezone():
    r = client.post("/predict", json={"id_14": 9999})
    assert r.status_code == 422
