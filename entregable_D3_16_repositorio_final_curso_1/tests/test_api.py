from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


VALID_CUSTOMER = {
    "customer_id": "NT-000001",
    "age": 32.0,
    "tenure_months": 27,
    "monthly_fee": 101.65,
    "total_spent": 2467.19,
    "support_calls": 0,
    "complaints": 0,
    "last_payment_delay": 3,
    "digital_usage_score": 80.4,
    "marketing_score": 91.0,
    "preferred_contact_hour": 15,
    "gender": "Femenino",
    "region": "Sur",
    "customer_segment": "Masivo",
    "contract_type": "Bianual",
    "internet_service": "Fibra",
    "tv_service": "No",
    "streaming_service": "Sí",
    "payment_method": "Efectivo",
}


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_valid_customer():
    response = client.post(
        "/predict",
        json=VALID_CUSTOMER,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["customer_id"] == "NT-000001"
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in [0, 1]


def test_predict_rejects_invalid_age():
    customer = VALID_CUSTOMER.copy()
    customer["age"] = -5

    response = client.post(
        "/predict",
        json=customer,
    )

    assert response.status_code == 422


def test_batch_rejects_non_csv():
    response = client.post(
        "/predict/batch",
        files={
            "file": (
                "customers.txt",
                b"hello world",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "Only CSV files are supported."
    )