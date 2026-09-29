from fastapi.testclient import TestClient

from src.api.main import app, EXPECTED_FEATURES


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_type"] == "Pipeline"
    assert data["expected_features"] == 401


def test_predict():
    features = {
        feature: 0.0
        for feature in EXPECTED_FEATURES
    }

    response = client.post(
        "/predict",
        json={"features": features}
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "default_probability" in data
    assert "inference_time_ms" in data

    assert data["prediction"] in [0, 1]
    assert 0 <= data["default_probability"] <= 1
    assert data["inference_time_ms"] >= 0


def test_predict_missing_features():
    response = client.post(
        "/predict",
        json={
            "features": {
                "AMT_CREDIT": 100000.0
            }
        }
    )

    assert response.status_code == 422

    data = response.json()

    assert "detail" in data
    assert data["detail"]["missing_features_count"] > 0