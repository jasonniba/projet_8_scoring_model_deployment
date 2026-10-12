import subprocess
import sys
import time

import requests
import pytest


# =========================================================
# CONFIGURATION
# =========================================================

API_HOST = "127.0.0.1"
API_PORT = 8001

BASE_URL = (
    f"http://{API_HOST}:{API_PORT}"
)

VALID_CLIENT_ID = 135970

UNKNOWN_CLIENT_ID = 999999999


# =========================================================
# SERVEUR FASTAPI
# =========================================================

@pytest.fixture(
    scope="session",
    autouse=True,
)
def api_server():

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.api.main:app",
            "--host",
            API_HOST,
            "--port",
            str(API_PORT),
        ],

        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # -----------------------------------------------------
    # ATTENDRE QUE L'API SOIT DISPONIBLE
    # -----------------------------------------------------

    ready = False

    for _ in range(30):

        try:

            response = requests.get(
                f"{BASE_URL}/health",
                timeout=1,
            )

            if response.status_code == 200:

                ready = True
                break

        except requests.RequestException:
            pass

        time.sleep(1)


    if not ready:

        process.terminate()

        stdout, stderr = (
            process.communicate(
                timeout=5
            )
        )

        raise RuntimeError(
            "\nL'API n'a pas démarré.\n\n"
            f"STDOUT:\n{stdout}\n\n"
            f"STDERR:\n{stderr}"
        )


    yield


    # -----------------------------------------------------
    # ARRET DU SERVEUR APRES LES TESTS
    # -----------------------------------------------------

    process.terminate()

    try:

        process.wait(
            timeout=5
        )

    except subprocess.TimeoutExpired:

        process.kill()


# =========================================================
# TEST HEALTH
# =========================================================

def test_health():

    response = requests.get(
        f"{BASE_URL}/health",
        timeout=10,
    )

    assert response.status_code == 200

    data = response.json()

    assert data[
        "status"
    ] == "ok"

    assert data[
        "model_loaded"
    ] is True

    assert data[
        "model_type"
    ] == "ONNX Runtime"

    assert data[
        "expected_features"
    ] == 401

    assert data[
        "clients_available"
    ] == 10000

    assert data[
        "input_type"
    ] == "client_id"


# =========================================================
# TEST CLIENTS
# =========================================================

def test_clients():

    response = requests.get(
        f"{BASE_URL}/clients",
        timeout=10,
    )

    assert response.status_code == 200

    data = response.json()

    assert data[
        "available_clients"
    ] == 10000

    assert len(
        data[
            "examples"
        ]
    ) > 0


# =========================================================
# TEST PREDICTION VALIDE
# =========================================================

def test_predict_valid_client():

    response = requests.post(
        f"{BASE_URL}/predict",

        json={
            "client_id":
                VALID_CLIENT_ID
        },

        timeout=30,
    )

    assert response.status_code == 200

    data = response.json()

    assert data[
        "client_id"
    ] == VALID_CLIENT_ID

    assert data[
        "prediction"
    ] in [
        0,
        1,
    ]

    assert (
        0.0
        <= data[
            "default_probability"
        ]
        <= 1.0
    )

    assert data[
        "inference_time_ms"
    ] >= 0

    assert data[
        "request_duration_ms"
    ] >= 0


# =========================================================
# TEST CLIENT INCONNU
# =========================================================

def test_predict_unknown_client():

    response = requests.post(
        f"{BASE_URL}/predict",

        json={
            "client_id":
                UNKNOWN_CLIENT_ID
        },

        timeout=30,
    )

    assert response.status_code == 404

    data = response.json()

    assert (
        "introuvable"
        in data[
            "detail"
        ][
            "message"
        ]
    )


# =========================================================
# TEST PAYLOAD INVALIDE
# =========================================================

def test_predict_invalid_payload():

    response = requests.post(
        f"{BASE_URL}/predict",

        json={},

        timeout=10,
    )

    assert response.status_code == 422