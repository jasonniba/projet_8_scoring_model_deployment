import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.db.database import (
    save_error,
    save_prediction,
)

from src.monitoring.logger import (
    log_event,
)


# =========================================================
# APPLICATION FASTAPI
# =========================================================

app = FastAPI(
    title="Credit Scoring API",
    description=(
        "API de scoring de crédit utilisant "
        "ONNX Runtime."
    ),
    version="3.0.0",
)


# =========================================================
# CHEMINS
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "credit_scoring_model_onnx"
)

ONNX_MODEL_PATH = (
    MODEL_DIR
    / "model.onnx"
)

METADATA_PATH = (
    MODEL_DIR
    / "metadata.json"
)

SCORING_CLIENTS_PATH = (
    MODEL_DIR
    / "scoring_clients.parquet"
)


# =========================================================
# CHARGEMENT DES METADONNEES
# =========================================================

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8",
) as file:

    metadata = json.load(file)


EXPECTED_FEATURES = metadata[
    "features"
]

MODEL_CLASSES = np.asarray(
    metadata["classes"]
)

DEFAULT_CLASS_INDEX = (
    list(MODEL_CLASSES)
    .index(1)
)


# =========================================================
# CHARGEMENT DES CLIENTS
# =========================================================

scoring_clients = pd.read_parquet(
    SCORING_CLIENTS_PATH
)


if "SK_ID_CURR" not in scoring_clients.columns:

    raise RuntimeError(
        "La colonne SK_ID_CURR est absente "
        "de scoring_clients.parquet."
    )


scoring_clients = (
    scoring_clients
    .set_index("SK_ID_CURR")
)


# =========================================================
# VERIFICATION DES FEATURES
# =========================================================

missing_model_features = (
    set(EXPECTED_FEATURES)
    - set(scoring_clients.columns)
)


if missing_model_features:

    raise RuntimeError(
        "Certaines features attendues par le modèle "
        "sont absentes de scoring_clients.parquet."
    )


# =========================================================
# CHARGEMENT DU MODELE ONNX
# =========================================================

onnx_session = ort.InferenceSession(
    str(ONNX_MODEL_PATH),
    providers=[
        "CPUExecutionProvider"
    ],
)


ONNX_INPUT_NAME = (
    onnx_session
    .get_inputs()[0]
    .name
)


# =========================================================
# SCHEMA D'ENTREE
# =========================================================

class PredictionRequest(BaseModel):

    client_id: int


# =========================================================
# ROUTE RACINE
# =========================================================

@app.get("/")
def root():

    return {
        "message":
            "Credit Scoring API is running",

        "inference_engine":
            "ONNX Runtime",

        "input":
            "client_id",
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status":
            "ok",

        "model_loaded":
            True,

        "model_type":
            "ONNX Runtime",

        "expected_features":
            len(EXPECTED_FEATURES),

        "clients_available":
            len(scoring_clients),

        "input_type":
            "client_id",
    }


# =========================================================
# EXEMPLES DE CLIENTS
# =========================================================

@app.get("/clients")
def clients():

    client_ids = (
        scoring_clients
        .index
        .tolist()[:10]
    )

    return {
        "available_clients":
            len(scoring_clients),

        "examples":
            [
                int(client_id)
                for client_id
                in client_ids
            ],
    }


# =========================================================
# PREDICTION
# =========================================================

@app.post("/predict")
def predict(
    request: PredictionRequest,
):

    request_start = (
        time.perf_counter()
    )

    client_id = request.client_id


    # =====================================================
    # VERIFICATION CLIENT
    # =====================================================

    if client_id not in scoring_clients.index:

        request_duration_ms = (
            time.perf_counter()
            - request_start
        ) * 1000


        error_message = (
            f"Client {client_id} introuvable."
        )


        save_error(
            features={
                "client_id":
                    client_id
            },

            status_code=404,

            error_message=(
                error_message
            ),

            request_duration_ms=round(
                request_duration_ms,
                3,
            ),
        )


        log_event(
            {
                "event":
                    "prediction_error",

                "status_code":
                    404,

                "client_id":
                    client_id,

                "error_message":
                    error_message,

                "request_duration_ms":
                    round(
                        request_duration_ms,
                        3,
                    ),
            }
        )


        raise HTTPException(
            status_code=404,

            detail={
                "message":
                    error_message
            },
        )


    # =====================================================
    # RECUPERATION DES 401 FEATURES
    # =====================================================

    client_features = (
        scoring_clients
        .loc[
            client_id,
            EXPECTED_FEATURES,
        ]
    )


    # =====================================================
    # PREPARATION POUR ONNX
    # =====================================================

    client_array = (
        client_features
        .to_numpy(
            dtype=np.float32
        )
        .reshape(
            1,
            -1,
        )
    )


    # =====================================================
    # INFERENCE
    # =====================================================

    inference_start = (
        time.perf_counter()
    )


    outputs = onnx_session.run(
        None,
        {
            ONNX_INPUT_NAME:
                client_array
        },
    )


    inference_time_ms = (
        time.perf_counter()
        - inference_start
    ) * 1000


    prediction = int(
        np.asarray(
            outputs[0]
        )[0]
    )


    probabilities = np.asarray(
        outputs[1]
    )[0]


    default_probability = float(
        probabilities[
            DEFAULT_CLASS_INDEX
        ]
    )


    # =====================================================
    # FEATURES POUR LE MONITORING
    # =====================================================

    features_for_storage = {}


    for feature in EXPECTED_FEATURES:

        value = client_features[
            feature
        ]

        if pd.isna(value):

            features_for_storage[
                feature
            ] = None

        else:

            features_for_storage[
                feature
            ] = float(value)


    # =====================================================
    # TEMPS DE TRAITEMENT API
    # =====================================================

    request_duration_ms = (
        time.perf_counter()
        - request_start
    ) * 1000


    # =====================================================
    # POSTGRESQL
    # =====================================================

    save_prediction(
        features=features_for_storage,

        prediction=prediction,

        default_probability=(
            default_probability
        ),

        inference_time_ms=round(
            inference_time_ms,
            3,
        ),

        request_duration_ms=round(
            request_duration_ms,
            3,
        ),

        status_code=200,
    )


    # =====================================================
    # ELASTICSEARCH
    # =====================================================

    log_event(
        {
            "event":
                "prediction_success",

            "status_code":
                200,

            "client_id":
                client_id,

            "prediction":
                prediction,

            "default_probability":
                default_probability,

            "inference_time_ms":
                round(
                    inference_time_ms,
                    3,
                ),

            "request_duration_ms":
                round(
                    request_duration_ms,
                    3,
                ),

            "inference_engine":
                "onnxruntime",
        }
    )


    # =====================================================
    # REPONSE UTILISATEUR
    # =====================================================

    return {
        "client_id":
            client_id,

        "prediction":
            prediction,

        "default_probability":
            round(
                default_probability,
                6,
            ),

        "inference_time_ms":
            round(
                inference_time_ms,
                3,
            ),

        "request_duration_ms":
            round(
                request_duration_ms,
                3,
            ),
    }