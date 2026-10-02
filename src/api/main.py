import time
from pathlib import Path

import mlflow.sklearn
import numpy as np
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
        "API de scoring permettant de calculer "
        "le risque de défaut d'un client."
    ),
    version="1.0.0",
)


# =========================================================
# CHEMINS
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "credit_scoring_model"
)


# =========================================================
# CHARGEMENT DU MODELE
# =========================================================

model = mlflow.sklearn.load_model(
    str(MODEL_PATH)
)

EXPECTED_FEATURES = list(
    model.feature_names_in_
)

MODEL_CLASSES = list(
    model.named_steps[
        "classifier"
    ].classes_
)

DEFAULT_CLASS_INDEX = (
    MODEL_CLASSES.index(1)
)


# =========================================================
# SCHEMA D'ENTREE
# =========================================================

class PredictionRequest(BaseModel):

    features: dict[
        str,
        float | None,
    ]


# =========================================================
# ROUTE RACINE
# =========================================================

@app.get("/")
def root():

    return {
        "message": (
            "Credit Scoring API is running"
        )
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": True,
        "model_type": (
            type(model).__name__
        ),
        "expected_features": (
            len(EXPECTED_FEATURES)
        ),
    }


# =========================================================
# PREDICTION
# =========================================================

@app.post("/predict")
def predict(
    request: PredictionRequest,
):

    # -----------------------------------------------------
    # DEBUT DU TRAITEMENT
    # -----------------------------------------------------

    request_start = (
        time.perf_counter()
    )


    # -----------------------------------------------------
    # VALIDATION DES FEATURES
    # -----------------------------------------------------

    received_features = set(
        request.features.keys()
    )

    expected_features = set(
        EXPECTED_FEATURES
    )

    missing_features = (
        expected_features
        - received_features
    )

    unknown_features = (
        received_features
        - expected_features
    )


    # -----------------------------------------------------
    # REQUETE INVALIDE
    # -----------------------------------------------------

    if (
        missing_features
        or unknown_features
    ):

        request_duration_ms = (
            time.perf_counter()
            - request_start
        ) * 1000

        error_message = (
            "Missing or unknown features"
        )


        # Stockage PostgreSQL
        save_error(
            features=request.features,
            status_code=422,
            error_message=error_message,
            request_duration_ms=round(
                request_duration_ms,
                3,
            ),
        )


        # Log Elasticsearch
        log_event(
            {
                "event": (
                    "prediction_error"
                ),

                "status_code": 422,

                "error_message": (
                    error_message
                ),

                "request_duration_ms": (
                    round(
                        request_duration_ms,
                        3,
                    )
                ),

                "features": (
                    request.features
                ),

                "missing_features_count": (
                    len(
                        missing_features
                    )
                ),

                "unknown_features_count": (
                    len(
                        unknown_features
                    )
                ),
            }
        )


        # Réponse API 422
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    error_message
                ),

                "missing_features_count": (
                    len(
                        missing_features
                    )
                ),

                "unknown_features_count": (
                    len(
                        unknown_features
                    )
                ),

                "missing_features": sorted(
                    missing_features
                ),

                "unknown_features": sorted(
                    unknown_features
                ),
            },
        )


    # -----------------------------------------------------
    # CONSTRUCTION DU DATAFRAME
    # -----------------------------------------------------

    client_df = pd.DataFrame(
        [
            [
                request.features[
                    feature
                ]
                for feature
                in EXPECTED_FEATURES
            ]
        ],
        columns=EXPECTED_FEATURES,
    )


    # -----------------------------------------------------
    # INFERENCE
    # -----------------------------------------------------

    inference_start = (
        time.perf_counter()
    )


    # Une seule traversée du modèle
    probabilities = (
        model.predict_proba(
            client_df
        )[0]
    )


    # Classe correspondant à la probabilité maximale
    prediction_index = int(
        np.argmax(
            probabilities
        )
    )

    prediction = (
        MODEL_CLASSES[
            prediction_index
        ]
    )


    # Probabilité de la classe 1 = défaut
    default_probability = (
        probabilities[
            DEFAULT_CLASS_INDEX
        ]
    )


    inference_time_ms = (
        time.perf_counter()
        - inference_start
    ) * 1000


    # -----------------------------------------------------
    # DUREE DU TRAITEMENT AVANT MONITORING
    # -----------------------------------------------------

    request_duration_ms = (
        time.perf_counter()
        - request_start
    ) * 1000


    # -----------------------------------------------------
    # STOCKAGE POSTGRESQL
    # -----------------------------------------------------

    save_prediction(
        features=request.features,
        prediction=int(
            prediction
        ),
        default_probability=float(
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


    # -----------------------------------------------------
    # LOG ELASTICSEARCH
    # -----------------------------------------------------

    log_event(
        {
            "event": (
                "prediction_success"
            ),

            "status_code": 200,

            "prediction": int(
                prediction
            ),

            "default_probability": float(
                default_probability
            ),

            "inference_time_ms": round(
                inference_time_ms,
                3,
            ),

            "request_duration_ms": round(
                request_duration_ms,
                3,
            ),

            "features": (
                request.features
            ),
        }
    )


    # -----------------------------------------------------
    # REPONSE API
    # -----------------------------------------------------

    return {
        "prediction": int(
            prediction
        ),

        "default_probability": float(
            default_probability
        ),

        "inference_time_ms": round(
            inference_time_ms,
            3,
        ),

        "request_duration_ms": round(
            request_duration_ms,
            3,
        ),
    }