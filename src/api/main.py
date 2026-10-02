from pathlib import Path
import time

import mlflow
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.monitoring.logger import log_event

from src.db.database import save_prediction, save_error


app = FastAPI(
    title="Credit Scoring API",
    description="API de prédiction du risque de défaut de crédit",
    version="1.0.0",
)


# Racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Chemin vers le modèle
MODEL_PATH = PROJECT_ROOT / "models" / "credit_scoring_model"

# Chargement du modèle
model = mlflow.sklearn.load_model(str(MODEL_PATH))

# Liste des 401 variables attendues
EXPECTED_FEATURES = list(model.feature_names_in_)


class PredictionRequest(BaseModel):
    features: dict[str, float | None]


@app.get("/")
def root():
    return {
        "message": "Credit Scoring API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "model_type": type(model).__name__,
        "expected_features": len(EXPECTED_FEATURES),
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    # Début du traitement complet de la requête
    request_start = time.perf_counter()

    received_features = set(request.features.keys())
    expected_features = set(EXPECTED_FEATURES)

    missing_features = expected_features - received_features
    unknown_features = received_features - expected_features

    # Requête incorrecte
    if missing_features or unknown_features:

        request_duration_ms = (
            time.perf_counter() - request_start
        ) * 1000

        save_error(
            features=request.features,
            status_code=422,
            error_message="Missing or unknown features",
            request_duration_ms=round(request_duration_ms, 3),
        )

        log_event(
    {
        "event": "prediction_error",
        "status_code": 422,
        "error_message": "Missing or unknown features",
        "request_duration_ms": round(request_duration_ms, 3),
        "features": request.features,
        "missing_features_count": len(missing_features),
        "unknown_features_count": len(unknown_features),
    }
)
        raise HTTPException(
            status_code=422,
            detail={
                "missing_features_count": len(missing_features),
                "missing_features_sample": sorted(missing_features)[:10],
                "unknown_features": sorted(unknown_features)[:10],
            },
        )

    # Construction du DataFrame
    client_df = pd.DataFrame(
        [[request.features[feature] for feature in EXPECTED_FEATURES]],
        columns=EXPECTED_FEATURES,
    )

    # Temps d'inférence uniquement
    inference_start = time.perf_counter()

    probabilities = model.predict_proba(client_df)[0]
    prediction = model.predict(client_df)[0]

    inference_time_ms = (
        time.perf_counter() - inference_start
    ) * 1000

    # Probabilité de défaut = classe 1
    classes = list(model.named_steps["classifier"].classes_)
    default_class_index = classes.index(1)

    default_probability = probabilities[default_class_index]

    # Durée totale du traitement
    request_duration_ms = (
        time.perf_counter() - request_start
    ) * 1000

    # Stockage PostgreSQL
    save_prediction(
        features=request.features,
        prediction=int(prediction),
        default_probability=float(default_probability),
        inference_time_ms=round(inference_time_ms, 3),
        request_duration_ms=round(request_duration_ms, 3),
        status_code=200,
    )

    log_event(
    {
        "event": "prediction_success",
        "status_code": 200,
        "prediction": int(prediction),
        "default_probability": float(default_probability),
        "inference_time_ms": round(inference_time_ms, 3),
        "request_duration_ms": round(request_duration_ms, 3),
        "features": request.features,
    }
)
    return {
        "prediction": int(prediction),
        "default_probability": float(default_probability),
        "inference_time_ms": round(inference_time_ms, 3),
        "request_duration_ms": round(request_duration_ms, 3),
    }