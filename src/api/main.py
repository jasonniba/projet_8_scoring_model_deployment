from pathlib import Path
import time

import mlflow
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


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

# Liste des 401 variables attendues par le modèle
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

    received_features = set(request.features.keys())
    expected_features = set(EXPECTED_FEATURES)

    # Variables manquantes
    missing_features = expected_features - received_features

    # Variables inconnues
    unknown_features = received_features - expected_features

    # Validation des données reçues
    if missing_features or unknown_features:
        raise HTTPException(
            status_code=422,
            detail={
                "missing_features_count": len(missing_features),
                "missing_features_sample": sorted(missing_features)[:10],
                "unknown_features": sorted(unknown_features)[:10],
            },
        )

    # Mise dans l'ordre exact attendu par le modèle
    client_df = pd.DataFrame(
        [[request.features[feature] for feature in EXPECTED_FEATURES]],
        columns=EXPECTED_FEATURES,
    )

    # Mesure du temps d'inférence
    start_time = time.perf_counter()

    probabilities = model.predict_proba(client_df)[0]
    prediction = model.predict(client_df)[0]

    inference_time_ms = (time.perf_counter() - start_time) * 1000

    # Retrouver la position de la classe 1 = défaut
    classes = list(model.named_steps["classifier"].classes_)
    default_class_index = classes.index(1)

    default_probability = probabilities[default_class_index]

    return {
        "prediction": int(prediction),
        "default_probability": float(default_probability),
        "inference_time_ms": round(inference_time_ms, 3),
    }