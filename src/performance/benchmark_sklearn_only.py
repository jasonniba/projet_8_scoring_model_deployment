import json
import time
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd


# =========================================================
# CHEMINS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "credit_scoring_model"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "validation_features.parquet"
)

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)

OUTPUT_PATH = (
    REPORTS_DIR
    / "benchmark_sklearn.json"
)


# =========================================================
# DEMARRAGE
# =========================================================

print("=" * 70)
print("BENCHMARK MLFLOW / SCIKIT-LEARN")
print("=" * 70)

print()
print("Chargement du modèle MLflow...", flush=True)


# =========================================================
# MODELE
# =========================================================

model = mlflow.sklearn.load_model(
    str(MODEL_PATH)
)

classes = np.asarray(
    model.named_steps[
        "classifier"
    ].classes_
)

print("Modèle chargé.")
print(
    "Features :",
    len(model.feature_names_in_)
)


# =========================================================
# DONNEES
# =========================================================

print()
print(
    "Chargement des données...",
    flush=True,
)

X = pd.read_parquet(
    VALIDATION_PATH
).iloc[:1000].copy()

print(
    "Clients testés :",
    len(X)
)


# =========================================================
# WARM-UP
# =========================================================

print()
print("Warm-up...")

for i in range(10):

    model.predict_proba(
        X.iloc[[i]]
    )

print("Warm-up terminé.")


# =========================================================
# BENCHMARK
# =========================================================

print()
print(
    "Benchmark en cours...",
    flush=True,
)

times = []
predictions = []
probabilities_class_1 = []


for i in range(len(X)):

    row = X.iloc[[i]]

    start = time.perf_counter()

    probabilities = (
        model.predict_proba(
            row
        )[0]
    )

    elapsed_ms = (
        time.perf_counter()
        - start
    ) * 1000

    prediction = classes[
        np.argmax(
            probabilities
        )
    ]

    times.append(
        elapsed_ms
    )

    predictions.append(
        int(prediction)
    )

    probabilities_class_1.append(
        float(
            probabilities[1]
        )
    )


times = np.asarray(
    times
)


# =========================================================
# RESULTATS
# =========================================================

result = {
    "engine": "MLflow / Scikit-learn",
    "requests": len(X),

    "mean_ms": float(
        times.mean()
    ),

    "p50_ms": float(
        np.percentile(
            times,
            50,
        )
    ),

    "p95_ms": float(
        np.percentile(
            times,
            95,
        )
    ),

    "p99_ms": float(
        np.percentile(
            times,
            99,
        )
    ),

    "predictions": predictions,

    "probabilities_class_1": (
        probabilities_class_1
    ),
}


REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        result,
        file,
        indent=2,
    )


print()
print("=" * 70)
print("RESULTATS SCIKIT-LEARN")
print("=" * 70)

print(
    f"Moyenne : "
    f"{result['mean_ms']:.3f} ms"
)

print(
    f"P50     : "
    f"{result['p50_ms']:.3f} ms"
)

print(
    f"P95     : "
    f"{result['p95_ms']:.3f} ms"
)

print(
    f"P99     : "
    f"{result['p99_ms']:.3f} ms"
)

print()
print(
    "Rapport :",
    OUTPUT_PATH
)

print("=" * 70)