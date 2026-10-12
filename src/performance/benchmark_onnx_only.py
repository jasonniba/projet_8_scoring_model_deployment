import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd


# =========================================================
# CHEMINS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "credit_scoring_model_onnx"
    / "model.onnx"
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
    / "benchmark_onnx.json"
)


# =========================================================
# DEMARRAGE
# =========================================================

print("=" * 70)
print("BENCHMARK ONNX RUNTIME")
print("=" * 70)

print()
print(
    "Chargement du modèle ONNX...",
    flush=True,
)


# =========================================================
# MODELE
# =========================================================

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=[
        "CPUExecutionProvider"
    ],
)

input_name = (
    session
    .get_inputs()[0]
    .name
)

print(
    "Input :",
    input_name
)

print(
    "Outputs :",
    [
        output.name
        for output
        in session.get_outputs()
    ]
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

    row = (
        X.iloc[[i]]
        .to_numpy(
            dtype=np.float32
        )
    )

    session.run(
        None,
        {
            input_name:
            row
        },
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

    row = (
        X.iloc[[i]]
        .to_numpy(
            dtype=np.float32
        )
    )

    start = time.perf_counter()

    outputs = session.run(
        None,
        {
            input_name:
            row
        },
    )

    elapsed_ms = (
        time.perf_counter()
        - start
    ) * 1000

    labels = np.asarray(
        outputs[0]
    )

    probabilities = np.asarray(
        outputs[1]
    )[0]

    prediction = int(
        labels[0]
    )

    times.append(
        elapsed_ms
    )

    predictions.append(
        prediction
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
    "engine": "ONNX Runtime",
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
print("RESULTATS ONNX")
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