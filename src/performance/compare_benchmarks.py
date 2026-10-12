import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)


with open(
    REPORTS_DIR
    / "benchmark_sklearn.json",
    encoding="utf-8",
) as file:

    sklearn = json.load(
        file
    )


with open(
    REPORTS_DIR
    / "benchmark_onnx.json",
    encoding="utf-8",
) as file:

    onnx = json.load(
        file
    )


# =========================================================
# PREDICTIONS
# =========================================================

sk_predictions = (
    sklearn[
        "predictions"
    ]
)

onnx_predictions = (
    onnx[
        "predictions"
    ]
)


identical = sum(
    a == b
    for a, b
    in zip(
        sk_predictions,
        onnx_predictions,
    )
)


equivalence = (
    identical
    / len(sk_predictions)
    * 100
)


# =========================================================
# PROBABILITES
# =========================================================

sk_probabilities = (
    sklearn[
        "probabilities_class_1"
    ]
)

onnx_probabilities = (
    onnx[
        "probabilities_class_1"
    ]
)


max_difference = max(
    abs(a - b)
    for a, b
    in zip(
        sk_probabilities,
        onnx_probabilities,
    )
)


# =========================================================
# SPEEDUP
# =========================================================

speedup = (
    sklearn["mean_ms"]
    / onnx["mean_ms"]
)


gain_percent = (
    (
        sklearn["mean_ms"]
        - onnx["mean_ms"]
    )
    / sklearn["mean_ms"]
    * 100
)


# =========================================================
# AFFICHAGE
# =========================================================

print("=" * 70)
print("COMPARAISON MLFLOW / SCIKIT-LEARN VS ONNX")
print("=" * 70)

print()
print("SCIKIT-LEARN")
print(
    f"Moyenne : "
    f"{sklearn['mean_ms']:.3f} ms"
)
print(
    f"P95     : "
    f"{sklearn['p95_ms']:.3f} ms"
)

print()
print("ONNX")
print(
    f"Moyenne : "
    f"{onnx['mean_ms']:.3f} ms"
)
print(
    f"P95     : "
    f"{onnx['p95_ms']:.3f} ms"
)

print()
print("GAIN")
print(
    f"Accélération : "
    f"x{speedup:.2f}"
)

print(
    f"Réduction du temps : "
    f"{gain_percent:.2f} %"
)

print()
print("EQUIVALENCE")
print(
    f"Prédictions identiques : "
    f"{equivalence:.2f} %"
)

print(
    f"Ecart maximal probabilité : "
    f"{max_difference:.8f}"
)

print()
print("=" * 70)