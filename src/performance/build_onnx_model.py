from pathlib import Path

import mlflow.sklearn

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


print("DEBUT CONVERSION ONNX", flush=True)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "credit_scoring_model"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "credit_scoring_model_onnx"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "model.onnx"
)


print(
    "Chargement du modèle MLflow...",
    flush=True,
)

model = mlflow.sklearn.load_model(
    str(MODEL_PATH)
)


print(
    "Modèle chargé.",
    flush=True,
)

print(
    "Nombre de features :",
    len(model.feature_names_in_),
    flush=True,
)


classifier = model.named_steps[
    "classifier"
]


print(
    "Conversion vers ONNX...",
    flush=True,
)


initial_type = [
    (
        "input",
        FloatTensorType(
            [
                None,
                len(model.feature_names_in_),
            ]
        ),
    )
]


onnx_model = convert_sklearn(
    model,
    initial_types=initial_type,
    options={
        id(classifier): {
            "zipmap": False
        }
    },
    target_opset=18,
)


print(
    "Conversion terminée.",
    flush=True,
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


with open(
    OUTPUT_PATH,
    "wb",
) as file:

    file.write(
        onnx_model.SerializeToString()
    )


print(
    "Modèle sauvegardé :",
    OUTPUT_PATH,
    flush=True,
)

print(
    "FIN CONVERSION ONNX",
    flush=True,
)