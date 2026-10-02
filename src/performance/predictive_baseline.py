from pathlib import Path

import mlflow.sklearn
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)


# =========================================================
# CHEMINS
# =========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "credit_scoring_model"
)


X_PATH = (
    PROJECT_ROOT
    / "data"
    / "validation_features.parquet"
)


Y_PATH = (
    PROJECT_ROOT
    / "data"
    / "validation_target.parquet"
)


# =========================================================
# CHARGEMENT DU MODELE
# =========================================================

print("=" * 60)
print("BASELINE PREDICTIVE - CREDIT SCORING")
print("=" * 60)


model = mlflow.sklearn.load_model(
    str(MODEL_PATH)
)


# =========================================================
# CHARGEMENT DES DONNEES
# =========================================================

X_valid = pd.read_parquet(
    X_PATH
)


y_valid = pd.read_parquet(
    Y_PATH
)["TARGET"]


print(
    "Nombre de clients :",
    len(X_valid),
)


print(
    "Nombre de features :",
    X_valid.shape[1],
)


print(
    "Nombre de défauts réels :",
    int(
        (y_valid == 1).sum()
    ),
)


# =========================================================
# VERIFICATION DES FEATURES
# =========================================================

expected_features = list(
    model.feature_names_in_
)


features_identical = (
    list(X_valid.columns)
    == expected_features
)


print(
    "Features identiques au modèle :",
    features_identical,
)


if not features_identical:

    raise ValueError(
        "Les colonnes de validation ne correspondent "
        "pas aux 401 features du modèle."
    )


# =========================================================
# PREDICTIONS
# =========================================================

print()
print(
    "Calcul des prédictions..."
)


predictions = model.predict(
    X_valid
)


probabilities = (
    model.predict_proba(
        X_valid
    )
)


# Recherche de la colonne correspondant
# à la classe 1 = défaut
classes = list(
    model.named_steps[
        "classifier"
    ].classes_
)


default_class_index = (
    classes.index(1)
)


default_probabilities = (
    probabilities[
        :,
        default_class_index,
    ]
)


# =========================================================
# CALCUL DES METRIQUES
# =========================================================

roc_auc = roc_auc_score(
    y_valid,
    default_probabilities,
)


recall = recall_score(
    y_valid,
    predictions,
    zero_division=0,
)


f1 = f1_score(
    y_valid,
    predictions,
    zero_division=0,
)


accuracy = accuracy_score(
    y_valid,
    predictions,
)


tn, fp, fn, tp = (
    confusion_matrix(
        y_valid,
        predictions,
    )
    .ravel()
)


# =========================================================
# AFFICHAGE DES RESULTATS
# =========================================================

print()
print("=" * 60)
print("PERFORMANCES PREDICTIVES")
print("=" * 60)


print(
    f"ROC AUC  : {roc_auc:.4f}"
)


print(
    f"Recall   : {recall:.4f}"
)


print(
    f"F1 Score : {f1:.4f}"
)


print(
    f"Accuracy : {accuracy:.4f}"
)


# =========================================================
# MATRICE DE CONFUSION
# =========================================================

print()
print("=" * 60)
print("MATRICE DE CONFUSION")
print("=" * 60)


print(
    f"Vrais négatifs  (TN) : {tn}"
)


print(
    f"Faux positifs   (FP) : {fp}"
)


print(
    f"Faux négatifs   (FN) : {fn}"
)


print(
    f"Vrais positifs  (TP) : {tp}"
)


# =========================================================
# DISTRIBUTION DES CLASSES
# =========================================================

print()
print("=" * 60)
print("DISTRIBUTION DES VRAIS LABELS")
print("=" * 60)


print(
    y_valid
    .value_counts()
    .sort_index()
)


print()
print("=" * 60)
print("DISTRIBUTION DES PREDICTIONS")
print("=" * 60)


print(
    pd.Series(
        predictions,
        name="prediction",
    )
    .value_counts()
    .sort_index()
)


print()
print("=" * 60)
print("BASELINE PREDICTIVE TERMINEE")
print("=" * 60)