from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# =========================================================
# CHEMINS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

APPLICATION_TRAIN_PATH = Path(
    r"C:\Users\nibaj\Openclassrooms\Projet 6\data\raw\application_train.csv"
)

VALIDATION_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "validation_features.parquet"
)

VALIDATION_TARGET_PATH = (
    PROJECT_ROOT
    / "data"
    / "validation_target.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "models"
    / "credit_scoring_model_onnx"
    / "scoring_clients.parquet"
)

VALIDATION_SIZE = 10_000


# =========================================================
# VERIFICATION DES FICHIERS
# =========================================================

print("=" * 70)
print("CONSTRUCTION DE LA TABLE CLIENT -> FEATURES")
print("=" * 70)

if not APPLICATION_TRAIN_PATH.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {APPLICATION_TRAIN_PATH}"
    )

if not VALIDATION_FEATURES_PATH.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {VALIDATION_FEATURES_PATH}"
    )

if not VALIDATION_TARGET_PATH.exists():
    raise FileNotFoundError(
        f"Fichier introuvable : {VALIDATION_TARGET_PATH}"
    )


print()
print("Dataset source :", APPLICATION_TRAIN_PATH)
print("Features       :", VALIDATION_FEATURES_PATH)
print("Target         :", VALIDATION_TARGET_PATH)


# =========================================================
# 1. IDENTIFIANTS CLIENTS + TARGET
# =========================================================

print()
print("1. Chargement des identifiants clients...")

application = pd.read_csv(
    APPLICATION_TRAIN_PATH,
    usecols=[
        "SK_ID_CURR",
        "TARGET",
    ],
)

client_ids = application[
    ["SK_ID_CURR"]
].copy()

target = application[
    "TARGET"
].copy()

print(
    "Nombre de clients source :",
    len(application),
)


# =========================================================
# 2. RECONSTRUCTION DU PREMIER SPLIT
# =========================================================

print()
print("2. Reconstruction du split 80 / 20...")

(
    _,
    validation_ids_large,
    _,
    validation_target_large,
) = train_test_split(
    client_ids,
    target,
    test_size=0.20,
    stratify=target,
    random_state=42,
)

print(
    "Validation intermédiaire :",
    len(validation_ids_large),
)


# =========================================================
# 3. RECONSTRUCTION DES 10 000 CLIENTS
# =========================================================

print()
print("3. Reconstruction du jeu de validation final...")

if len(validation_ids_large) > VALIDATION_SIZE:

    (
        validation_ids,
        _,
        validation_target,
        _,
    ) = train_test_split(
        validation_ids_large,
        validation_target_large,
        train_size=VALIDATION_SIZE,
        stratify=validation_target_large,
        random_state=42,
    )

else:

    validation_ids = validation_ids_large.copy()

    validation_target = validation_target_large.copy()


validation_target = validation_target.loc[
    validation_ids.index
]

print(
    "Clients retenus :",
    len(validation_ids),
)


# =========================================================
# 4. CHARGEMENT DES 401 FEATURES
# =========================================================

print()
print("4. Chargement des 401 features...")

features = pd.read_parquet(
    VALIDATION_FEATURES_PATH
)

print(
    "Shape features :",
    features.shape,
)


if features.shape[1] != 401:
    raise ValueError(
        f"401 features attendues, "
        f"{features.shape[1]} trouvées."
    )


if len(features) != len(validation_ids):
    raise ValueError(
        "Le nombre de clients ne correspond pas "
        "au nombre de lignes de features."
    )


# =========================================================
# 5. VERIFICATION DE L'ALIGNEMENT
# =========================================================

print()
print("5. Vérification avec validation_target.parquet...")

stored_target = (
    pd.read_parquet(
        VALIDATION_TARGET_PATH
    )["TARGET"]
    .reset_index(drop=True)
)

reconstructed_target = (
    validation_target
    .reset_index(drop=True)
)

target_identical = stored_target.equals(
    reconstructed_target
)

print(
    "TARGET identique :",
    target_identical,
)


if not target_identical:
    raise ValueError(
        "Le split reconstruit ne correspond pas "
        "au jeu de validation existant."
    )


# =========================================================
# 6. CREATION DE LA TABLE DE SCORING
# =========================================================

print()
print("6. Association client -> 401 features...")

validation_ids = (
    validation_ids
    .reset_index(drop=True)
)

features = (
    features
    .reset_index(drop=True)
)

scoring_clients = pd.concat(
    [
        validation_ids,
        features,
    ],
    axis=1,
)


# =========================================================
# 7. CONTROLES
# =========================================================

if scoring_clients[
    "SK_ID_CURR"
].duplicated().any():

    raise ValueError(
        "Des identifiants clients sont dupliqués."
    )


print(
    "Shape finale :",
    scoring_clients.shape,
)

print(
    "Clients uniques :",
    scoring_clients[
        "SK_ID_CURR"
    ].nunique(),
)

print(
    "Features modèle :",
    scoring_clients.shape[1] - 1,
)


# =========================================================
# 8. SAUVEGARDE
# =========================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

scoring_clients.to_parquet(
    OUTPUT_PATH,
    index=False,
)


print()
print("=" * 70)
print("TABLE DE SCORING CREEE AVEC SUCCES")
print("=" * 70)

print(
    "Fichier :",
    OUTPUT_PATH,
)

print()
print(
    "Exemples de client_id utilisables :"
)

print(
    scoring_clients[
        "SK_ID_CURR"
    ].head(10).to_string(
        index=False
    )
)