from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# =========================================================
# 1. CHEMINS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OLD_DATA_DIR = (
    PROJECT_ROOT.parent
    / "Projet 6"
    / "data"
    / "raw"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "credit_scoring_model"
)

REFERENCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "reference_features.parquet"
)

SIMULATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "simulation_features.parquet"
)


# =========================================================
# 2. CHARGEMENT DES DONNÉES DU PROJET 6
# =========================================================

print("Chargement des données...")

app_train = pd.read_csv(
    OLD_DATA_DIR / "application_train.csv"
)

bureau = pd.read_csv(
    OLD_DATA_DIR / "bureau.csv"
)

bb = pd.read_csv(
    OLD_DATA_DIR / "bureau_balance.csv"
)

previous_app = pd.read_csv(
    OLD_DATA_DIR / "previous_application.csv"
)

installments = pd.read_csv(
    OLD_DATA_DIR / "installments_payments.csv"
)

cc = pd.read_csv(
    OLD_DATA_DIR / "credit_card_balance.csv"
)

pos_cash = pd.read_csv(
    OLD_DATA_DIR / "POS_CASH_balance.csv"
)


# =========================================================
# 3. DATASET PRINCIPAL
# =========================================================

train = app_train.copy()


# =========================================================
# 4. FEATURE ENGINEERING PRINCIPAL
# =========================================================

train["DAYS_EMPLOYED"] = train[
    "DAYS_EMPLOYED"
].replace(
    365243,
    np.nan,
)

train["AGE_YEARS"] = (
    -train["DAYS_BIRTH"] / 365
)

train["CREDIT_INCOME_RATIO"] = (
    train["AMT_CREDIT"]
    / train["AMT_INCOME_TOTAL"]
)

train["ANNUITY_INCOME_RATIO"] = (
    train["AMT_ANNUITY"]
    / train["AMT_INCOME_TOTAL"]
)

train["CREDIT_TERM"] = (
    train["AMT_ANNUITY"]
    / train["AMT_CREDIT"]
)

train["EXT_SOURCE_MEAN"] = train[
    [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
    ]
].mean(axis=1)


# =========================================================
# 5. BUREAU BALANCE
# =========================================================

print("Agrégation bureau_balance...")

bb_encode = pd.get_dummies(bb)

bb_agg = bb_encode.groupby(
    "SK_ID_BUREAU"
).agg(
    {
        "MONTHS_BALANCE": [
            "min",
            "max",
            "size",
        ],
        "STATUS_0": "mean",
        "STATUS_1": "mean",
        "STATUS_2": "mean",
        "STATUS_3": "mean",
        "STATUS_4": "mean",
        "STATUS_5": "mean",
        "STATUS_C": "mean",
        "STATUS_X": "mean",
    }
)

bb_agg.columns = [
    "BB_" + "_".join(col)
    for col in bb_agg.columns
]

bb_agg = bb_agg.reset_index()

bb_agg = bb_agg.merge(
    bureau[
        [
            "SK_ID_BUREAU",
            "SK_ID_CURR",
        ]
    ],
    on="SK_ID_BUREAU",
    how="left",
)

bb_agg = (
    bb_agg
    .drop(columns="SK_ID_BUREAU")
    .groupby("SK_ID_CURR")
    .mean()
    .reset_index()
)


# =========================================================
# 6. BUREAU
# =========================================================

print("Agrégation bureau...")

bureau_encode = pd.get_dummies(
    bureau
)

bureau_agg = bureau_encode.groupby(
    "SK_ID_CURR"
).agg(
    {
        "SK_ID_BUREAU": "count",

        "DAYS_CREDIT": [
            "mean",
            "min",
            "max",
        ],

        "CREDIT_DAY_OVERDUE": [
            "mean",
            "max",
        ],

        "AMT_CREDIT_SUM": [
            "mean",
            "sum",
            "max",
        ],

        "AMT_CREDIT_SUM_DEBT": [
            "mean",
            "sum",
        ],

        "AMT_CREDIT_SUM_OVERDUE": [
            "mean",
            "sum",
        ],

        "AMT_ANNUITY": [
            "mean",
            "sum",
        ],
    }
)

bureau_agg.columns = [
    "BUREAU_" + "_".join(col)
    for col in bureau_agg.columns
]

bureau_agg = bureau_agg.reset_index()


bureau_cat_cols = [
    col
    for col in bureau_encode.columns
    if col.startswith("CREDIT_ACTIVE_")
    or col.startswith("CREDIT_CURRENCY_")
    or col.startswith("CREDIT_TYPE_")
]


bureau_cat_agg = (
    bureau_encode
    .groupby("SK_ID_CURR")[
        bureau_cat_cols
    ]
    .mean()
    .reset_index()
)


bureau_agg = bureau_agg.merge(
    bureau_cat_agg,
    on="SK_ID_CURR",
    how="left",
)


# =========================================================
# 7. PREVIOUS APPLICATION
# =========================================================

print("Agrégation previous_application...")

previous_app[
    "APP_CREDIT_RATIO"
] = (
    previous_app["AMT_APPLICATION"]
    / previous_app["AMT_CREDIT"]
)

previous_app[
    "APPROVED"
] = (
    previous_app[
        "NAME_CONTRACT_STATUS"
    ]
    == "Approved"
).astype(int)

previous_app[
    "REFUSED"
] = (
    previous_app[
        "NAME_CONTRACT_STATUS"
    ]
    == "Refused"
).astype(int)


previous_agg = previous_app.groupby(
    "SK_ID_CURR"
).agg(
    {
        "SK_ID_PREV": "count",

        "AMT_ANNUITY": [
            "mean",
            "max",
        ],

        "AMT_APPLICATION": [
            "mean",
            "max",
            "sum",
        ],

        "AMT_CREDIT": [
            "mean",
            "max",
            "sum",
        ],

        "AMT_DOWN_PAYMENT": [
            "mean",
            "max",
        ],

        "AMT_GOODS_PRICE": [
            "mean",
            "max",
        ],

        "DAYS_DECISION": [
            "mean",
            "min",
            "max",
        ],

        "CNT_PAYMENT": [
            "mean",
            "sum",
        ],

        "APP_CREDIT_RATIO": [
            "mean",
            "max",
        ],

        "APPROVED": [
            "mean",
            "sum",
        ],

        "REFUSED": [
            "mean",
            "sum",
        ],
    }
)

previous_agg.columns = [
    "PREV_" + "_".join(col)
    for col in previous_agg.columns
]

previous_agg = previous_agg.reset_index()


# =========================================================
# 8. INSTALLMENTS
# =========================================================

print("Agrégation installments...")

installments[
    "PAYMENT_DELAY"
] = (
    installments[
        "DAYS_ENTRY_PAYMENT"
    ]
    - installments[
        "DAYS_INSTALMENT"
    ]
)

installments[
    "PAYMENT_RATIO"
] = (
    installments[
        "AMT_PAYMENT"
    ]
    / installments[
        "AMT_INSTALMENT"
    ]
)


installments_agg = installments.groupby(
    "SK_ID_CURR"
).agg(
    {
        "SK_ID_PREV": "nunique",

        "NUM_INSTALMENT_NUMBER": [
            "mean",
            "max",
        ],

        "DAYS_INSTALMENT": [
            "mean",
            "min",
            "max",
        ],

        "DAYS_ENTRY_PAYMENT": [
            "mean",
            "min",
            "max",
        ],

        "AMT_INSTALMENT": [
            "mean",
            "sum",
            "max",
        ],

        "AMT_PAYMENT": [
            "mean",
            "sum",
            "max",
        ],

        "PAYMENT_DELAY": [
            "mean",
            "max",
        ],

        "PAYMENT_RATIO": [
            "mean",
            "min",
        ],
    }
)

installments_agg.columns = [
    "INS_" + "_".join(col)
    for col in installments_agg.columns
]

installments_agg = (
    installments_agg
    .reset_index()
)


# =========================================================
# 9. CREDIT CARD BALANCE
# =========================================================

print("Agrégation credit_card_balance...")

cc[
    "LIMIT_USE"
] = (
    cc["AMT_BALANCE"]
    / cc[
        "AMT_CREDIT_LIMIT_ACTUAL"
    ]
)

cc[
    "LATE_PAYMENT"
] = (
    cc["SK_DPD"] > 0
).astype(int)


cc_agg = cc.groupby(
    "SK_ID_CURR"
).agg(
    {
        "SK_ID_PREV": "nunique",

        "MONTHS_BALANCE": [
            "min",
            "max",
            "size",
        ],

        "AMT_BALANCE": [
            "mean",
            "max",
            "sum",
        ],

        "AMT_CREDIT_LIMIT_ACTUAL": [
            "mean",
            "max",
        ],

        "AMT_DRAWINGS_CURRENT": [
            "mean",
            "max",
            "sum",
        ],

        "AMT_PAYMENT_CURRENT": [
            "mean",
            "max",
            "sum",
        ],

        "SK_DPD": [
            "mean",
            "max",
            "sum",
        ],

        "SK_DPD_DEF": [
            "mean",
            "max",
            "sum",
        ],

        "LIMIT_USE": [
            "mean",
            "max",
        ],

        "LATE_PAYMENT": [
            "mean",
            "sum",
        ],
    }
)

cc_agg.columns = [
    "CC_" + "_".join(col)
    for col in cc_agg.columns
]

cc_agg = cc_agg.reset_index()


# =========================================================
# 10. POS CASH
# =========================================================

print("Agrégation POS_CASH_balance...")

pos_cash[
    "POS_LATE_PAYMENT"
] = (
    pos_cash["SK_DPD"] > 0
).astype(int)

pos_cash[
    "REMAINING_RATIO"
] = (
    pos_cash[
        "CNT_INSTALMENT_FUTURE"
    ]
    / pos_cash[
        "CNT_INSTALMENT"
    ]
)


pos_agg = pos_cash.groupby(
    "SK_ID_CURR"
).agg(
    {
        "SK_ID_PREV": "nunique",

        "MONTHS_BALANCE": [
            "min",
            "max",
            "size",
        ],

        "CNT_INSTALMENT": [
            "mean",
            "max",
        ],

        "CNT_INSTALMENT_FUTURE": [
            "mean",
            "min",
            "max",
        ],

        "SK_DPD": [
            "mean",
            "max",
            "sum",
        ],

        "SK_DPD_DEF": [
            "mean",
            "max",
            "sum",
        ],

        "POS_LATE_PAYMENT": [
            "mean",
            "sum",
        ],

        "REMAINING_RATIO": [
            "mean",
            "min",
        ],
    }
)

pos_agg.columns = [
    "POS_" + "_".join(col)
    for col in pos_agg.columns
]

pos_agg = pos_agg.reset_index()


# =========================================================
# 11. FUSION DES TABLES
# =========================================================

print("Fusion des tables...")

tables_to_merge = [
    bb_agg,
    bureau_agg,
    previous_agg,
    installments_agg,
    cc_agg,
    pos_agg,
]


for table in tables_to_merge:

    train = train.merge(
        table,
        on="SK_ID_CURR",
        how="left",
    )


# =========================================================
# 12. CHARGEMENT DU MODÈLE
# =========================================================

print("Chargement du modèle...")

model = mlflow.sklearn.load_model(
    str(MODEL_PATH)
)

expected_features = list(
    model.feature_names_in_
)

print(
    "Features attendues par le modèle :",
    len(expected_features),
)


# =========================================================
# 13. PRÉPARATION DES FEATURES
# =========================================================

y = train[
    "TARGET"
]

X = train.drop(
    columns=[
        "TARGET",
        "SK_ID_CURR",
    ],
    errors="ignore",
)


# One-hot encoding
X = pd.get_dummies(
    X,
    dummy_na=True,
)


# Nettoyage des valeurs infinies
X = X.replace(
    [
        np.inf,
        -np.inf,
    ],
    np.nan,
)


# Alignement strict avec le modèle
X = X.reindex(
    columns=expected_features,
    fill_value=0,
)


# Même type que dans le notebook
X = X.astype(
    np.float32
)


print(
    "Shape après alignement :",
    X.shape,
)


# Vérifications
assert (
    X.shape[1]
    == len(expected_features)
)

assert (
    list(X.columns)
    == expected_features
)


# =========================================================
# 14. SPLIT TRAIN / VALIDATION
# =========================================================

X_train_small, X_valid_small, y_train_small, y_valid_small = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
)


print(
    "X_train_small :",
    X_train_small.shape,
)

print(
    "X_valid_small :",
    X_valid_small.shape,
)


# =========================================================
# 15. DATASET DE RÉFÉRENCE
# =========================================================

reference_sample = (
    X_train_small
    .sample(
        n=min(
            5000,
            len(X_train_small),
        ),
        random_state=42,
    )
)


# =========================================================
# 16. DATASET DE SIMULATION PRODUCTION
# =========================================================

simulation_sample = (
    X_valid_small
    .sample(
        n=min(
            500,
            len(X_valid_small),
        ),
        random_state=42,
    )
)


# =========================================================
# 17. SAUVEGARDE DES FICHIERS
# =========================================================

REFERENCE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


reference_sample.to_parquet(
    REFERENCE_PATH,
    index=False,
)


simulation_sample.to_parquet(
    SIMULATION_PATH,
    index=False,
)


# =========================================================
# 18. VÉRIFICATIONS FINALES
# =========================================================

print()
print("=" * 60)

print(
    "Référence créée avec succès."
)

print(
    "Chemin référence :",
    REFERENCE_PATH,
)

print(
    "Shape référence :",
    reference_sample.shape,
)

print(
    "Colonnes référence identiques au modèle :",
    list(reference_sample.columns)
    == expected_features,
)

print(
    "Fichier référence existe :",
    REFERENCE_PATH.exists(),
)


print()

print(
    "Batch de simulation créé avec succès."
)

print(
    "Chemin simulation :",
    SIMULATION_PATH,
)

print(
    "Shape simulation :",
    simulation_sample.shape,
)

print(
    "Colonnes simulation identiques au modèle :",
    list(simulation_sample.columns)
    == expected_features,
)

print(
    "Fichier simulation existe :",
    SIMULATION_PATH.exists(),
)

print("=" * 60)