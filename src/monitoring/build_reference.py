from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# =========================================================
# CHEMINS DU PROJET
# =========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


RAW_DATA_DIR = (
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


DATA_DIR = (
    PROJECT_ROOT
    / "data"
)


DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# CHARGEMENT DU MODELE
# =========================================================

print("=" * 70)
print("CHARGEMENT DU MODELE")
print("=" * 70)


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
# CHARGEMENT DES DONNEES HOME CREDIT
# =========================================================

print()
print("=" * 70)
print("CHARGEMENT DES DONNEES")
print("=" * 70)


app_train = pd.read_csv(
    RAW_DATA_DIR
    / "application_train.csv"
)


app_test = pd.read_csv(
    RAW_DATA_DIR
    / "application_test.csv"
)


bureau = pd.read_csv(
    RAW_DATA_DIR
    / "bureau.csv"
)


bureau_balance = pd.read_csv(
    RAW_DATA_DIR
    / "bureau_balance.csv"
)


previous_app = pd.read_csv(
    RAW_DATA_DIR
    / "previous_application.csv"
)


installments = pd.read_csv(
    RAW_DATA_DIR
    / "installments_payments.csv"
)


cc = pd.read_csv(
    RAW_DATA_DIR
    / "credit_card_balance.csv"
)


pos_cash = pd.read_csv(
    RAW_DATA_DIR
    / "POS_CASH_balance.csv"
)


print(
    "application_train :",
    app_train.shape,
)

print(
    "application_test :",
    app_test.shape,
)


# =========================================================
# COPIES TRAIN / TEST
# =========================================================

train = app_train.copy()
test = app_test.copy()


# =========================================================
# FEATURE ENGINEERING APPLICATION TRAIN / TEST
# =========================================================

print()
print("=" * 70)
print("FEATURE ENGINEERING APPLICATION")
print("=" * 70)


for df in [
    train,
    test,
]:

    # Valeur spéciale Home Credit
    df["DAYS_EMPLOYED"] = (
        df["DAYS_EMPLOYED"]
        .replace(
            365243,
            np.nan,
        )
    )

    # Age en années
    df["AGE_YEARS"] = (
        -df["DAYS_BIRTH"]
        / 365
    )

    # Rapport crédit / revenus
    df["CREDIT_INCOME_RATIO"] = (
        df["AMT_CREDIT"]
        / df["AMT_INCOME_TOTAL"]
    )

    # Rapport annuité / revenus
    df["ANNUITY_INCOME_RATIO"] = (
        df["AMT_ANNUITY"]
        / df["AMT_INCOME_TOTAL"]
    )

    # Durée / poids du crédit
    df["CREDIT_TERM"] = (
        df["AMT_ANNUITY"]
        / df["AMT_CREDIT"]
    )

    # Moyenne des scores externes
    df["EXT_SOURCE_MEAN"] = (
        df[
            [
                "EXT_SOURCE_1",
                "EXT_SOURCE_2",
                "EXT_SOURCE_3",
            ]
        ]
        .mean(
            axis=1
        )
    )


# =========================================================
# BUREAU BALANCE
# =========================================================

print(
    "Agrégation bureau_balance..."
)


bb_encode = pd.get_dummies(
    bureau_balance,
    columns=[
        "STATUS",
    ],
)


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


bb_agg = (
    bb_agg
    .reset_index()
)


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
    .drop(
        columns="SK_ID_BUREAU"
    )
    .groupby(
        "SK_ID_CURR"
    )
    .mean()
    .reset_index()
)


# =========================================================
# BUREAU
# =========================================================

print(
    "Agrégation bureau..."
)


bureau_encode = pd.get_dummies(
    bureau,
    columns=[
        "CREDIT_ACTIVE",
        "CREDIT_CURRENCY",
        "CREDIT_TYPE",
    ],
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


bureau_agg = (
    bureau_agg
    .reset_index()
)


# Colonnes catégorielles du bureau

bureau_cat_cols = [
    col
    for col in bureau_encode.columns
    if (
        col.startswith(
            "CREDIT_ACTIVE_"
        )
        or col.startswith(
            "CREDIT_CURRENCY_"
        )
        or col.startswith(
            "CREDIT_TYPE_"
        )
    )
]


bureau_cat_agg = (
    bureau_encode
    .groupby(
        "SK_ID_CURR"
    )[bureau_cat_cols]
    .mean()
    .reset_index()
)


bureau_agg = bureau_agg.merge(
    bureau_cat_agg,
    on="SK_ID_CURR",
    how="left",
)


# =========================================================
# PREVIOUS APPLICATION
# =========================================================

print(
    "Agrégation previous_application..."
)


previous_app[
    "APP_CREDIT_RATIO"
] = (
    previous_app[
        "AMT_APPLICATION"
    ]
    / previous_app[
        "AMT_CREDIT"
    ]
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


previous_agg = (
    previous_agg
    .reset_index()
)


# =========================================================
# INSTALLMENTS PAYMENTS
# =========================================================

print(
    "Agrégation installments_payments..."
)


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
# CREDIT CARD BALANCE
# =========================================================

print(
    "Agrégation credit_card_balance..."
)


cc[
    "LIMIT_USE"
] = (
    cc[
        "AMT_BALANCE"
    ]
    / cc[
        "AMT_CREDIT_LIMIT_ACTUAL"
    ]
)


cc[
    "LATE_PAYMENT"
] = (
    cc["SK_DPD"]
    > 0
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


cc_agg = (
    cc_agg
    .reset_index()
)


# =========================================================
# POS CASH BALANCE
# =========================================================

print(
    "Agrégation POS_CASH_balance..."
)


pos_cash[
    "POS_LATE_PAYMENT"
] = (
    pos_cash[
        "SK_DPD"
    ]
    > 0
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


pos_agg = (
    pos_agg
    .reset_index()
)


# =========================================================
# MERGE DE TOUTES LES TABLES
# =========================================================

print()
print("=" * 70)
print("FUSION DES TABLES")
print("=" * 70)


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

    test = test.merge(
        table,
        on="SK_ID_CURR",
        how="left",
    )


print(
    "Shape train après merge :",
    train.shape,
)

print(
    "Shape test après merge :",
    test.shape,
)


# =========================================================
# ONE-HOT ENCODING
# =========================================================

print()
print("=" * 70)
print("ENCODAGE DES VARIABLES")
print("=" * 70)


target = train[
    "TARGET"
].copy()


train_no_target = train.drop(
    columns=[
        "TARGET",
    ]
)


# On combine train et test uniquement pour obtenir
# exactement les mêmes colonnes catégorielles.

combined = pd.concat(
    [
        train_no_target,
        test,
    ],
    axis=0,
    ignore_index=True,
)


combined = pd.get_dummies(
    combined,
    dummy_na=True,
)


train_final = (
    combined
    .iloc[
        :len(train_no_target),
        :
    ]
    .copy()
)


train_final.index = (
    train_no_target.index
)


train_final[
    "TARGET"
] = target.values


# Libération du gros DataFrame combiné
del combined


# =========================================================
# PREPARATION DES 401 FEATURES DU MODELE
# =========================================================

print()
print("=" * 70)
print("ALIGNEMENT SUR LE MODELE")
print("=" * 70)


X_monitor = train_final.drop(
    columns=[
        "TARGET",
        "SK_ID_CURR",
    ],
    errors="ignore",
)


y_monitor = train_final[
    "TARGET"
].copy()


# Encoder au cas où une catégorie subsiste
X_monitor = pd.get_dummies(
    X_monitor,
    dummy_na=True,
)


# Nettoyage des divisions par zéro
X_monitor = X_monitor.replace(
    [
        np.inf,
        -np.inf,
    ],
    np.nan,
)


# Aligner exactement sur les features
# attendues par le modèle de production
X_monitor = X_monitor.reindex(
    columns=expected_features,
    fill_value=0,
)


# Réduction mémoire
X_monitor = X_monitor.astype(
    np.float32
)


print(
    "Shape après alignement :",
    X_monitor.shape,
)


print(
    "Colonnes identiques au modèle :",
    list(
        X_monitor.columns
    )
    == expected_features,
)


if X_monitor.shape[1] != len(
    expected_features
):

    raise ValueError(
        "Le nombre de features ne correspond pas au modèle."
    )


# =========================================================
# SPLIT TRAIN / VALIDATION
# =========================================================

print()
print("=" * 70)
print("SPLIT TRAIN / VALIDATION")
print("=" * 70)


(
    X_train_small,
    X_valid_small,
    y_train_small,
    y_valid_small,
) = train_test_split(
    X_monitor,
    y_monitor,
    test_size=0.20,
    stratify=y_monitor,
    random_state=42,
)


print(
    "X_train_small :",
    X_train_small.shape,
)


print(
    "X_valid_small :",
    X_valid_small.shape,
)


print(
    "y_train_small :",
    y_train_small.shape,
)


print(
    "y_valid_small :",
    y_valid_small.shape,
)


# =========================================================
# DATASET DE REFERENCE POUR EVIDENTLY
# =========================================================

print()
print("=" * 70)
print("CREATION DU DATASET DE REFERENCE")
print("=" * 70)


REFERENCE_SIZE = min(
    5000,
    len(
        X_train_small
    ),
)


reference_features = (
    X_train_small
    .sample(
        n=REFERENCE_SIZE,
        random_state=42,
    )
    .copy()
)


REFERENCE_PATH = (
    DATA_DIR
    / "reference_features.parquet"
)


reference_features.to_parquet(
    REFERENCE_PATH,
    index=False,
)


print(
    "Fichier référence :",
    REFERENCE_PATH,
)


print(
    "Shape référence :",
    reference_features.shape,
)


print(
    "Colonnes référence identiques au modèle :",
    list(
        reference_features.columns
    )
    == expected_features,
)


# =========================================================
# DATASET DE SIMULATION PRODUCTION
# =========================================================

print()
print("=" * 70)
print("CREATION DU BATCH DE PRODUCTION SIMULEE")
print("=" * 70)


SIMULATION_SIZE = min(
    500,
    len(
        X_valid_small
    ),
)


simulation_features = (
    X_valid_small
    .sample(
        n=SIMULATION_SIZE,
        random_state=42,
    )
    .copy()
)


SIMULATION_PATH = (
    DATA_DIR
    / "simulation_features.parquet"
)


simulation_features.to_parquet(
    SIMULATION_PATH,
    index=False,
)


print(
    "Fichier simulation :",
    SIMULATION_PATH,
)


print(
    "Shape simulation :",
    simulation_features.shape,
)


print(
    "Colonnes simulation identiques au modèle :",
    list(
        simulation_features.columns
    )
    == expected_features,
)


# =========================================================
# VALIDATION LABELLisee POUR LA BASELINE PREDICTIVE
# =========================================================

print()
print("=" * 70)
print("CREATION DU DATASET DE BASELINE PREDICTIVE")
print("=" * 70)


VALIDATION_SIZE = min(
    10000,
    len(
        X_valid_small
    ),
)


# On extrait 10 000 clients de la validation
# en conservant la proportion des classes 0 / 1.

if VALIDATION_SIZE < len(
    X_valid_small
):

    (
        validation_features,
        _,
        validation_target,
        _,
    ) = train_test_split(
        X_valid_small,
        y_valid_small,
        train_size=VALIDATION_SIZE,
        stratify=y_valid_small,
        random_state=42,
    )

else:

    validation_features = (
        X_valid_small.copy()
    )

    validation_target = (
        y_valid_small.copy()
    )


# Vérification de l'ordre des lignes
validation_target = (
    validation_target
    .loc[
        validation_features.index
    ]
)


VALIDATION_FEATURES_PATH = (
    DATA_DIR
    / "validation_features.parquet"
)


VALIDATION_TARGET_PATH = (
    DATA_DIR
    / "validation_target.parquet"
)


validation_features.to_parquet(
    VALIDATION_FEATURES_PATH,
    index=False,
)


validation_target.reset_index(
    drop=True
).to_frame(
    name="TARGET"
).to_parquet(
    VALIDATION_TARGET_PATH,
    index=False,
)


print(
    "Fichier validation features :",
    VALIDATION_FEATURES_PATH,
)


print(
    "Fichier validation target :",
    VALIDATION_TARGET_PATH,
)


print(
    "Shape validation baseline :",
    validation_features.shape,
)


print(
    "Shape target baseline :",
    validation_target.shape,
)


print(
    "Colonnes validation identiques au modèle :",
    list(
        validation_features.columns
    )
    == expected_features,
)


print(
    "Répartition TARGET :"
)


print(
    validation_target
    .value_counts()
    .sort_index()
)


# =========================================================
# VERIFICATIONS FINALES
# =========================================================

print()
print("=" * 70)
print("VERIFICATIONS FINALES")
print("=" * 70)


print(
    "Fichier référence existe :",
    REFERENCE_PATH.exists(),
)


print(
    "Fichier simulation existe :",
    SIMULATION_PATH.exists(),
)


print(
    "Fichier validation features existe :",
    VALIDATION_FEATURES_PATH.exists(),
)


print(
    "Fichier validation target existe :",
    VALIDATION_TARGET_PATH.exists(),
)


print()
print("=" * 70)
print("BUILD REFERENCE TERMINE")
print("=" * 70)