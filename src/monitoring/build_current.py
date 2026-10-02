import os
from pathlib import Path

import pandas as pd
import psycopg


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "reference_features.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "current_features.parquet"
)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://scoring_user:scoring_password@localhost:5433/scoring_db",
)


def load_current_features():

    query = """
        SELECT features_json
        FROM predictions
        WHERE status_code = 200
          AND features_json IS NOT NULL
        ORDER BY created_at DESC
        LIMIT 500
    """

    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute(query).fetchall()

    if not rows:
        raise RuntimeError(
            "Aucune prédiction valide trouvée dans PostgreSQL."
        )

    features = [
        row[0]
        for row in rows
    ]

    current_df = pd.DataFrame(features)

    return current_df


if __name__ == "__main__":

    print("Lecture des données de production PostgreSQL...")

    current_df = load_current_features()

    # Charger les colonnes exactes de référence
    reference_df = pd.read_parquet(
        REFERENCE_PATH
    )

    expected_columns = list(
        reference_df.columns
    )

    # Alignement strict avec la référence
    current_df = current_df.reindex(
        columns=expected_columns
    )

    current_df = current_df.astype(
        "float32"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    current_df.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 60)

    print(
        "Nombre de prédictions utilisées :",
        len(current_df),
    )

    print(
        "Nombre de features :",
        current_df.shape[1],
    )

    print(
        "Shape current :",
        current_df.shape,
    )

    print(
        "Colonnes identiques à la référence :",
        list(current_df.columns)
        == expected_columns,
    )

    print(
        "Fichier créé :",
        OUTPUT_PATH,
    )

    print("=" * 60)