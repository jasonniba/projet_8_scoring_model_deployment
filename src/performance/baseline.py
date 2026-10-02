import os

import numpy as np
import pandas as pd
import psycopg


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://scoring_user:scoring_password@localhost:5433/scoring_db",
)


def load_baseline_data():

    query = """
        SELECT
            created_at,
            prediction,
            default_probability,
            inference_time_ms,
            request_duration_ms,
            status_code
        FROM predictions
        WHERE status_code = 200
          AND inference_time_ms IS NOT NULL
          AND request_duration_ms IS NOT NULL
        ORDER BY created_at DESC
        LIMIT 500
    """

    with psycopg.connect(DATABASE_URL) as conn:

        df = pd.read_sql_query(
            query,
            conn,
        )

    return df


def print_metric_summary(
    name,
    values,
):

    values = (
        pd.to_numeric(
            values,
            errors="coerce",
        )
        .dropna()
    )

    print()
    print(name)
    print("-" * 50)

    print(
        f"Moyenne : {values.mean():.3f} ms"
    )

    print(
        f"P50     : {np.percentile(values, 50):.3f} ms"
    )

    print(
        f"P95     : {np.percentile(values, 95):.3f} ms"
    )

    print(
        f"P99     : {np.percentile(values, 99):.3f} ms"
    )

    print(
        f"Min     : {values.min():.3f} ms"
    )

    print(
        f"Max     : {values.max():.3f} ms"
    )


if __name__ == "__main__":

    df = load_baseline_data()

    print("=" * 60)
    print("BASELINE OPERATIONNELLE - API DE SCORING")
    print("=" * 60)

    print(
        f"Nombre de requêtes analysées : {len(df)}"
    )

    print_metric_summary(
        "TEMPS D'INFERENCE",
        df["inference_time_ms"],
    )

    print_metric_summary(
        "DUREE TOTALE DE LA REQUETE",
        df["request_duration_ms"],
    )

    if len(df) > 1:

        first_time = df[
            "created_at"
        ].min()

        last_time = df[
            "created_at"
        ].max()

        duration_seconds = (
            last_time
            - first_time
        ).total_seconds()

        if duration_seconds > 0:

            throughput = (
                len(df)
                / duration_seconds
            )

            print()
            print("DEBIT APPROXIMATIF")
            print("-" * 50)

            print(
                f"Durée observée : "
                f"{duration_seconds:.2f} sec"
            )

            print(
                f"Débit moyen : "
                f"{throughput:.2f} req/s"
            )

    print()
    print("DISTRIBUTION DES PREDICTIONS")
    print("-" * 50)

    print(
        df["prediction"]
        .value_counts()
        .sort_index()
    )

    print()
    print("=" * 60)