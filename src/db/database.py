import os

import psycopg
from psycopg.types.json import Jsonb


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://scoring_user:scoring_password@localhost:5433/scoring_db",
)


def save_prediction(
    features,
    prediction,
    default_probability,
    inference_time_ms,
    request_duration_ms,
    status_code=200,
):
    """
    Enregistre une prédiction dans PostgreSQL.

    Une panne de PostgreSQL ne doit pas empêcher
    l'API de retourner la prédiction au client.
    """

    try:

        with psycopg.connect(
            DATABASE_URL
        ) as conn:

            with conn.cursor() as cursor:

                cursor.execute(
                    """
                    INSERT INTO predictions (
                        prediction,
                        default_probability,
                        inference_time_ms,
                        request_duration_ms,
                        status_code,
                        error_message,
                        features_json
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s
                    )
                    """,
                    (
                        prediction,
                        default_probability,
                        inference_time_ms,
                        request_duration_ms,
                        status_code,
                        None,
                        Jsonb(features),
                    ),
                )

            conn.commit()

    except Exception as exc:

        print(
            "PostgreSQL prediction logging error: "
            f"{type(exc).__name__}: {exc}"
        )


def save_error(
    features,
    status_code,
    error_message,
    request_duration_ms,
):
    """
    Enregistre une requête API en erreur.

    Une panne de PostgreSQL ne doit pas modifier
    le comportement de l'API.
    """

    try:

        with psycopg.connect(
            DATABASE_URL
        ) as conn:

            with conn.cursor() as cursor:

                cursor.execute(
                    """
                    INSERT INTO predictions (
                        prediction,
                        default_probability,
                        inference_time_ms,
                        request_duration_ms,
                        status_code,
                        error_message,
                        features_json
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s
                    )
                    """,
                    (
                        None,
                        None,
                        None,
                        request_duration_ms,
                        status_code,
                        error_message,
                        Jsonb(features),
                    ),
                )

            conn.commit()

    except Exception as exc:

        print(
            "PostgreSQL error logging error: "
            f"{type(exc).__name__}: {exc}"
        )