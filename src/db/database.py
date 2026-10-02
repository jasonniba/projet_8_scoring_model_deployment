import os

from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool


# =========================================================
# CONFIGURATION POSTGRESQL
# =========================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    (
        "postgresql://"
        "scoring_user:"
        "scoring_password"
        "@localhost:5433/"
        "scoring_db"
    ),
)


# =========================================================
# POOL DE CONNEXIONS
# =========================================================

pool = ConnectionPool(
    conninfo=DATABASE_URL,
    min_size=2,
    max_size=10,
    kwargs={
        "autocommit": True,
    },
    open=True,
)


# =========================================================
# SAUVEGARDER UNE PREDICTION
# =========================================================

def save_prediction(
    features,
    prediction,
    default_probability,
    inference_time_ms,
    request_duration_ms,
    status_code=200,
):

    try:

        with pool.connection() as connection:

            with connection.cursor() as cursor:

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
                        int(
                            prediction
                        ),

                        float(
                            default_probability
                        ),

                        float(
                            inference_time_ms
                        ),

                        float(
                            request_duration_ms
                        ),

                        int(
                            status_code
                        ),

                        None,

                        Jsonb(
                            features
                        ),
                    ),
                )

    except Exception as exc:

        # Le monitoring ne doit pas
        # empêcher l'API de répondre.
        print(
            "PostgreSQL prediction logging error:",
            exc,
        )


# =========================================================
# SAUVEGARDER UNE ERREUR API
# =========================================================

def save_error(
    features,
    status_code,
    error_message,
    request_duration_ms,
):

    try:

        with pool.connection() as connection:

            with connection.cursor() as cursor:

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

                        float(
                            request_duration_ms
                        ),

                        int(
                            status_code
                        ),

                        str(
                            error_message
                        ),

                        Jsonb(
                            features
                        ),
                    ),
                )

    except Exception as exc:

        print(
            "PostgreSQL error logging error:",
            exc,
        )