import json
import os
from pathlib import Path

import psycopg


PROJECT_ROOT = Path(__file__).resolve().parents[2]

JSON_PATH = (
    PROJECT_ROOT
    / "reports"
    / "data_drift_result.json"
)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://scoring_user:scoring_password@localhost:5433/scoring_db",
)


REFERENCE_PERIOD = "training_reference_5000"

CURRENT_PERIOD = "production_simulation_500"


def load_drift_metrics():
    with open(
        JSON_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    drift_metrics = []

    for metric in data["metrics"]:

        config = metric.get(
            "config",
            {},
        )

        metric_type = config.get(
            "type"
        )

        # On garde uniquement les métriques
        # de drift par variable
        if metric_type != "evidently:metric_v2:ValueDrift":
            continue

        feature_name = config.get(
            "column"
        )

        test_name = config.get(
            "method"
        )

        threshold = config.get(
            "threshold"
        )

        drift_score = metric.get(
            "value"
        )

        # Vérifications de sécurité
        if (
            feature_name is None
            or threshold is None
            or drift_score is None
        ):
            continue

        # Pour Jensen-Shannon et Wasserstein :
        # plus le score est élevé, plus le drift est important
        drift_detected = (
            float(drift_score)
            >= float(threshold)
        )

        drift_metrics.append(
            {
                "feature_name": feature_name,
                "drift_detected": drift_detected,
                "drift_score": float(
                    drift_score
                ),
                "test_name": test_name,
            }
        )

    return drift_metrics


def save_drift_results(
    drift_metrics
):

    with psycopg.connect(
        DATABASE_URL
    ) as conn:

        with conn.cursor() as cursor:

            for result in drift_metrics:

                cursor.execute(
                    """
                    INSERT INTO drift_results (
                        feature_name,
                        drift_detected,
                        drift_score,
                        test_name,
                        reference_period,
                        current_period
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s
                    )
                    """,
                    (
                        result[
                            "feature_name"
                        ],
                        result[
                            "drift_detected"
                        ],
                        result[
                            "drift_score"
                        ],
                        result[
                            "test_name"
                        ],
                        REFERENCE_PERIOD,
                        CURRENT_PERIOD,
                    ),
                )

        conn.commit()


if __name__ == "__main__":

    print(
        "Lecture du rapport Evidently..."
    )

    drift_metrics = (
        load_drift_metrics()
    )

    print(
        "Nombre de features trouvées :",
        len(drift_metrics),
    )

    drifted_count = sum(
        result[
            "drift_detected"
        ]
        for result
        in drift_metrics
    )

    print(
        "Features en drift :",
        drifted_count,
    )

    print(
        "Features sans drift :",
        len(drift_metrics)
        - drifted_count,
    )

    print()

    print(
        "Enregistrement dans PostgreSQL..."
    )

    save_drift_results(
        drift_metrics
    )

    print()
    print("=" * 60)

    print(
        "Résultats de drift sauvegardés."
    )

    print(
        "Nombre de lignes insérées :",
        len(drift_metrics),
    )

    print(
        "Drifts détectés :",
        drifted_count,
    )

    print("=" * 60)