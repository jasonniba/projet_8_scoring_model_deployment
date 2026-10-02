from pathlib import Path
import time

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SIMULATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "simulation_features.parquet"
)

API_URL = "http://127.0.0.1:8000/predict"


def prepare_features(row):
    """
    Convertit une ligne pandas en dictionnaire JSON compatible.

    Les NaN deviennent None afin qu'ils puissent être envoyés
    correctement à FastAPI.
    """

    return {
        feature: (
            None
            if pd.isna(value)
            else float(value)
        )
        for feature, value in row.items()
    }


def main():

    print("Chargement du batch de simulation...")

    simulation_df = pd.read_parquet(
        SIMULATION_PATH
    )

    print(
        "Shape du batch :",
        simulation_df.shape,
    )

    if simulation_df.shape[1] != 401:
        raise ValueError(
            f"401 features attendues, "
            f"{simulation_df.shape[1]} trouvées."
        )

    session = requests.Session()

    success_count = 0
    error_count = 0

    start_time = time.perf_counter()

    print()
    print("Envoi des requêtes vers FastAPI...")

    for index, row in simulation_df.iterrows():

        features = prepare_features(row)

        try:

            response = session.post(
                API_URL,
                json={
                    "features": features
                },
                timeout=30,
            )

            if response.status_code == 200:
                success_count += 1

            else:
                error_count += 1

                print(
                    f"Erreur ligne {index} : "
                    f"status={response.status_code}"
                )

                print(
                    response.text[:500]
                )

        except requests.RequestException as exc:

            error_count += 1

            print(
                f"Erreur réseau ligne {index} :",
                exc,
            )

        processed = (
            success_count
            + error_count
        )

        if processed % 50 == 0:

            print(
                f"{processed}/"
                f"{len(simulation_df)} requêtes traitées"
            )

    total_duration = (
        time.perf_counter()
        - start_time
    )

    print()
    print("=" * 60)

    print(
        "Simulation terminée."
    )

    print(
        "Requêtes réussies :",
        success_count,
    )

    print(
        "Requêtes en erreur :",
        error_count,
    )

    print(
        "Durée totale :",
        round(total_duration, 2),
        "secondes",
    )

    if total_duration > 0:

        print(
            "Débit moyen :",
            round(
                len(simulation_df)
                / total_duration,
                2,
            ),
            "requêtes/seconde",
        )

    print("=" * 60)


if __name__ == "__main__":
    main()