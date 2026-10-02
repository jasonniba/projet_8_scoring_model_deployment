import cProfile
import pstats
import time
from pathlib import Path

import pandas as pd

from src.api.main import (
    PredictionRequest,
    predict,
)


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


SIMULATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "simulation_features.parquet"
)


REPORTS_DIR = (
    PROJECT_ROOT
    / "reports"
)


REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


PROFILE_PATH = (
    REPORTS_DIR
    / "api_profile.prof"
)


PROFILE_TEXT_PATH = (
    REPORTS_DIR
    / "api_profile.txt"
)


NUMBER_OF_REQUESTS = 20


# =========================================================
# PREPARATION D'UNE REQUETE
# =========================================================

print("=" * 70)
print("PROFILING API - CREDIT SCORING")
print("=" * 70)


X = pd.read_parquet(
    SIMULATION_PATH
)


row = X.iloc[0]


features = {
    column: (
        None
        if pd.isna(value)
        else float(value)
    )
    for column, value in row.items()
}


request = PredictionRequest(
    features=features
)


print(
    "Nombre de features :",
    len(features),
)


# =========================================================
# WARM-UP
# =========================================================

print()
print(
    "Warm-up du modèle..."
)


for _ in range(3):

    predict(
        request
    )


print(
    "Warm-up terminé."
)


# =========================================================
# FONCTION A PROFILER
# =========================================================

def run_predictions():

    for _ in range(
        NUMBER_OF_REQUESTS
    ):

        predict(
            request
        )


# =========================================================
# PROFILING
# =========================================================

print()
print(
    f"Profiling de {NUMBER_OF_REQUESTS} requêtes..."
)


profiler = cProfile.Profile()


start_time = time.perf_counter()


profiler.enable()


run_predictions()


profiler.disable()


total_duration = (
    time.perf_counter()
    - start_time
)


# =========================================================
# SAUVEGARDE DU PROFIL
# =========================================================

profiler.dump_stats(
    str(PROFILE_PATH)
)


with open(
    PROFILE_TEXT_PATH,
    "w",
    encoding="utf-8",
) as file:

    stats = pstats.Stats(
        profiler,
        stream=file,
    )

    stats.strip_dirs()

    stats.sort_stats(
        "cumulative"
    )

    stats.print_stats(
        40
    )


# =========================================================
# AFFICHAGE TERMINAL
# =========================================================

stats = pstats.Stats(
    profiler
)


stats.strip_dirs()

stats.sort_stats(
    "cumulative"
)


print()
print("=" * 70)
print("TOP 30 - TEMPS CUMULE")
print("=" * 70)


stats.print_stats(
    30
)


print()
print("=" * 70)
print("RESUME")
print("=" * 70)


print(
    "Nombre de requêtes :",
    NUMBER_OF_REQUESTS,
)


print(
    f"Temps total : "
    f"{total_duration:.3f} sec"
)


print(
    f"Temps moyen complet : "
    f"{(
        total_duration
        / NUMBER_OF_REQUESTS
    ) * 1000:.3f} ms/requête"
)


print(
    f"Débit séquentiel : "
    f"{NUMBER_OF_REQUESTS / total_duration:.2f} req/s"
)


print()
print(
    "Profil binaire :",
    PROFILE_PATH,
)


print(
    "Rapport texte :",
    PROFILE_TEXT_PATH,
)


print()
print("=" * 70)
print("PROFILING TERMINE")
print("=" * 70)