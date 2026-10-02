from pathlib import Path

import pandas as pd
import evidently


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "reference_features.parquet"
)

CURRENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "current_features.parquet"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

REPORT_PATH = (
    REPORT_DIR
    / "data_drift_report.html"
)


# ---------------------------------------------------------
# Compatibilité Evidently selon la version installée
# ---------------------------------------------------------

try:
    from evidently import Report
    from evidently.presets import DataDriftPreset

except ImportError:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset


# ---------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------

print("Version Evidently :", evidently.__version__)
print("Chargement des données...")

reference_df = pd.read_parquet(
    REFERENCE_PATH
)

current_df = pd.read_parquet(
    CURRENT_PATH
)


print(
    "Reference :",
    reference_df.shape,
)

print(
    "Current :",
    current_df.shape,
)


# ---------------------------------------------------------
# Vérification des colonnes
# ---------------------------------------------------------

if list(reference_df.columns) != list(current_df.columns):
    raise ValueError(
        "Les colonnes reference et current ne sont pas identiques."
    )


print(
    "Colonnes identiques :",
    True,
)


# ---------------------------------------------------------
# Rapport Evidently
# ---------------------------------------------------------

print()
print("Analyse du Data Drift avec Evidently...")

report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

result = report.run(
    reference_data=reference_df,
    current_data=current_df,
)

JSON_PATH = (
    REPORT_DIR
    / "data_drift_result.json"
)

JSON_PATH.write_text(
    result.json(),
    encoding="utf-8",
)

print(
    "Résultat JSON créé :",
    JSON_PATH,
)

# ---------------------------------------------------------
# Sauvegarde HTML
# ---------------------------------------------------------

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Certaines versions d'Evidently renvoient
# le résultat dans report.run(), d'autres utilisent report.
if result is not None and hasattr(
    result,
    "save_html",
):
    result.save_html(
        str(REPORT_PATH)
    )

elif hasattr(
    report,
    "save_html",
):
    report.save_html(
        str(REPORT_PATH)
    )

else:
    raise RuntimeError(
        "Impossible de sauvegarder le rapport HTML "
        "avec cette version d'Evidently."
    )


print()
print("=" * 60)

print(
    "Analyse terminée."
)

print(
    "Rapport créé :",
    REPORT_PATH,
)

print(
    "Rapport existe :",
    REPORT_PATH.exists(),
)

print("=" * 60)