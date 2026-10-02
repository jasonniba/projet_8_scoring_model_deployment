import os

import numpy as np
import pandas as pd
import psycopg
import streamlit as st

from elasticsearch import Elasticsearch


# =========================================================
# CONFIGURATION STREAMLIT
# =========================================================

st.set_page_config(
    page_title="Credit Scoring Monitoring",
    page_icon="📊",
    layout="wide",
)


# =========================================================
# CONFIGURATION POSTGRESQL
# =========================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://scoring_user:scoring_password@localhost:5433/scoring_db",
)


# =========================================================
# CONFIGURATION ELASTICSEARCH
# =========================================================

ELASTICSEARCH_URL = os.getenv(
    "ELASTICSEARCH_URL",
    "http://localhost:9200",
)

ELASTICSEARCH_INDEX = os.getenv(
    "ELASTICSEARCH_INDEX",
    "credit-scoring-api-logs",
)


# =========================================================
# CHARGEMENT DES PREDICTIONS POSTGRESQL
# =========================================================

@st.cache_data(ttl=30)
def load_predictions():

    query = """
        SELECT
            id,
            created_at,
            prediction,
            default_probability,
            inference_time_ms,
            request_duration_ms,
            status_code,
            error_message
        FROM predictions
        ORDER BY created_at DESC
    """

    with psycopg.connect(DATABASE_URL) as conn:

        predictions_df = pd.read_sql_query(
            query,
            conn,
        )

    return predictions_df


# =========================================================
# CHARGEMENT DES RESULTATS DE DRIFT POSTGRESQL
# =========================================================

@st.cache_data(ttl=30)
def load_drift_results():

    query = """
        SELECT
            id,
            created_at,
            feature_name,
            drift_detected,
            drift_score,
            test_name,
            reference_period,
            current_period
        FROM drift_results
        ORDER BY drift_score DESC
    """

    with psycopg.connect(DATABASE_URL) as conn:

        drift_df = pd.read_sql_query(
            query,
            conn,
        )

    return drift_df


# =========================================================
# CHARGEMENT DES LOGS ELASTICSEARCH
# =========================================================

@st.cache_data(ttl=30)
def load_elasticsearch_logs():

    es = Elasticsearch(
        ELASTICSEARCH_URL
    )

    response = es.search(
        index=ELASTICSEARCH_INDEX,
        size=100,
        sort=[
            {
                "@timestamp": {
                    "order": "desc"
                }
            }
        ],
        query={
            "match_all": {}
        },
    )

    logs = [
        hit["_source"]
        for hit in response["hits"]["hits"]
    ]

    return pd.DataFrame(logs)


# =========================================================
# TITRE
# =========================================================

st.title(
    "📊 Monitoring du modèle de Credit Scoring"
)

st.caption(
    "Suivi des prédictions, performances API, Data Drift et logs techniques"
)


# =========================================================
# CHARGEMENT POSTGRESQL
# =========================================================

try:

    predictions = load_predictions()
    drift_results = load_drift_results()

except Exception as exc:

    st.error(
        f"Erreur de connexion PostgreSQL : {exc}"
    )

    st.stop()


# =========================================================
# 1. MONITORING DE L'API
# =========================================================

st.header(
    "1. Monitoring de l'API"
)


total_requests = len(
    predictions
)


success_requests = int(
    (
        predictions["status_code"]
        == 200
    ).sum()
)


error_requests = int(
    (
        predictions["status_code"]
        != 200
    ).sum()
)


valid_latency = (
    predictions[
        "request_duration_ms"
    ]
    .dropna()
)


valid_inference = (
    predictions[
        "inference_time_ms"
    ]
    .dropna()
)


if not valid_latency.empty:

    avg_latency = (
        valid_latency.mean()
    )

    p95_latency = (
        valid_latency.quantile(
            0.95
        )
    )

else:

    avg_latency = 0
    p95_latency = 0


if not valid_inference.empty:

    avg_inference = (
        valid_inference.mean()
    )

else:

    avg_inference = 0


api_col1, api_col2, api_col3, api_col4 = st.columns(4)


api_col1.metric(
    "Requêtes totales",
    total_requests,
)


api_col2.metric(
    "Requêtes réussies",
    success_requests,
)


api_col3.metric(
    "Erreurs",
    error_requests,
)


api_col4.metric(
    "Latence moyenne",
    f"{avg_latency:.2f} ms",
)


api_col5, api_col6 = st.columns(2)


api_col5.metric(
    "Latence P95",
    f"{p95_latency:.2f} ms",
)


api_col6.metric(
    "Temps d'inférence moyen",
    f"{avg_inference:.2f} ms",
)


if total_requests > 0:

    error_rate = (
        error_requests
        / total_requests
    )

else:

    error_rate = 0


st.write(
    "Taux d'erreur API :",
    f"{error_rate:.2%}",
)


# =========================================================
# 2. DISTRIBUTION DES SCORES
# =========================================================

st.header(
    "2. Distribution des scores"
)


scores = (
    predictions.loc[
        predictions[
            "status_code"
        ]
        == 200,
        "default_probability",
    ]
    .dropna()
)


if not scores.empty:

    histogram, bins = np.histogram(
        scores,
        bins=20,
        range=(0, 1),
    )


    score_distribution = pd.DataFrame(
        {
            "Probabilité de défaut": (
                bins[:-1]
            ),

            "Nombre de clients": (
                histogram
            ),
        }
    )


    score_distribution = (
        score_distribution.set_index(
            "Probabilité de défaut"
        )
    )


    st.bar_chart(
        score_distribution
    )


    score_col1, score_col2, score_col3 = (
        st.columns(3)
    )


    score_col1.metric(
        "Probabilité moyenne",
        f"{scores.mean():.4f}",
    )


    score_col2.metric(
        "Probabilité minimale",
        f"{scores.min():.4f}",
    )


    score_col3.metric(
        "Probabilité maximale",
        f"{scores.max():.4f}",
    )


else:

    st.info(
        "Aucun score disponible."
    )


# =========================================================
# 3. MONITORING DU DATA DRIFT
# =========================================================

st.header(
    "3. Monitoring du Data Drift"
)


total_features = len(
    drift_results
)


if total_features > 0:

    drifted_features = int(
        drift_results[
            "drift_detected"
        ].sum()
    )

    drift_share = (
        drifted_features
        / total_features
    )

else:

    drifted_features = 0
    drift_share = 0


drift_col1, drift_col2, drift_col3 = (
    st.columns(3)
)


drift_col1.metric(
    "Features analysées",
    total_features,
)


drift_col2.metric(
    "Features en drift",
    drifted_features,
)


drift_col3.metric(
    "Part des features en drift",
    f"{drift_share:.2%}",
)


DATASET_DRIFT_THRESHOLD = 0.50


if drift_share >= DATASET_DRIFT_THRESHOLD:

    st.error(
        "⚠️ Dataset Drift global détecté."
    )

else:

    st.success(
        "✅ Dataset Drift global non détecté."
    )


st.caption(
    f"Seuil global Evidently utilisé : "
    f"{DATASET_DRIFT_THRESHOLD:.0%}"
)


# =========================================================
# FEATURES EN DRIFT
# =========================================================

st.subheader(
    "Features présentant un drift"
)


drifted_df = (
    drift_results[
        drift_results[
            "drift_detected"
        ]
        == True
    ]
    .copy()
)


if not drifted_df.empty:

    st.dataframe(
        drifted_df[
            [
                "feature_name",
                "drift_score",
                "test_name",
                "reference_period",
                "current_period",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


    st.subheader(
        "Top 10 des scores de drift"
    )


    top_drift = (
        drifted_df[
            [
                "feature_name",
                "drift_score",
            ]
        ]
        .sort_values(
            "drift_score",
            ascending=False,
        )
        .head(10)
        .set_index(
            "feature_name"
        )
    )


    st.bar_chart(
        top_drift
    )


else:

    st.success(
        "Aucune feature en drift."
    )


# =========================================================
# 4. DERNIERES REQUETES API
# =========================================================

st.header(
    "4. Dernières requêtes API"
)


if not predictions.empty:

    st.dataframe(
        predictions[
            [
                "created_at",
                "prediction",
                "default_probability",
                "inference_time_ms",
                "request_duration_ms",
                "status_code",
                "error_message",
            ]
        ].head(50),
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "Aucune requête enregistrée."
    )


# =========================================================
# 5. LOGS ELASTICSEARCH
# =========================================================

st.header(
    "5. Logs Elasticsearch"
)


try:

    logs_df = (
        load_elasticsearch_logs()
    )


    if logs_df.empty:

        st.info(
            "Aucun log Elasticsearch disponible."
        )


    else:

        total_logs = len(
            logs_df
        )


        if "event" in logs_df.columns:

            success_logs = int(
                (
                    logs_df["event"]
                    == "prediction_success"
                ).sum()
            )


            error_logs = int(
                (
                    logs_df["event"]
                    == "prediction_error"
                ).sum()
            )


        else:

            success_logs = 0
            error_logs = 0


        log_col1, log_col2, log_col3 = (
            st.columns(3)
        )


        log_col1.metric(
            "Logs affichés",
            total_logs,
        )


        log_col2.metric(
            "Prediction Success",
            success_logs,
        )


        log_col3.metric(
            "Prediction Error",
            error_logs,
        )


        # ---------------------------------------------
        # REPARTITION DES EVENEMENTS
        # ---------------------------------------------

        if "event" in logs_df.columns:

            st.subheader(
                "Répartition des événements"
            )


            events_distribution = (
                logs_df[
                    "event"
                ]
                .value_counts()
                .rename(
                    "Nombre"
                )
                .to_frame()
            )


            st.bar_chart(
                events_distribution
            )


        # ---------------------------------------------
        # TABLEAU DES LOGS
        # ---------------------------------------------

        st.subheader(
            "Derniers événements de l'API"
        )


        columns_to_display = [
            column
            for column in [
                "@timestamp",
                "event",
                "status_code",
                "prediction",
                "default_probability",
                "inference_time_ms",
                "request_duration_ms",
                "error_message",
                "missing_features_count",
                "unknown_features_count",
            ]
            if column in logs_df.columns
        ]


        if columns_to_display:

            st.dataframe(
                logs_df[
                    columns_to_display
                ],
                use_container_width=True,
                hide_index=True,
            )


        else:

            st.dataframe(
                logs_df,
                use_container_width=True,
                hide_index=True,
            )


except Exception as exc:

    st.warning(
        "Elasticsearch indisponible : "
        f"{exc}"
    )


# =========================================================
# 6. ARCHITECTURE DU MONITORING
# =========================================================

st.header(
    "6. Architecture du monitoring"
)


st.code(
    """
FastAPI / Modèle de scoring
          |
          |----> PostgreSQL
          |       |
          |       |---- prédictions
          |       |---- probabilités
          |       |---- latences
          |       |---- erreurs
          |       |---- résultats de drift
          |
          |----> Elasticsearch
                  |
                  |---- logs prediction_success
                  |---- logs prediction_error

PostgreSQL
     |
     v
Données de production
     |
     +----------+
                |
Reference ------+
                |
                v
           Evidently AI
                |
                v
          Data Drift
                |
                v
          PostgreSQL
                |
                v
            Streamlit
""",
    language="text",
)