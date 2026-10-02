import atexit
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from elasticsearch import Elasticsearch


# =========================================================
# CONFIGURATION
# =========================================================

ELASTICSEARCH_URL = os.getenv(
    "ELASTICSEARCH_URL",
    "http://localhost:9200",
)

INDEX_NAME = (
    "credit-scoring-api-logs"
)


# =========================================================
# CLIENT ELASTICSEARCH
# =========================================================

es = Elasticsearch(
    ELASTICSEARCH_URL
)


# =========================================================
# EXECUTEUR ASYNCHRONE
# =========================================================

# Les logs Elasticsearch sont envoyés
# en arrière-plan afin de ne pas bloquer
# la réponse de l'API.
#
# 4 workers permettent plusieurs écritures
# simultanées sans créer un thread par requête.

executor = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="elasticsearch_logger",
)


# =========================================================
# ENVOI REEL DU LOG
# =========================================================

def _send_event(
    document: dict,
):

    try:

        es.index(
            index=INDEX_NAME,
            document=document,
        )

    except Exception as exc:

        # Le monitoring ne doit jamais
        # interrompre une prédiction.
        print(
            "Elasticsearch logging error:",
            exc,
        )


# =========================================================
# FONCTION UTILISEE PAR L'API
# =========================================================

def log_event(
    event: dict,
):

    # Copie pour éviter de modifier
    # le dictionnaire original.
    document = dict(
        event
    )

    document[
        "@timestamp"
    ] = datetime.now(
        timezone.utc
    ).isoformat()

    try:

        # L'envoi est placé dans la file
        # d'exécution puis l'API continue
        # immédiatement.
        executor.submit(
            _send_event,
            document,
        )

    except RuntimeError as exc:

        print(
            "Elasticsearch async logging error:",
            exc,
        )


# =========================================================
# ARRET PROPRE
# =========================================================

def shutdown_logger():

    executor.shutdown(
        wait=True
    )


atexit.register(
    shutdown_logger
)