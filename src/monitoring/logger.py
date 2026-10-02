import os
from datetime import datetime, timezone

from elasticsearch import Elasticsearch


ELASTICSEARCH_URL = os.getenv(
    "ELASTICSEARCH_URL",
    "http://localhost:9200",
)

ES_INDEX = os.getenv(
    "ELASTICSEARCH_INDEX",
    "credit-scoring-api-logs",
)

es_client = Elasticsearch(ELASTICSEARCH_URL)


def log_event(event: dict) -> None:
    """
    Envoie un événement structuré dans Elasticsearch.

    Une erreur Elasticsearch ne doit jamais empêcher
    l'API de répondre au client.
    """

    document = {
        "@timestamp": datetime.now(timezone.utc).isoformat(),
        **event,
    }

    try:
        es_client.index(
            index=ES_INDEX,
            document=document,
        )

    except Exception as exc:
        print(
            f"Elasticsearch logging error: "
            f"{type(exc).__name__}: {exc}"
        )