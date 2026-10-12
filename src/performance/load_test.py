import argparse
import statistics
import threading
import time

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

import numpy as np
import requests


# =========================================================
# SESSIONS HTTP PAR THREAD
# =========================================================

thread_local = threading.local()


def get_session():

    if not hasattr(
        thread_local,
        "session",
    ):

        thread_local.session = (
            requests.Session()
        )

    return thread_local.session


# =========================================================
# UNE REQUETE
# =========================================================

def send_request(
    predict_url,
    client_id,
):

    session = get_session()

    start = time.perf_counter()

    try:

        response = session.post(
            predict_url,
            json={
                "client_id":
                    int(client_id)
            },
            timeout=30,
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000

        return {
            "success":
                response.status_code == 200,

            "status_code":
                response.status_code,

            "latency_ms":
                elapsed_ms,
        }

    except requests.RequestException:

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000

        return {
            "success":
                False,

            "status_code":
                None,

            "latency_ms":
                elapsed_ms,
        }


# =========================================================
# ARGUMENTS
# =========================================================

parser = argparse.ArgumentParser(
    description=(
        "Load test de l'API "
        "Credit Scoring"
    )
)

parser.add_argument(
    "--requests",
    type=int,
    default=1000,
    help="Nombre total de requêtes",
)

parser.add_argument(
    "--concurrency",
    type=int,
    default=20,
    help="Nombre de requêtes concurrentes",
)

parser.add_argument(
    "--url",
    type=str,
    default="http://127.0.0.1:8000",
    help="URL de base de l'API",
)

args = parser.parse_args()


# =========================================================
# URL
# =========================================================

base_url = args.url.rstrip("/")

predict_url = (
    f"{base_url}/predict"
)

clients_url = (
    f"{base_url}/clients"
)


# =========================================================
# RECUPERATION DES CLIENTS
# =========================================================

print()
print("=" * 70)
print("LOAD TEST - CREDIT SCORING API")
print("=" * 70)

print(
    f"URL : {predict_url}"
)

print(
    f"Nombre de requêtes : "
    f"{args.requests}"
)

print(
    f"Concurrence : "
    f"{args.concurrency}"
)


response = requests.get(
    clients_url,
    timeout=10,
)

response.raise_for_status()

clients_data = response.json()

client_ids = (
    clients_data[
        "examples"
    ]
)


if not client_ids:

    raise RuntimeError(
        "Aucun client disponible "
        "pour le test."
    )


print(
    f"Clients utilisés : "
    f"{len(client_ids)}"
)


# =========================================================
# WARM-UP
# =========================================================

print()
print("Warm-up...")

for client_id in client_ids:

    warmup_response = (
        requests.post(
            predict_url,
            json={
                "client_id":
                    int(client_id)
            },
            timeout=30,
        )
    )

    warmup_response.raise_for_status()


print("Warm-up terminé.")


# =========================================================
# PREPARATION DES REQUETES
# =========================================================

request_client_ids = [

    client_ids[
        i % len(client_ids)
    ]

    for i in range(
        args.requests
    )
]


# =========================================================
# LOAD TEST
# =========================================================

start_total = (
    time.perf_counter()
)


results = []


with ThreadPoolExecutor(
    max_workers=args.concurrency
) as executor:

    futures = [

        executor.submit(
            send_request,
            predict_url,
            client_id,
        )

        for client_id
        in request_client_ids
    ]


    for future in as_completed(
        futures
    ):

        results.append(
            future.result()
        )


total_duration = (
    time.perf_counter()
    - start_total
)


# =========================================================
# RESULTATS
# =========================================================

successes = [
    result
    for result in results
    if result["success"]
]

errors = [
    result
    for result in results
    if not result["success"]
]

latencies = np.asarray(
    [
        result[
            "latency_ms"
        ]
        for result
        in results
    ]
)


successful_throughput = (
    len(successes)
    / total_duration
)

sent_throughput = (
    len(results)
    / total_duration
)

error_rate = (
    len(errors)
    / len(results)
    * 100
)


print()
print("=" * 70)
print("RESULTATS")
print("=" * 70)

print(
    f"Durée totale        : "
    f"{total_duration:.3f} sec"
)

print(
    f"Succès              : "
    f"{len(successes)}"
)

print(
    f"Erreurs             : "
    f"{len(errors)}"
)

print(
    f"Taux d'erreur        : "
    f"{error_rate:.2f} %"
)

print(
    f"Débit réussi         : "
    f"{successful_throughput:.2f} req/s"
)

print(
    f"Débit envoyé         : "
    f"{sent_throughput:.2f} req/s"
)


print()
print("LATENCE HTTP CLIENT")
print("-" * 70)

print(
    f"Moyenne : "
    f"{latencies.mean():.3f} ms"
)

print(
    f"P50     : "
    f"{np.percentile(latencies, 50):.3f} ms"
)

print(
    f"P95     : "
    f"{np.percentile(latencies, 95):.3f} ms"
)

print(
    f"P99     : "
    f"{np.percentile(latencies, 99):.3f} ms"
)

print(
    f"Min     : "
    f"{latencies.min():.3f} ms"
)

print(
    f"Max     : "
    f"{latencies.max():.3f} ms"
)


# =========================================================
# CODES D'ERREUR EVENTUELS
# =========================================================

if errors:

    status_codes = {}

    for error in errors:

        code = error[
            "status_code"
        ]

        status_codes[
            code
        ] = (
            status_codes.get(
                code,
                0,
            )
            + 1
        )

    print()
    print(
        "ERREURS PAR STATUS"
    )

    print(
        "-" * 70
    )

    for code, count in (
        status_codes.items()
    ):

        print(
            f"{code}: {count}"
        )


print()
print("=" * 70)
print("LOAD TEST TERMINE")
print("=" * 70)