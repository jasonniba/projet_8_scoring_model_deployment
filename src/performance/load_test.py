import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests


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


thread_local = threading.local()


# =========================================================
# SESSION HTTP PAR THREAD
# =========================================================

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
# PREPARATION DES PAYLOADS
# =========================================================

def load_payloads():

    df = pd.read_parquet(
        SIMULATION_PATH
    )

    payloads = []

    for _, row in df.iterrows():

        features = {
            column: (
                None
                if pd.isna(value)
                else float(value)
            )
            for column, value
            in row.items()
        }

        payloads.append(
            {
                "features": features
            }
        )

    return payloads


# =========================================================
# UNE REQUETE
# =========================================================

def send_request(
    url,
    payload,
    timeout,
):

    session = get_session()

    start = time.perf_counter()

    try:

        response = session.post(
            url,
            json=payload,
            timeout=timeout,
        )

        latency_ms = (
            time.perf_counter()
            - start
        ) * 1000

        return {
            "success": (
                response.status_code
                == 200
            ),
            "status_code": (
                response.status_code
            ),
            "latency_ms": (
                latency_ms
            ),
            "error": None,
        }

    except requests.RequestException as exc:

        latency_ms = (
            time.perf_counter()
            - start
        ) * 1000

        return {
            "success": False,
            "status_code": None,
            "latency_ms": (
                latency_ms
            ),
            "error": str(exc),
        }


# =========================================================
# TEST DE CHARGE
# =========================================================

def run_load_test(
    url,
    payloads,
    total_requests,
    concurrency,
    timeout,
):

    print()
    print("=" * 70)
    print("LOAD TEST - CREDIT SCORING API")
    print("=" * 70)

    print(
        "URL :",
        url,
    )

    print(
        "Nombre de requêtes :",
        total_requests,
    )

    print(
        "Concurrence :",
        concurrency,
    )

    print()

    # -----------------------------------------------------
    # Warm-up
    # -----------------------------------------------------

    print(
        "Warm-up..."
    )

    for i in range(3):

        send_request(
            url,
            payloads[
                i % len(payloads)
            ],
            timeout,
        )

    print(
        "Warm-up terminé."
    )

    # -----------------------------------------------------
    # Test
    # -----------------------------------------------------

    start_test = (
        time.perf_counter()
    )

    results = []

    with ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:

        futures = []

        for i in range(
            total_requests
        ):

            payload = payloads[
                i % len(payloads)
            ]

            future = executor.submit(
                send_request,
                url,
                payload,
                timeout,
            )

            futures.append(
                future
            )

        for future in as_completed(
            futures
        ):

            results.append(
                future.result()
            )

    total_duration = (
        time.perf_counter()
        - start_test
    )

    # -----------------------------------------------------
    # Analyse
    # -----------------------------------------------------

    successes = [
        result
        for result in results
        if result[
            "success"
        ]
    ]

    errors = [
        result
        for result in results
        if not result[
            "success"
        ]
    ]

    success_latencies = np.array(
        [
            result[
                "latency_ms"
            ]
            for result
            in successes
        ]
    )

    success_count = len(
        successes
    )

    error_count = len(
        errors
    )

    error_rate = (
        error_count
        / total_requests
        * 100
    )

    throughput = (
        success_count
        / total_duration
    )

    attempted_throughput = (
        total_requests
        / total_duration
    )

    # -----------------------------------------------------
    # RESULTATS
    # -----------------------------------------------------

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
        f"{success_count}"
    )

    print(
        f"Erreurs              : "
        f"{error_count}"
    )

    print(
        f"Taux d'erreur        : "
        f"{error_rate:.2f} %"
    )

    print(
        f"Débit réussi         : "
        f"{throughput:.2f} req/s"
    )

    print(
        f"Débit envoyé         : "
        f"{attempted_throughput:.2f} req/s"
    )

    if len(
        success_latencies
    ) > 0:

        print()
        print(
            "LATENCE HTTP CLIENT"
        )

        print(
            "-" * 70
        )

        print(
            f"Moyenne : "
            f"{success_latencies.mean():.3f} ms"
        )

        print(
            f"P50     : "
            f"{np.percentile(success_latencies, 50):.3f} ms"
        )

        print(
            f"P95     : "
            f"{np.percentile(success_latencies, 95):.3f} ms"
        )

        print(
            f"P99     : "
            f"{np.percentile(success_latencies, 99):.3f} ms"
        )

        print(
            f"Min     : "
            f"{success_latencies.min():.3f} ms"
        )

        print(
            f"Max     : "
            f"{success_latencies.max():.3f} ms"
        )

    if errors:

        print()
        print(
            "PREMIERES ERREURS"
        )

        print(
            "-" * 70
        )

        for error in errors[:5]:

            print(
                error
            )

    print()
    print("=" * 70)
    print("LOAD TEST TERMINE")
    print("=" * 70)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--url",
        default=(
            "http://127.0.0.1:8000/predict"
        ),
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=30,
    )

    args = parser.parse_args()

    payloads = (
        load_payloads()
    )

    run_load_test(
        url=args.url,
        payloads=payloads,
        total_requests=args.requests,
        concurrency=args.concurrency,
        timeout=args.timeout,
    )