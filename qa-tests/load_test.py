"""Load test — locust scenarios for /predict_by_id.

Memenuhi spec IF5251 nomor QA 3c.

Run:
    locust -f load_test.py --host http://localhost:8000 \\
        --users 50 --spawn-rate 10 --run-time 60s --headless \\
        --html reports/load_test_report.html
"""
import random

from locust import HttpUser, between, task

# Pool of valid transaction IDs (from train_identity.csv).
TXN_IDS = list(range(2987004, 2989000))


class FraudPredictionUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(7)
    def predict_by_id(self):
        txn_id = random.choice(TXN_IDS)
        with self.client.post(
            "/predict_by_id",
            json={"transaction_id": txn_id},
            catch_response=True,
            name="/predict_by_id",
        ) as r:
            if r.status_code == 200:
                r.success()
            elif r.status_code == 404:
                # Unknown ID is a normal outcome of random sampling, not a failure.
                r.success()
            else:
                r.failure(f"status {r.status_code}: {r.text[:100]}")

    @task(2)
    def explain_by_id(self):
        txn_id = random.choice(TXN_IDS)
        with self.client.post(
            "/explain_by_id",
            json={"transaction_id": txn_id},
            catch_response=True,
            name="/explain_by_id",
        ) as r:
            if r.status_code in (200, 404):
                r.success()
            else:
                r.failure(f"status {r.status_code}")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
