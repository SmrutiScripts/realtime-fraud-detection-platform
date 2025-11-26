# simulator.py
"""
Continuously generates random transactions and sends them
to the FastAPI service at /score_transaction.

Run this in a separate terminal while the API is running:
    python simulator.py
"""

import time
import random
from datetime import datetime
import requests

API_URL = "http://localhost:8000/score_transaction"


def generate_random_transaction():
    """
    Returns a dict compatible with TransactionIn model.
    """
    amount = random.lognormvariate(7, 0.7)
    time_since_last_txn = random.expovariate(1 / 300)
    is_international = 1 if random.random() < 0.15 else 0
    device_trust_score = random.betavariate(2, 5)
    num_recent_chargebacks = random.choices([0, 1, 2, 3], weights=[0.8, 0.15, 0.04, 0.01])[0]

    return {
        "amount": amount,
        "time_since_last_txn": time_since_last_txn,
        "is_international": is_international,
        "device_trust_score": device_trust_score,
        "num_recent_chargebacks": num_recent_chargebacks,
    }


def main():
    print("Starting transaction simulator...")
    while True:
        txn = generate_random_transaction()
        try:
            response = requests.post(API_URL, json=txn, timeout=3)
            if response.status_code == 200:
                data = response.json()
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"amount={txn['amount']:.2f} "
                    f"intl={txn['is_international']} "
                    f"trust={txn['device_trust_score']:.2f} "
                    f"proba={data['fraud_probability']:.3f} "
                    f"fraud={data['is_fraud']}"
                )
            else:
                print(f"Error: status={response.status_code}")
        except Exception as e:
            print(f"Request failed: {e}")

        time.sleep(0.5)  # 2 txns per second approx.


if __name__ == "__main__":
    main()
