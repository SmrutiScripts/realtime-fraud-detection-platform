# api.py
"""
FastAPI app that:
1. Loads the trained fraud model & scaler.
2. Exposes an endpoint /score_transaction to score a single transaction.
3. Logs each transaction + prediction into a SQLite database (fraud.db).

Run:
    uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
from datetime import datetime
from contextlib import asynccontextmanager
import sqlite3
import joblib
import numpy as np
import threading

# ---------- Pydantic Request Model ----------


class TransactionIn(BaseModel):
    """
    This defines the expected JSON body for a scoring request.
    Example JSON:
    {
        "amount": 1234.5,
        "time_since_last_txn": 60,
        "is_international": 1,
        "device_trust_score": 0.3,
        "num_recent_chargebacks": 2
    }
    """

    amount: float = Field(..., gt=0)
    time_since_last_txn: float = Field(..., ge=0)
    is_international: int = Field(..., ge=0, le=1)
    device_trust_score: float = Field(..., ge=0.0, le=1.0)
    num_recent_chargebacks: int = Field(..., ge=0)


# ---------- Globals / Paths ----------

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
DB_PATH = "fraud.db"

model = None
scaler = None
db_lock = threading.Lock()  # simple lock to avoid concurrent writes issues


# ---------- Init Helpers ----------


def init_model():
    global model, scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded.")


def init_db():
    """
    Create the SQLite database and table if they don't exist.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            amount REAL NOT NULL,
            time_since_last_txn REAL NOT NULL,
            is_international INTEGER NOT NULL,
            device_trust_score REAL NOT NULL,
            num_recent_chargebacks INTEGER NOT NULL,
            fraud_probability REAL NOT NULL,
            is_fraud_flag INTEGER NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()
    print("Database initialized (fraud.db).")


# ---------- Lifespan (replaces @app.on_event) ----------


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_model()
    init_db()
    yield
    # Shutdown (optional: close resources, etc.)


app = FastAPI(
    title="Real-Time Fraud Detection API",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- Helper to Log to DB ----------


def log_transaction_to_db(txn: TransactionIn, fraud_proba: float, is_fraud_flag: int):
    """
    Insert a row into SQLite.
    """
    with db_lock:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO transactions (
                timestamp,
                amount,
                time_since_last_txn,
                is_international,
                device_trust_score,
                num_recent_chargebacks,
                fraud_probability,
                is_fraud_flag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(timespec="seconds"),
                txn.amount,
                txn.time_since_last_txn,
                txn.is_international,
                txn.device_trust_score,
                txn.num_recent_chargebacks,
                fraud_proba,
                is_fraud_flag,
            ),
        )
        conn.commit()
        conn.close()


# ---------- API Endpoint ----------


@app.post("/score_transaction")
def score_transaction(txn: TransactionIn):
    """
    Score a transaction for fraud risk.

    Returns:
    {
        "fraud_probability": 0.87,
        "is_fraud": true,
        "threshold": 0.7
    }
    """
    feature_vector = np.array(
        [
            [
                txn.amount,
                txn.time_since_last_txn,
                txn.is_international,
                txn.device_trust_score,
                txn.num_recent_chargebacks,
            ]
        ]
    )

    # Scale using the same scaler from training
    features_scaled = scaler.transform(feature_vector)

    # Predict probability of fraud class (class label 1)
    proba = float(model.predict_proba(features_scaled)[0][1])

    # Simple threshold rule (you can tune this)
    threshold = 0.7
    is_fraud_flag = int(proba >= threshold)

    # Log to DB
    log_transaction_to_db(txn, proba, is_fraud_flag)

    return {
        "fraud_probability": proba,
        "is_fraud": bool(is_fraud_flag),
        "threshold": threshold,
    }
