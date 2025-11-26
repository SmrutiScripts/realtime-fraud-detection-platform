# dashboard.py
"""
Streamlit dashboard to visualize fraud detection metrics.

Run:
    streamlit run dashboard.py
"""

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

DB_PATH = "fraud.db"


def load_data(limit: int = 5000) -> pd.DataFrame:
    """
    Load the most recent transactions from SQLite database.
    """
    if not Path(DB_PATH).exists():
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT *
        FROM transactions
        ORDER BY id DESC
        LIMIT {limit}
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Reverse so newest at bottom
    df = df.sort_values("id")
    return df


def main():
    st.set_page_config(page_title="Fraud Analytics Dashboard", layout="wide")
    st.title("🕵️ Real-Time Fraud Detection – Analytics Dashboard")

    st.sidebar.header("Controls")
    limit = st.sidebar.slider(
        "Load last N transactions",
        min_value=100,
        max_value=10000,
        value=2000,
        step=100,
    )
    auto_refresh = st.sidebar.checkbox("Auto-refresh every 5 seconds", value=False)

    df = load_data(limit=limit)

    if df.empty:
        st.warning("No data found yet. Start the API and simulator first.")
        return

    # ---------- KPIs ----------
    total_txns = len(df)
    flagged = int(df["is_fraud_flag"].sum())
    fraud_rate = flagged / total_txns * 100 if total_txns > 0 else 0.0
    avg_proba = df["fraud_probability"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", total_txns)
    col2.metric("Flagged as Fraud", flagged)
    col3.metric("Fraud Rate (%)", f"{fraud_rate:.2f}")
    col4.metric("Avg Fraud Probability", f"{avg_proba:.3f}")

    st.markdown("---")

    # ---------- Fraud probability over time ----------
    st.subheader("Fraud Probability Over Time")
    df_plot = df[["id", "fraud_probability", "is_fraud_flag"]].copy()
    df_plot = df_plot.set_index("id")
    st.line_chart(df_plot["fraud_probability"])

    # ---------- Distribution of probabilities ----------
    st.subheader("Distribution of Fraud Probability")
    fig, ax = plt.subplots()
    ax.hist(df["fraud_probability"], bins=30)
    ax.set_xlabel("Fraud Probability")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # ---------- Recent high-risk transactions ----------
    st.subheader("Recent High-Risk Transactions")
    high_risk = df[df["fraud_probability"] >= 0.7].copy()
    if high_risk.empty:
        st.info("No high-risk transactions found (probability >= 0.7).")
    else:
        # sort by timestamp instead of 'id' (since 'id' isn't in the subset)
        high_risk = high_risk.sort_values("timestamp", ascending=False)
        st.dataframe(
            high_risk[
                [
                    "timestamp",
                    "amount",
                    "time_since_last_txn",
                    "is_international",
                    "device_trust_score",
                    "num_recent_chargebacks",
                    "fraud_probability",
                    "is_fraud_flag",
                ]
            ],
            use_container_width=True,
        )

    # ---------- Auto-refresh ----------
    if auto_refresh:
        import time

        time.sleep(5)
        # For newer Streamlit versions, st.rerun() is preferred
        st.experimental_rerun()


if __name__ == "__main__":
    main()
