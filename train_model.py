# 1. Generates synthetic transaction data.
# 2. Creates a binary label: 1 = fraud, 0 = normal.
# 3. Trains a RandomForestClassifier.
# 4. Saves the model and scaler to disk using joblib.
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def generate_synthetic_transactions(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)
    #Features:
    #amount: transaction amount in local currency
    #time_since_last_txn: seconds since this user’s last transaction
    #is_international: 1 if foreign country, else 0
    #device_trust_score: 0.0 (very risky) to 1.0 (trusted device)
    #num_recent_chargebacks: count of chargebacks in last 90 days


    amount = rng.lognormal(mean=7, sigma=0.7, size=n_samples)
    time_since_last_txn = rng.exponential(scale=300, size=n_samples)
    is_international = rng.binomial(1, 0.15, size=n_samples)
    device_trust_score = rng.beta(a=2, b=5, size=n_samples)
    num_recent_chargebacks = rng.poisson(lam=0.2, size=n_samples)

    data = pd.DataFrame(
        {
            "amount": amount,
            "time_since_last_txn": time_since_last_txn,
            "is_international": is_international,
            "device_trust_score": device_trust_score,
            "num_recent_chargebacks": num_recent_chargebacks,
        }

    )

    # Construct a fraud probability using some "business" rules + noise
    fraud_logit = (
        0.0008 * data["amount"]  
            + 0.001 * (300 - np.clip(data["time_since_last_txn"], 0, 300)) 
            + 2.0 * data["is_international"]
            - 3.0 * data["device_trust_score"]
            + 1.5 * data["num_recent_chargebacks"]
                  )
    
  
    fraud_logit += rng.normal(0, 1.0, size=n_samples)

    fraud_prob = 1 / (1 + np.exp(-fraud_logit))
    labels = rng.binomial(1, np.clip(fraud_prob, 0.001, 0.999))
    data["is_fraud"] = labels

    return data

def main():
    print("Generating synthetic transactions...")
    df = generate_synthetic_transactions()
     
    feature_cols = [
        "amount",
        "time_since_last_txn",
        "is_international",
        "device_trust_score",
        "num_recent_chargebacks",
    ]

    X = df[feature_cols].values
    Y = df["is_fraud"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42, stratify=Y
  )

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    Y_proba = model.predict_proba(X_test)[:,1]

    print("\nClassification Report:\n")
    print(classification_report(Y_test, Y_pred))

    auc = roc_auc_score(Y_test, Y_proba)
    print(f"ROC-AUC: {auc:.4f}")

    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("\nSaved model.pkl and scaler.pkl")

if __name__ == "__main__":
    main()
