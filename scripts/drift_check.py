"""
Drift check — Phase 2

Per the Phase 2 decision: the model stays FIXED across refreshes rather than
retraining automatically on every new CSV (a single small batch retraining a
tree ensemble each run risks instability). Instead, each new batch is used
to VALIDATE the existing model: since new CSVs include the real
Delivery_Time outcome, we can measure actual accuracy/F1 on new data, not
guess at drift via distribution comparisons.

If performance degrades past a threshold, this flags "retrain recommended"
for a human decision, it does not retrain automatically. Retraining is a
deliberate, reviewed action, not a silent background one.
"""

import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

BASELINE_ACCURACY = 0.859
BASELINE_F1 = 0.739
F1_DRIFT_THRESHOLD = 0.65  # below this, flag retrain recommended
ACCURACY_DRIFT_THRESHOLD = 0.75


def check_drift(new_cleaned_df, model_path="../data/processed/model.pkl"):
    """
    new_cleaned_df: the newly-cleaned batch (output of feature_engineering.py
                     run against the new CSV), must already have SLA_Breach
                     computed the same way as the original.
    Returns: dict with accuracy, f1, drift_detected (bool), message (str)
    """
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

    clf = model_data["model"]
    columns = model_data["columns"]

    cat_features = ["Weather", "Traffic", "Vehicle", "Area", "Time_Band", "Weekday"]
    num_features = ["Agent_Age", "Agent_Rating", "Distance_km", "Is_Weekend", "Is_Holiday"]

    df = new_cleaned_df.dropna(subset=["Distance_km"]).copy()
    df["Is_Weekend"] = df["Is_Weekend"].astype(int)
    df["Is_Holiday"] = df["Is_Holiday"].astype(int)

    X = pd.get_dummies(df[cat_features + num_features], columns=cat_features)
    # align columns to the training-time schema; new categories not seen in
    # training become all-zero rows for those dummy columns, missing
    # training-time columns are added as zero
    X = X.reindex(columns=columns, fill_value=0)
    y = df["SLA_Breach"].astype(int)

    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    drift_detected = (f1 < F1_DRIFT_THRESHOLD) or (acc < ACCURACY_DRIFT_THRESHOLD)

    if drift_detected:
        message = (
            f"Retrain recommended: F1 dropped to {f1:.3f} (baseline {BASELINE_F1}), "
            f"accuracy {acc*100:.1f}% (baseline {BASELINE_ACCURACY*100:.1f}%). "
            f"Model is still being used for this refresh, flagged for review, not "
            f"auto-retrained."
        )
    else:
        message = f"Model validated on new batch: {acc*100:.1f}% accuracy, {f1:.3f} F1. No drift detected."

    return {
        "accuracy": round(acc, 3),
        "f1": round(f1, 3),
        "baseline_accuracy": BASELINE_ACCURACY,
        "baseline_f1": BASELINE_F1,
        "drift_detected": drift_detected,
        "message": message,
    }
