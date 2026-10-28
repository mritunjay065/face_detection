"""
Training script for the Tabular / HistGradientBoosting Behavior Classifier.
Splits data by subject_id (GroupShuffleSplit) to ensure generalizability.
"""
import os
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix


def train_model(
    csv_path: str = "data/features/features.csv",
    output_model_path: str = "models/engagement_model.joblib"
):
    if not os.path.exists(csv_path):
        print(f"[Error] Dataset not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    feature_cols = [
        "on_target_ratio",
        "longest_off_target_sec",
        "gaze_shift_freq",
        "downward_gaze_ratio",
        "mean_head_yaw",
        "mean_head_pitch",
        "head_movement_std",
        "blink_rate_bpm",
        "eye_closure_ratio",
        "signal_confidence",
    ]

    # Verify column existence
    existing_cols = [c for c in feature_cols if c in df.columns]
    X = df[existing_cols]
    y = df["label"]
    groups = df["subject_id"] if "subject_id" in df.columns else range(len(df))

    # Generalization split across subjects
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples across distinct subjects.")

    model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=15,
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\n--- Model Evaluation on Unseen Subjects ---")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump({"model": model, "features": existing_cols}, output_model_path)
    print(f"Model saved to {output_model_path}")


if __name__ == "__main__":
    train_model()
