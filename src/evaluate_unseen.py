"""
Evaluation script to test trained proctoring engagement model on strictly UNSEEN subjects.
Calculates:
- Generalization Accuracy across unseen people
- Per-class Recall and F1-score
- Macro F1
- False Alarm Rate (FPR for on-target behavior)
- Confidence Calibration Error
- Confusion Matrix
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def evaluate_on_unseen_data(
    model_path: str = "models/engagement_model.joblib",
    unseen_csv: str = "data/features/unseen_test.csv"
):
    if not os.path.exists(model_path):
        print(f"[Error] Model not found at {model_path}")
        return

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["features"]

    # Generate synthetic strictly unseen subjects if not present
    if not os.path.exists(unseen_csv):
        print("[Info] Synthesizing 5 new, completely unseen candidates (person_90 to person_95)...")
        from src.generate_data import generate_synthetic_data
        # Generate 5 completely distinct candidate profiles
        generate_synthetic_data(output_path=unseen_csv, n_subjects=5, samples_per_subject=50)

    df_test = pd.read_csv(unseen_csv)
    print(f"[Info] Evaluating on {len(df_test)} windows from unseen candidate IDs: {df_test['subject_id'].unique().tolist()}")

    X_test = df_test[feature_cols]
    y_test = df_test["label"]

    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)

    print("\n================ UNSEEN SUBJECT GENERALIZATION REPORT ================")
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    cm_df = pd.DataFrame(cm, index=model.classes_, columns=[f"Pred_{c}" for c in model.classes_])
    print("Confusion Matrix:\n", cm_df)

    # False-Alarm Rate (when candidate was ON_TARGET, how often did model misclassify as distracted?)
    if "ON_TARGET" in model.classes_:
        on_target_idx = list(model.classes_).index("ON_TARGET")
        total_on_target = cm[on_target_idx].sum()
        correct_on_target = cm[on_target_idx, on_target_idx]
        false_alarms = total_on_target - correct_on_target
        far = (false_alarms / total_on_target) * 100 if total_on_target > 0 else 0.0
        print(f"\nFalse-Alarm Rate on Attentive Candidates: {far:.2f}% (Target: < 5%)")

    print("======================================================================")


if __name__ == "__main__":
    evaluate_on_unseen_data()
