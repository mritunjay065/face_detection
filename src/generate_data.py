"""
Generates synthetic sample dataset for behavior classification training.
Useful for validating the training pipeline and testing group generalization.
"""
import os
import numpy as np
import pandas as pd


def generate_synthetic_data(output_path: str = "data/features/features.csv", n_subjects: int = 10, samples_per_subject: int = 40):
    rows = []
    labels = ["ON_TARGET", "MIXED_GAZE", "SUSTAINED_LOOK_AWAY", "EYES_CLOSED"]

    for subj_i in range(1, n_subjects + 1):
        subj_id = f"person_{subj_i:02d}"
        for _ in range(samples_per_subject):
            chosen_label = np.random.choice(labels, p=[0.45, 0.25, 0.20, 0.10])

            if chosen_label == "ON_TARGET":
                on_target = np.random.uniform(0.75, 1.0)
                off_target_sec = np.random.uniform(0.0, 1.0)
                gaze_shift = np.random.uniform(0.1, 0.6)
                down_ratio = np.random.uniform(0.0, 0.15)
                yaw = np.random.normal(0.0, 6.0)
                pitch = np.random.normal(0.0, 5.0)
                head_std = np.random.uniform(2.0, 8.0)
                blink_rate = np.random.uniform(10.0, 22.0)
                closure_ratio = np.random.uniform(0.01, 0.05)
                signal_conf = np.random.uniform(0.85, 0.98)

            elif chosen_label == "MIXED_GAZE":
                on_target = np.random.uniform(0.40, 0.74)
                off_target_sec = np.random.uniform(1.0, 2.8)
                gaze_shift = np.random.uniform(0.5, 1.5)
                down_ratio = np.random.uniform(0.15, 0.40)
                yaw = np.random.normal(0.0, 14.0)
                pitch = np.random.normal(-4.0, 10.0)
                head_std = np.random.uniform(6.0, 15.0)
                blink_rate = np.random.uniform(12.0, 28.0)
                closure_ratio = np.random.uniform(0.02, 0.08)
                signal_conf = np.random.uniform(0.75, 0.95)

            elif chosen_label == "SUSTAINED_LOOK_AWAY":
                on_target = np.random.uniform(0.0, 0.35)
                off_target_sec = np.random.uniform(3.0, 5.0)
                gaze_shift = np.random.uniform(0.2, 1.0)
                down_ratio = np.random.uniform(0.1, 0.8)
                yaw = np.random.choice([-1, 1]) * np.random.uniform(20.0, 45.0)
                pitch = np.random.normal(-10.0, 15.0)
                head_std = np.random.uniform(4.0, 12.0)
                blink_rate = np.random.uniform(8.0, 20.0)
                closure_ratio = np.random.uniform(0.01, 0.06)
                signal_conf = np.random.uniform(0.70, 0.92)

            else:  # EYES_CLOSED
                on_target = np.random.uniform(0.0, 0.2)
                off_target_sec = np.random.uniform(3.5, 5.0)
                gaze_shift = np.random.uniform(0.0, 0.2)
                down_ratio = np.random.uniform(0.0, 0.2)
                yaw = np.random.normal(0.0, 10.0)
                pitch = np.random.normal(-15.0, 8.0)
                head_std = np.random.uniform(1.0, 5.0)
                blink_rate = np.random.uniform(0.0, 6.0)
                closure_ratio = np.random.uniform(0.70, 1.0)
                signal_conf = np.random.uniform(0.80, 0.95)

            rows.append({
                "subject_id": subj_id,
                "on_target_ratio": round(float(on_target), 3),
                "longest_off_target_sec": round(float(off_target_sec), 2),
                "gaze_shift_freq": round(float(gaze_shift), 2),
                "downward_gaze_ratio": round(float(down_ratio), 3),
                "mean_head_yaw": round(float(yaw), 2),
                "mean_head_pitch": round(float(pitch), 2),
                "head_movement_std": round(float(head_std), 2),
                "blink_rate_bpm": round(float(blink_rate), 1),
                "eye_closure_ratio": round(float(closure_ratio), 3),
                "signal_confidence": round(float(signal_conf), 3),
                "label": chosen_label,
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} sample feature windows saved to {output_path}")


if __name__ == "__main__":
    generate_synthetic_data()
