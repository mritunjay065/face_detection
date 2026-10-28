"""
Real-time predictor and behavior inference engine.
Combines rule-based estimates and optional trained ML classifiers.
"""
from typing import Dict, Optional, Tuple
import os
import joblib
import numpy as np

from src.scoring import AttentionResult, estimate_engagement, ScoreSmoother


class AttentionPredictor:
    def __init__(self, model_path: Optional[str] = "models/engagement_model.joblib"):
        self.smoother = ScoreSmoother(alpha=0.25)
        self.ml_model = None
        self.feature_names = []

        if model_path and os.path.exists(model_path):
            try:
                bundle = joblib.load(model_path)
                self.ml_model = bundle.get("model")
                self.feature_names = bundle.get("features", [])
                print(f"[Info] Loaded trained ML engagement model from {model_path}")
            except Exception as e:
                print(f"[Warning] Could not load ML model: {e}")

    def predict(
        self,
        temporal_features: Dict[str, float],
        current_gaze_alignment: float,
        current_head_alignment: float,
        face_visible: bool,
        is_closed: bool,
        closure_duration: float
    ) -> AttentionResult:
        # Compute baseline rule-based estimate
        base_result = estimate_engagement(
            gaze_alignment=current_gaze_alignment,
            head_alignment=current_head_alignment,
            on_target_ratio=temporal_features.get("on_target_ratio", 0.0),
            signal_quality=temporal_features.get("signal_confidence", 0.0),
            face_visible=face_visible,
            eyes_closed=is_closed,
            closure_duration=closure_duration
        )

        smoothed_score = self.smoother.update(base_result.score)
        state = base_result.state

        # If trained ML model is available, use it to classify behavioral states
        if self.ml_model is not None and face_visible and base_result.confidence >= 0.4:
            try:
                feat_vec = np.array([[
                    temporal_features.get(col, 0.0) for col in self.feature_names
                ]])
                ml_state = self.ml_model.predict(feat_vec)[0]
                state = ml_state
            except Exception as e:
                pass

        return AttentionResult(
            score=smoothed_score,
            state=state,
            confidence=base_result.confidence,
            gaze_alignment=current_gaze_alignment,
            head_alignment=current_head_alignment,
            on_target_ratio=temporal_features.get("on_target_ratio", 0.0)
        )
