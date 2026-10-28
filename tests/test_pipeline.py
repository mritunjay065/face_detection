"""
Comprehensive unit tests for scoring, temporal features, eye state, and head pose.
"""
import unittest
import numpy as np
from src.scoring import estimate_engagement, ScoreSmoother
from src.eye_state import EyeStateDetector
from src.features import TemporalFeatureEngine, FrameFeature
from src.head_pose import HeadPoseEstimator
from src.gaze import GazeEstimator


class TestFaceAttentionPipeline(unittest.TestCase):

    def test_scoring_on_target(self):
        res = estimate_engagement(
            gaze_alignment=0.90,
            head_alignment=0.85,
            on_target_ratio=0.95,
            signal_quality=0.95,
            face_visible=True
        )
        self.assertIsNotNone(res.score)
        self.assertGreaterEqual(res.score, 80.0)
        self.assertEqual(res.state, "ON_TARGET")
        self.assertGreaterEqual(res.confidence, 0.9)

    def test_scoring_face_not_visible(self):
        res = estimate_engagement(
            gaze_alignment=0.9,
            head_alignment=0.9,
            on_target_ratio=0.9,
            signal_quality=0.9,
            face_visible=False
        )
        self.assertIsNone(res.score)
        self.assertEqual(res.state, "FACE_NOT_VISIBLE")
        self.assertEqual(res.confidence, 0.0)

    def test_score_smoother(self):
        smoother = ScoreSmoother(alpha=0.5)
        s1 = smoother.update(100.0)
        self.assertEqual(s1, 100.0)
        s2 = smoother.update(50.0)
        self.assertEqual(s2, 75.0)

    def test_ear_calculation(self):
        # Synthetic open eye points
        eye_points = np.array([
            [0.0, 0.0],    # p1 (corner)
            [10.0, 5.0],   # p2 (top 1)
            [20.0, 5.0],   # p3 (top 2)
            [30.0, 0.0],   # p4 (corner)
            [20.0, -5.0],  # p5 (bottom 2)
            [10.0, -5.0],  # p6 (bottom 1)
        ])
        ear = EyeStateDetector.calculate_ear(eye_points)
        self.assertGreater(ear, 0.3)

    def test_temporal_engine(self):
        engine = TemporalFeatureEngine(window_seconds=5.0)
        # Add 5 frames of attentive behavior
        for t in range(5):
            engine.add_frame(FrameFeature(
                timestamp=float(t),
                face_visible=True,
                gaze_alignment=0.9,
                gaze_x=0.5,
                gaze_y=0.5,
                head_alignment=0.9,
                head_yaw=0.0,
                head_pitch=0.0,
                head_roll=0.0,
                ear=0.3,
                is_closed=False,
                is_blink=False,
                signal_quality=0.95
            ))
        metrics = engine.aggregate()
        self.assertEqual(metrics["face_visibility_ratio"], 1.0)
        self.assertEqual(metrics["on_target_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
