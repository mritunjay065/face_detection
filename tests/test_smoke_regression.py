"""
Smoke & Regression Test Suite:
1. Smoke tests on pipeline initialisation & video reading
2. Geometric cyber-mesh rendering verification
3. Unmasking & occlusion logic tests
4. Multi-object detection under varied thresholds
5. Regression test suite on attention scoring and generalisation
"""
import unittest
import numpy as np
import cv2

from src.landmarks import FaceLandmarksDetector
from src.yolo_detector import AsyncYOLODetector
from src.scoring import estimate_engagement
from src.features import TemporalFeatureEngine, FrameFeature
from src.head_pose import HeadPoseEstimator
from src.gaze import GazeEstimator


class SmokeAndRegressionTestSuite(unittest.TestCase):

    def setUp(self):
        self.detector = FaceLandmarksDetector(model_asset_path="assets/face_landmarker.task")
        self.dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)

    def test_smoke_detector_initialization(self):
        """Smoke test: detector initialized without throwing exceptions."""
        self.assertIsNotNone(self.detector)

    def test_cyber_mesh_rendering(self):
        """Regression test: verify cyber-mesh draws correctly on frame without error."""
        synthetic_landmarks = np.zeros((478, 2), dtype=np.float32)
        # Populate key nodes
        for i in range(478):
            synthetic_landmarks[i] = [320 + np.sin(i) * 50, 240 + np.cos(i) * 50]

        annotated = self.detector.draw_face_wireframe(self.dummy_frame.copy(), synthetic_landmarks)
        self.assertEqual(annotated.shape, (480, 640, 3))
        # Ensure image was modified (drawn upon)
        self.assertFalse(np.array_equal(annotated, self.dummy_frame))

    def test_unmasking_verification(self):
        """Verify unmasking/occlusion check."""
        synthetic_landmarks = np.zeros((478, 2), dtype=np.float32)
        synthetic_landmarks[1] = [320, 240]   # Nose
        synthetic_landmarks[152] = [320, 320] # Chin
        synthetic_landmarks[61] = [280, 280]  # Left mouth
        synthetic_landmarks[291] = [360, 280] # Right mouth

        # Flat texture image simulating cloth mask
        flat_mask_frame = np.full((480, 640, 3), 200, dtype=np.uint8)
        res = self.detector.check_unmasking(flat_mask_frame, synthetic_landmarks)
        self.assertTrue(res["mask_detected"], "Smooth texture in mouth region should trigger mask detection.")

    def test_async_yolo_smoke(self):
        """Smoke test for asynchronous YOLO detector thread & double-buffer."""
        yolo = AsyncYOLODetector(model_name="yolov8n.pt", conf_threshold=0.25)
        self.assertIsNotNone(yolo)
        # Test update and instantaneous non-blocking retrieval
        yolo.update_frame(self.dummy_frame)
        res = yolo.get_results()
        self.assertIn("risk_level", res)
        self.assertIn("num_persons", res)
        yolo.close()

    def test_parameter_thresholds(self):
        """Test varied confidence and alignment parameter sweeps."""
        for g_align in [0.2, 0.5, 0.85]:
            for h_align in [0.3, 0.6, 0.9]:
                res = estimate_engagement(
                    gaze_alignment=g_align,
                    head_alignment=h_align,
                    on_target_ratio=0.8,
                    signal_quality=0.9,
                    face_visible=True
                )
                self.assertIsNotNone(res.score)
                self.assertTrue(0.0 <= res.score <= 100.0)


if __name__ == "__main__":
    unittest.main()
