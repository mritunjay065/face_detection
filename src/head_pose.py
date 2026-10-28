"""
Head pose estimation using facial landmarks and OpenCV solvePnP.
Computes yaw, pitch, and roll in degrees.
"""
from typing import Dict, Optional, Tuple
import cv2
import numpy as np


class HeadPoseEstimator:
    def __init__(self):
        # 3D generic facial model points in world coordinates (mm)
        # Based on canonical face model (nose tip, chin, eye corners, mouth corners)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip (landmark 1)
            (0.0, -330.0, -65.0),     # Chin (landmark 199 or 152)
            (-225.0, 170.0, -135.0),  # Left eye outer corner (landmark 33)
            (225.0, 170.0, -135.0),   # Right eye outer corner (landmark 263)
            (-150.0, -150.0, -125.0), # Left mouth corner (landmark 61)
            (150.0, -150.0, -125.0)   # Right mouth corner (landmark 291)
        ], dtype=np.float64)

        # Standard MediaPipe landmark indices for head pose
        self.landmark_indices = [1, 152, 33, 263, 61, 291]

    def estimate_pose(
        self,
        landmarks_2d: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> Optional[Dict[str, float]]:
        """
        Estimate pitch, yaw, roll from 2D facial landmark coordinates.
        landmarks_2d: array of shape (N, 2) with pixel coordinates (x, y)
        image_shape: (height, width)
        """
        h, w = image_shape[:2]

        if len(landmarks_2d) < max(self.landmark_indices):
            return None

        # Select corresponding 2D points
        image_points = np.array([
            landmarks_2d[idx] for idx in self.landmark_indices
        ], dtype=np.float64)

        # Camera calibration approximation
        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return None

        rmat, _ = cv2.Rodrigues(rvec)
        # Compute Euler angles
        sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
            yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
            roll = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
        else:
            pitch = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
            yaw = np.degrees(np.arctan2(-rmat[2, 0], sy))
            roll = 0.0

        # Calculate alignment normalized score [0, 1]
        # Frontal head pose is near 0 yaw and 0 pitch
        # Allow +/- 15 deg tolerance without penalty
        abs_yaw = abs(yaw)
        abs_pitch = abs(pitch)
        
        yaw_penalty = max(0.0, (abs_yaw - 12.0) / 35.0)
        pitch_penalty = max(0.0, (abs_pitch - 10.0) / 30.0)
        alignment = max(0.0, 1.0 - (yaw_penalty * 0.6 + pitch_penalty * 0.4))

        return {
            "pitch": float(pitch),
            "yaw": float(yaw),
            "roll": float(roll),
            "alignment": float(min(1.0, max(0.0, alignment)))
        }
