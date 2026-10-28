"""
Self-Calibrating Eye State & Blink Detector.
Dynamically tracks running baseline of open-eye EAR to prevent false EYES_CLOSED classifications.
"""
from typing import Dict, List, Tuple
from collections import deque
import numpy as np


class EyeStateDetector:
    def __init__(self, ear_threshold: float = 0.15, blink_time_threshold: float = 0.40):
        # MediaPipe landmark indices for eye contours:
        # Left eye: horizontal (33, 133), vertical (160, 144) & (158, 153)
        self.left_eye_indices = [33, 160, 158, 133, 153, 144]
        # Right eye: horizontal (362, 263), vertical (385, 380) & (387, 373)
        self.right_eye_indices = [362, 385, 387, 263, 373, 380]

        self.fixed_ear_threshold = ear_threshold
        self.blink_time_threshold = blink_time_threshold

        # Running history of open eyes to compute personal baseline
        self.open_ear_history: deque[float] = deque(maxlen=60)
        self.calibrated_open_ear: float = 0.25

        self.closure_start_time: float = 0.0
        self.is_currently_closed: bool = False
        self.blink_count: int = 0

    @staticmethod
    def calculate_ear(eye_points: np.ndarray) -> float:
        """EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)"""
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        h = np.linalg.norm(eye_points[0] - eye_points[3])

        if h < 1e-5:
            return 0.0
        return float((v1 + v2) / (2.0 * h))

    def process(
        self,
        landmarks_2d: np.ndarray,
        current_timestamp: float
    ) -> Dict[str, float | bool]:
        if len(landmarks_2d) < 400:
            return {
                "left_ear": 0.0,
                "right_ear": 0.0,
                "avg_ear": 0.0,
                "dynamic_threshold": self.fixed_ear_threshold,
                "is_closed": False,
                "closure_duration": 0.0,
                "is_blink": False,
            }

        left_pts = np.array([landmarks_2d[i] for i in self.left_eye_indices])
        right_pts = np.array([landmarks_2d[i] for i in self.right_eye_indices])

        left_ear = self.calculate_ear(left_pts)
        right_ear = self.calculate_ear(right_pts)
        avg_ear = (left_ear + right_ear) / 2.0

        # Maintain dynamic threshold: true closed eye is ~50% of candidate's normal open-eye EAR
        # Only learn from clearly open eyes. A closed-eye sequence must never
        # lower the baseline and make the detector lose a prolonged closure.
        if avg_ear > max(0.12, self.calibrated_open_ear * 0.65):
            self.open_ear_history.append(avg_ear)
            if len(self.open_ear_history) >= 15:
                # 80th percentile of open eye
                self.calibrated_open_ear = float(np.percentile(self.open_ear_history, 80))

        # Dynamic threshold: closed eye is when EAR drops below 55% of open-eye baseline
        active_threshold = min(self.fixed_ear_threshold, self.calibrated_open_ear * 0.65)
        # Absolute floor safety: never flag closed if EAR > 0.16
        is_closed = (avg_ear < active_threshold) and (avg_ear < 0.15)

        is_blink_event = False
        duration = 0.0

        if is_closed:
            if not self.is_currently_closed:
                self.is_currently_closed = True
                self.closure_start_time = current_timestamp
            duration = max(0.0, current_timestamp - self.closure_start_time)
        else:
            if self.is_currently_closed:
                duration = max(0.0, current_timestamp - self.closure_start_time)
                if duration <= self.blink_time_threshold:
                    self.blink_count += 1
                    is_blink_event = True
                self.is_currently_closed = False
                self.closure_start_time = 0.0

        return {
            "left_ear": float(left_ear),
            "right_ear": float(right_ear),
            "avg_ear": float(avg_ear),
            "dynamic_threshold": float(active_threshold),
            "is_closed": bool(is_closed),
            "closure_duration": float(duration),
            "is_blink": bool(is_blink_event),
        }
