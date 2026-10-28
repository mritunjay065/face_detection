"""
Sensitive Gaze & Eye-Rolling Estimator.
Accurately detects:
- Eye-Rolling / Upward Saccades (looking up towards ceiling or top of screen)
- Downward Gaze (looking at desk, phone, or keyboard)
- Left / Right Gaze shifts
- Normal Center Gaze
"""
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np


class GazeEstimator:
    def __init__(self, history_len: int = 15):
        # MediaPipe landmark indices
        self.left_eye_outer = 33
        self.left_eye_inner = 133
        self.left_eye_top = 159
        self.left_eye_bottom = 145
        self.left_iris_center = 468

        self.right_eye_inner = 362
        self.right_eye_outer = 263
        self.right_eye_top = 386
        self.right_eye_bottom = 374
        self.right_iris_center = 473

        # Calibrated neutral center (default 0.50, 0.50)
        self.neutral_gx = 0.50
        self.neutral_gy = 0.50

        # Rolling history of vertical iris coordinates
        self.gy_history: deque[float] = deque(maxlen=history_len)

    def estimate_gaze(self, landmarks_2d: np.ndarray) -> Dict[str, float | bool | str]:
        if len(landmarks_2d) < 474:
            return {
                "gaze_x": 0.50,
                "gaze_y": 0.50,
                "raw_gy": 0.50,
                "alignment": 0.0,
                "is_eye_rolling": False,
                "gaze_zone": "CENTER",
            }

        # --- Left Eye Measurements ---
        l_outer = landmarks_2d[self.left_eye_outer]
        l_inner = landmarks_2d[self.left_eye_inner]
        l_top = landmarks_2d[self.left_eye_top]
        l_bottom = landmarks_2d[self.left_eye_bottom]
        l_iris = landmarks_2d[self.left_iris_center]

        l_w = max(1e-4, abs(l_inner[0] - l_outer[0]))
        l_h = max(1e-4, abs(l_bottom[1] - l_top[1]))

        # Normalized position in eyelid box [0.0 to 1.0]
        l_gx = (l_iris[0] - min(l_outer[0], l_inner[0])) / l_w
        l_gy = (l_iris[1] - l_top[1]) / l_h

        # --- Right Eye Measurements ---
        r_inner = landmarks_2d[self.right_eye_inner]
        r_outer = landmarks_2d[self.right_eye_outer]
        r_top = landmarks_2d[self.right_eye_top]
        r_bottom = landmarks_2d[self.right_eye_bottom]
        r_iris = landmarks_2d[self.right_iris_center]

        r_w = max(1e-4, abs(r_outer[0] - r_inner[0]))
        r_h = max(1e-4, abs(r_bottom[1] - r_top[1]))

        r_gx = (r_iris[0] - min(r_inner[0], r_outer[0])) / r_w
        r_gy = (r_iris[1] - r_top[1]) / r_h

        # Combine both eyes
        raw_gx = float((l_gx + r_gx) / 2.0)
        raw_gy = float((l_gy + r_gy) / 2.0)

        # Calibrated coordinates centered at (0.50, 0.50)
        calib_gx = float(np.clip(raw_gx - (self.neutral_gx - 0.50), 0.0, 1.0))
        calib_gy = float(np.clip(raw_gy - (self.neutral_gy - 0.50), 0.0, 1.0))

        self.gy_history.append(calib_gy)

        # -------------------------------------------------------------
        # SENSITIVE EYE-ROLLING / UPWARD LOOK DETECTION
        # When looking at camera/screen: calib_gy is ~ 0.42 - 0.58
        # When eyes roll up or look at ceiling: iris moves towards top lid,
        # causing calib_gy to drop below 0.35!
        # -------------------------------------------------------------
        is_eye_rolling = False
        if calib_gy < 0.34:
            is_eye_rolling = True
        elif len(self.gy_history) >= 4 and (np.mean(self.gy_history) < 0.36):
            is_eye_rolling = True

        # Gaze zone categorization
        if is_eye_rolling:
            gaze_zone = "EYE_ROLLING_UP"
        elif calib_gy > 0.65:
            gaze_zone = "LOOKING_DOWN"
        elif calib_gx < 0.36:
            gaze_zone = "LOOKING_LEFT"
        elif calib_gx > 0.64:
            gaze_zone = "LOOKING_RIGHT"
        else:
            gaze_zone = "CENTER"

        # Deviation penalties
        dev_x = abs(calib_gx - 0.50)
        dev_y = abs(calib_gy - 0.50)
        penalty_x = max(0.0, (dev_x - 0.08) / 0.22)
        penalty_y = max(0.0, (dev_y - 0.08) / 0.22)
        alignment = max(0.0, 1.0 - (penalty_x * 0.55 + penalty_y * 0.45))

        if is_eye_rolling:
            alignment = 0.10

        return {
            "gaze_x": calib_gx,
            "gaze_y": calib_gy,
            "raw_gy": raw_gy,
            "alignment": float(min(1.0, max(0.0, alignment))),
            "is_eye_rolling": bool(is_eye_rolling),
            "gaze_zone": gaze_zone,
        }

    def calibrate(self, current_raw_gx: float, current_raw_gy: float):
        """Calibrate current eye position as neutral forward center."""
        self.neutral_gx = current_raw_gx
        self.neutral_gy = current_raw_gy
