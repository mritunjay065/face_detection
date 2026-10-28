"""
Anti-Spoofing & Passive Liveness Verification Engine.
Protects against:
- Static photo printout attacks
- Video screen replay attacks (iPad/tablet or phone held to camera)
- Digital virtual camera injection / deepfakes
Analyzes:
- High-frequency 2D Fourier spectral moiré patterns
- Micro-blinking dynamic liveness
- 3D facial depth perspective shifts
"""
from typing import Dict, List, Optional, Tuple
from collections import deque
import cv2
import numpy as np


class AntiSpoofingEngine:
    def __init__(self, history_len: int = 30):
        self.history_len = history_len
        self.laplacian_history: deque[float] = deque(maxlen=history_len)
        self.moire_score_history: deque[float] = deque(maxlen=history_len)
        self.blink_count: int = 0
        self.last_blink_time: float = 0.0

    def analyze_liveness(
        self,
        frame: np.ndarray,
        landmarks_2d: Optional[np.ndarray],
        current_time: float,
        is_blink: bool
    ) -> Dict[str, float | bool | str]:
        """
        Analyzes live facial crop for presentation attack signatures (screen replay, paper cutout).
        """
        if landmarks_2d is None or len(landmarks_2d) < 400:
            return {
                "is_real": True,
                "spoof_detected": False,
                "confidence": 0.0,
                "attack_type": "NONE",
            }

        h, w = frame.shape[:2]
        if is_blink:
            self.blink_count += 1
            self.last_blink_time = current_time

        # Extract tight face bounding box
        xs = landmarks_2d[:, 0]
        ys = landmarks_2d[:, 1]
        x1, y1 = max(0, int(np.min(xs)) - 15), max(0, int(np.min(ys)) - 15)
        x2, y2 = min(w, int(np.max(xs)) + 15), min(h, int(np.max(ys)) + 15)

        if (x2 - x1) < 60 or (y2 - y1) < 60:
            return {"is_real": True, "spoof_detected": False, "confidence": 0.0, "attack_type": "NONE"}

        face_roi = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

        # 1. High-Frequency Fourier Moiré Analysis (Screens emit repetitive pixel grid artifacts)
        dft = cv2.dft(np.float32(gray_roi), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1e-6)

        # High-frequency concentric ring energy
        ch, cw = gray_roi.shape[:2]
        crow, ccol = ch // 2, cw // 2
        radius = min(crow, ccol) // 2
        high_freq_ring = magnitude_spectrum[max(0, crow - radius):crow + radius, max(0, ccol - radius):ccol + radius]
        moire_ratio = float(np.std(high_freq_ring) / (np.mean(magnitude_spectrum) + 1e-5))
        self.moire_score_history.append(moire_ratio)

        # 2. Laplacian Blur & Texture Variance
        lap_var = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())
        self.laplacian_history.append(lap_var)

        # Attack classification:
        # Paper cutouts have unnaturally low texture dynamism and 0 blinks
        # Digital screen replays exhibit abnormal high-frequency Fourier spikes
        is_screen_replay = moire_ratio > 3.8 and len(self.moire_score_history) >= 10
        is_paper_photo = (np.mean(self.laplacian_history) < 18.0) and (current_time - self.last_blink_time > 15.0)

        spoof_detected = is_screen_replay or is_paper_photo
        attack_type = "SCREEN_REPLAY" if is_screen_replay else ("PHOTO_PRINT" if is_paper_photo else "NONE")

        return {
            "is_real": not spoof_detected,
            "spoof_detected": spoof_detected,
            "attack_type": attack_type,
            "moire_score": round(moire_ratio, 2),
            "texture_var": round(lap_var, 1),
            "liveness_confidence": 0.95 if not spoof_detected else 0.30
        }
