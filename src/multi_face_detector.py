"""
Professional Face Counter & Multi-Person Detector for Online Assessment (OA).
Uses a high-recall dual-detection architecture:
1. MediaPipe Tasks FaceLandmarker (High-fidelity primary face tracking)
2. OpenCV Multi-Scale Face Detector (Ultra-fast multi-face counting across frame)
Immediately triggers HIGH RISK if >= 2 faces or people are detected.
"""
from typing import Dict, List, Optional, Tuple
import os
import cv2
import numpy as np


class DualFaceDetector:
    def __init__(self, model_asset_path: str = "assets/face_landmarker.task"):
        self.landmarker = None
        self.mp = None
        self.haar_cascade = None
        self._last_timestamp_ms = -1

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(cascade_path):
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)

        if os.path.exists(model_asset_path):
            try:
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                self.mp = mp
                base_options = python.BaseOptions(model_asset_path=model_asset_path)
                # Allow tracking up to 4 faces to detect collusion / multiple people
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.VIDEO,
                    num_faces=4,
                    min_face_detection_confidence=0.35,
                    min_face_presence_confidence=0.35,
                    min_tracking_confidence=0.35
                )
                self.landmarker = vision.FaceLandmarker.create_from_options(options)
            except Exception as e:
                print(f"[Warning] DualFaceDetector landmarker init error: {e}")

    def detect_all_faces(self, bgr_frame: np.ndarray) -> Dict:
        """
        Detects primary candidate landmarks AND counts all faces present in the frame.
        Returns:
            primary_landmarks: (478, 2) coords of primary candidate or None
            face_count: total number of distinct human faces visible
            face_boxes: bounding boxes of all detected faces [(x, y, w, h)]
            is_multiple_faces: boolean flag (True if >= 2 faces)
        """
        h, w = bgr_frame.shape[:2]
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        primary_landmarks = None
        face_boxes = []

        # 1. MediaPipe Multi-Face Detection
        mp_count = 0
        if self.landmarker is not None:
            try:
                mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = max(self._last_timestamp_ms + 1, int(cv2.getTickCount() * 1000 / cv2.getTickFrequency()))
                self._last_timestamp_ms = timestamp_ms
                res = self.landmarker.detect_for_video(mp_image, timestamp_ms)
                if res.face_landmarks:
                    mp_count = len(res.face_landmarks)
                    # Use the largest detected face as the candidate regardless of model result order.
                    primary_index = int(np.argmax([
                        (max(lm.x for lm in face_lm) - min(lm.x for lm in face_lm)) *
                        (max(lm.y for lm in face_lm) - min(lm.y for lm in face_lm))
                        for face_lm in res.face_landmarks
                    ]))
                    raw_pts = res.face_landmarks[primary_index]
                    primary_landmarks = np.array([[lm.x * w, lm.y * h] for lm in raw_pts], dtype=np.float32)

                    for face_lm in res.face_landmarks:
                        xs = [lm.x * w for lm in face_lm]
                        ys = [lm.y * h for lm in face_lm]
                        fx, fy = int(min(xs)), int(min(ys))
                        fw, fh = int(max(xs) - fx), int(max(ys) - fy)
                        face_boxes.append((fx, fy, fw, fh))
            except Exception as exc:
                print(f"[Warning] Face landmark inference failed: {exc}")

        # 2. Fast Haar Cascade Cross-Verification for peripheral / background faces
        if self.haar_cascade is not None:
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            haar_faces = self.haar_cascade.detectMultiScale(
                gray,
                scaleFactor=1.18,
                minNeighbors=5,
                minSize=(45, 45)
            )
            # Combine non-overlapping boxes
            for (hx, hy, hw, hh) in haar_faces:
                overlapping = False
                for (fx, fy, fw, fh) in face_boxes:
                    # Check overlap / IOU
                    ix1, iy1 = max(hx, fx), max(hy, fy)
                    ix2, iy2 = min(hx + hw, fx + fw), min(hy + hh, fy + fh)
                    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    union = hw * hh + fw * fh - intersection
                    if union > 0 and intersection / union > 0.25:
                        overlapping = True
                        break
                if not overlapping:
                    face_boxes.append((hx, hy, hw, hh))

        total_faces = max(mp_count, len(face_boxes))

        return {
            "primary_landmarks": primary_landmarks,
            "face_count": total_faces,
            "face_boxes": face_boxes,
            "is_multiple_faces": total_faces >= 2,
            "is_candidate_present": total_faces >= 1,
        }

    def annotate_faces(self, frame: np.ndarray, face_info: Dict) -> np.ndarray:
        """Highlights detected faces with professional security bounding boxes."""
        boxes = face_info.get("face_boxes", [])
        is_multi = face_info.get("is_multiple_faces", False)

        for i, (x, y, w, h) in enumerate(boxes):
            if is_multi and i > 0:
                color = (0, 0, 255)  # Red for unauthorized second/third face
                tag = f"UNAUTHORIZED PERSON #{i+1}"
            else:
                color = (0, 220, 0) if not is_multi else (0, 0, 255)
                tag = "CANDIDATE" if not is_multi else "COLLUSION DETECTED"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x, max(0, y - th - 6)), (x + tw + 4, y), color, -1)
            cv2.putText(frame, tag, (x + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        return frame
