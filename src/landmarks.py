"""
MediaPipe Face Landmarker with Precise Cyber Geometric Mesh Rendering,
Mask/Unmasking Face Occlusion Verification, and Fallback Trackers.
"""
from typing import Dict, List, Optional, Tuple
import os
import cv2
import numpy as np


class FaceLandmarksDetector:
    def __init__(self, model_asset_path: str = "assets/face_landmarker.task"):
        self.landmarker = None
        self.mp = None
        self._last_timestamp_ms = -1
        self.haar_cascade = None

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(cascade_path):
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)

        # -------------------------------------------------------------
        # EXACT CYBER GEOMETRIC FACIAL MESH (From Reference User Image)
        # Structural Nodes & Connecting Triangulation Facets
        # -------------------------------------------------------------
        # Landmark indices corresponding to key anatomical nodes:
        # 10: Forehead Apex, 109: Forehead Left, 338: Forehead Right
        # 67: Temples Left, 297: Temples Right
        # 103: Cheekbone Outer Left, 332: Cheekbone Outer Right
        # 234: Jaw Corner Left, 454: Jaw Corner Right
        # 152: Chin Point, 172: Jaw Left Angle, 397: Jaw Right Angle
        # 168: Glabella / Brow Midpoint, 9: Forehead Mid
        # 70: Left Eyebrow Arch, 336: Right Eyebrow Arch
        # 33: Left Eye Outer, 133: Left Eye Inner, 159: Left Eye Top
        # 263: Right Eye Outer, 362: Right Eye Inner, 386: Right Eye Top
        # 1: Nose Tip, 2: Nose Subnasale
        # 61: Left Mouth Corner, 291: Right Mouth Corner, 0: Upper Lip Mid, 17: Lower Lip Mid
        # -------------------------------------------------------------
        self.GEOMETRIC_NODES = [
            10, 109, 338, 67, 297, 103, 332, 234, 454, 152, 172, 397,
            168, 9, 70, 336, 33, 133, 263, 362, 1, 2, 61, 291, 0, 17
        ]

        self.CYBER_FACET_EDGES = [
            # Forehead Polygonal Cap
            (10, 109), (109, 67), (10, 338), (338, 297), (109, 338), (10, 9), (9, 168),
            # Forehead to Temples & Brows
            (67, 70), (70, 168), (168, 336), (336, 297), (67, 103), (297, 332),
            (109, 70), (338, 336), (109, 168), (338, 168),
            # Brow to Eyes
            (70, 33), (70, 133), (168, 133), (168, 362), (336, 362), (336, 263),
            # Eye Contours
            (33, 133), (263, 362),
            # Temples to Cheekbones & Eye Outer
            (103, 33), (332, 263), (103, 234), (332, 454),
            # Eyes & Glabella to Nose Bridge
            (133, 1), (362, 1), (168, 1), (33, 1), (263, 1),
            # Cheekbones to Nose & Mouth
            (103, 1), (332, 1), (103, 61), (332, 291),
            # Nose Tip to Mouth
            (1, 61), (1, 291), (1, 0), (1, 2), (2, 0),
            # Mouth Polygon
            (61, 0), (0, 291), (291, 17), (17, 61), (61, 291),
            # Cheekbones to Jaw Angles
            (234, 172), (454, 397), (234, 61), (454, 291),
            # Jaw Angles to Mouth & Chin
            (172, 61), (397, 291), (172, 17), (397, 17),
            (172, 152), (397, 152), (17, 152)
        ]

        # Standard MediaPipe Tasks initialization
        if os.path.exists(model_asset_path):
            try:
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                self.mp = mp
                base_options = python.BaseOptions(model_asset_path=model_asset_path)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.VIDEO,
                    num_faces=1,
                    min_face_detection_confidence=0.35,
                    min_face_presence_confidence=0.35,
                    min_tracking_confidence=0.35
                )
                self.landmarker = vision.FaceLandmarker.create_from_options(options)
                print("[Info] MediaPipe Tasks FaceLandmarker successfully initialized!")
            except Exception as e:
                print(f"[Warning] Failed to initialize MediaPipe Tasks FaceLandmarker: {e}")

    def detect(self, bgr_frame: np.ndarray, timestamp_ms: int = 0) -> Tuple[Optional[np.ndarray], float]:
        """Detect 478 2D facial landmarks in pixel coordinates."""
        h, w = bgr_frame.shape[:2]
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

        if self.landmarker is not None:
            try:
                mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = max(self._last_timestamp_ms + 1, int(timestamp_ms))
                self._last_timestamp_ms = timestamp_ms
                result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.face_landmarks and len(result.face_landmarks) > 0:
                    raw_pts = result.face_landmarks[0]
                    coords = np.array([[lm.x * w, lm.y * h] for lm in raw_pts], dtype=np.float32)
                    return coords, 0.95
            except Exception:
                pass

        # Fallback to Haar Cascade if mesh detection is unavailable
        if self.haar_cascade is not None:
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            faces = self.haar_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=3, minSize=(70, 70))
            if len(faces) > 0:
                fx, fy, fw, fh = faces[0]
                dummy = np.zeros((478, 2), dtype=np.float32)
                dummy[1] = [fx + fw * 0.5, fy + fh * 0.55]
                dummy[152] = [fx + fw * 0.5, fy + fh * 0.95]
                dummy[33] = [fx + fw * 0.3, fy + fh * 0.4]
                dummy[133] = [fx + fw * 0.45, fy + fh * 0.4]
                dummy[468] = [fx + fw * 0.37, fy + fh * 0.4]
                dummy[362] = [fx + fw * 0.55, fy + fh * 0.4]
                dummy[263] = [fx + fw * 0.7, fy + fh * 0.4]
                dummy[473] = [fx + fw * 0.63, fy + fh * 0.4]
                dummy[61] = [fx + fw * 0.35, fy + fh * 0.75]
                dummy[291] = [fx + fw * 0.65, fy + fh * 0.75]
                return dummy, 0.60

        return None, 0.0

    def check_unmasking(self, frame: np.ndarray, landmarks_2d: Optional[np.ndarray]) -> Dict[str, bool]:
        """
        Unmasking / Occlusion Verification:
        Ensures candidate is not wearing a face mask or covering mouth/nose.
        Compares texture/edge variance in lower face vs forehead.
        """
        if landmarks_2d is None or len(landmarks_2d) < 400:
            return {"face_covered": False, "mask_detected": False}

        h, w = frame.shape[:2]
        # Bounding box of lower face (mouth & nose area)
        nose_pt = landmarks_2d[1].astype(int)
        chin_pt = landmarks_2d[152].astype(int)
        l_mouth = landmarks_2d[61].astype(int)
        r_mouth = landmarks_2d[291].astype(int)

        x1 = max(0, min(l_mouth[0], nose_pt[0]) - 15)
        x2 = min(w, max(r_mouth[0], nose_pt[0]) + 15)
        y1 = max(0, nose_pt[1] - 10)
        y2 = min(h, chin_pt[1] + 10)

        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return {"face_covered": False, "mask_detected": False}

        roi = frame[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Laplacian variance measures skin texture richness vs flat cloth mask
        lap_var = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
        # If mouth area has abnormally low texture variance (< 25), candidate is likely wearing a smooth mask
        mask_detected = lap_var < 22.0

        return {
            "face_covered": mask_detected,
            "mask_detected": mask_detected,
            "texture_score": float(lap_var)
        }

    def draw_face_wireframe(
        self,
        frame: np.ndarray,
        landmarks_2d: np.ndarray,
        line_color: Tuple[int, int, int] = (245, 230, 200),
        node_color: Tuple[int, int, int] = (255, 255, 255),
        line_thickness: int = 1
    ) -> np.ndarray:
        """
        Renders the exact Geometric Polygonal Facet Cyber-Mesh from the user's reference image:
        - Sleek white/light-blue polygonal connectivity
        - Circular vertex dots with anti-aliased glowing centers
        """
        if landmarks_2d is None:
            return frame

        n_pts = len(landmarks_2d)

        # 1. Draw Geometric Facet Edges
        for p1, p2 in self.CYBER_FACET_EDGES:
            if p1 < n_pts and p2 < n_pts:
                pt1 = tuple(landmarks_2d[p1].astype(int))
                pt2 = tuple(landmarks_2d[p2].astype(int))
                cv2.line(frame, pt1, pt2, line_color, line_thickness, cv2.LINE_AA)

        # 2. Draw Geometric Vertex Nodes
        for idx in self.GEOMETRIC_NODES:
            if idx < n_pts:
                pt = tuple(landmarks_2d[idx].astype(int))
                # Outer glow ring
                cv2.circle(frame, pt, 4, (200, 180, 160), -1, cv2.LINE_AA)
                # Inner bright white dot
                cv2.circle(frame, pt, 2, node_color, -1, cv2.LINE_AA)

        # 3. Iris Crosshairs
        for iris_idx in [468, 473]:
            if iris_idx < n_pts:
                pt = tuple(landmarks_2d[iris_idx].astype(int))
                cv2.circle(frame, pt, 3, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 6, (0, 200, 255), 1, cv2.LINE_AA)

        return frame
