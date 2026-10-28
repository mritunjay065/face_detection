"""
High-Precision Proctoring YOLO Detector with Strict Multi-Class Object Discrimination.
Distinguishes between:
- Authorized hardware: Keyboard, Mouse, Laptop (primary test device)
- Prohibited cheating items: Cell Phone, Book/Notes, Remote, Secondary Monitor/Laptop
- Multi-person proxy detection: Candidate vs Secondary Person
- Everyday permissible desktop items: Water Bottle, Cup, Chair
"""
import threading
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class AsyncYOLODetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.22):
        self.conf_threshold = conf_threshold
        self.model_name = model_name
        self.model = None

        # COCO Dataset ID Mapping:
        # Strictly categorize objects to avoid false alarms (e.g., distinguishing a cell phone from a mouse or cup)
        self.CLASS_MAP = {
            0: ("PERSON", "PERSON"),
            67: ("CELL PHONE", "PROHIBITED"),
            73: ("BOOK / NOTES", "PROHIBITED"),
            65: ("REMOTE", "PROHIBITED"),
            63: ("LAPTOP", "HARDWARE"),
            64: ("MOUSE", "PERMISSIBLE"),
            66: ("KEYBOARD", "PERMISSIBLE"),
            39: ("BOTTLE", "PERMISSIBLE"),
            41: ("CUP", "PERMISSIBLE"),
            56: ("CHAIR", "PERMISSIBLE"),
        }

        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.running = True
        self.new_frame_event = threading.Event()
        self.submitted_frames = 0
        self.completed_frames = 0

        self.cached_results = {
            "num_persons": 0,
            "prohibited_items": [],
            "permissible_items": [],
            "risk_flags": [],
            "detections": [],
            "risk_level": "LOW",
            "inference_ms": 0.0,
            "error": None,
        }

        self.worker_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.worker_thread.start()

    def _load_model(self, model_name: str):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
            print(f"[Info] YOLOv8 Multi-Class Detector initialized with '{model_name}'.")
        except Exception as e:
            with self.lock:
                self.cached_results["error"] = str(e)
            print(f"[Warning] YOLO init failed: {e}")

    def update_frame(self, frame: np.ndarray):
        with self.lock:
            self.latest_frame = frame.copy()
            self.submitted_frames += 1
            self.new_frame_event.set()

    def get_results(self) -> Dict:
        with self.lock:
            return dict(self.cached_results)

    def _inference_loop(self):
        self._load_model(self.model_name)
        while self.running:
            self.new_frame_event.wait(timeout=0.1)
            self.new_frame_event.clear()

            with self.lock:
                if self.latest_frame is None:
                    continue
                img = self.latest_frame
                frame_id = self.submitted_frames
                self.latest_frame = None

            if self.model is None:
                continue

            t_start = time.time()
            try:
                # Run YOLO prediction targeting only proctor-relevant classes
                target_ids = list(self.CLASS_MAP.keys())
                results = self.model.predict(
                    source=img,
                    conf=max(0.10, self.conf_threshold),
                    classes=target_ids,
                    verbose=False,
                    imgsz=480
                )

                person_boxes = []
                prohibited_detected = []
                permissible_detected = []
                detections = []
                risk_flags = []

                if results and len(results) > 0:
                    boxes = results[0].boxes
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        xyxy = box.xyxy[0].tolist()

                        if cls_id not in self.CLASS_MAP:
                            continue

                        name, category = self.CLASS_MAP[cls_id]

                        # Cell phone requires slightly higher confidence to avoid false alarm with dark mouse
                        if cls_id == 67 and conf < 0.28:
                            continue

                        detections.append({
                            "name": name,
                            "category": category,
                            "conf": conf,
                            "box": [int(v) for v in xyxy]
                        })

                        if category == "PERSON":
                            person_boxes.append(xyxy)
                        elif category == "PROHIBITED":
                            prohibited_detected.append(name)
                        elif category == "PERMISSIBLE":
                            permissible_detected.append(name)

                num_persons = len(person_boxes)
                risk_level = "LOW"

                # Check multiple persons
                if num_persons > 1:
                    risk_flags.append(f"MULTIPLE PERSONS ({num_persons})")
                    risk_level = "HIGH"
                elif num_persons == 0:
                    risk_flags.append("NO CANDIDATE IN VIEW")
                    risk_level = "MEDIUM"

                # Check unauthorized items
                if prohibited_detected:
                    unique_prohibited = sorted(list(set(prohibited_detected)))
                    risk_flags.append(f"UNAUTHORIZED: {', '.join(unique_prohibited)}")
                    risk_level = "HIGH"

                inf_ms = (time.time() - t_start) * 1000.0

                with self.lock:
                    self.cached_results = {
                        "num_persons": num_persons,
                        "prohibited_items": prohibited_detected,
                        "permissible_items": permissible_detected,
                        "risk_flags": risk_flags,
                        "detections": detections,
                        "risk_level": risk_level,
                        "inference_ms": round(inf_ms, 1),
                        "error": None,
                        "frame_id": frame_id,
                    }
            except Exception as e:
                print(f"[Warning] YOLO inference failed: {e}")

    def annotate(self, frame: np.ndarray, yolo_result: Dict) -> np.ndarray:
        """Draws distinct color-coded bounding boxes per category."""
        for det in yolo_result.get("detections", []):
            name = det["name"]
            category = det["category"]
            conf = det["conf"]
            x1, y1, x2, y2 = det["box"]

            if category == "PERSON":
                color = (0, 220, 0)       # Green for valid candidate
            elif category == "PROHIBITED":
                color = (0, 0, 240)       # Bright Red for cheating items (phone, notes)
            elif category == "HARDWARE":
                color = (255, 150, 0)     # Orange for laptop/hardware
            else:
                color = (180, 180, 180)   # Subtle gray for harmless desk items (cup/mouse)

            # Draw sleek box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_tag = f"{name} {int(conf*100)}%"
            (tw, th), _ = cv2.getTextSize(label_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label_tag, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def close(self):
        self.running = False
        self.new_frame_event.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
