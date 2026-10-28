"""
Webcam video capture utility with timestamp tracking and frame rate regulation.
"""
from typing import Generator, Optional, Tuple
import time
import cv2
import numpy as np


class VideoCaptureManager:
    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480, target_fps: int = 30):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_interval = 1.0 / target_fps
        self.last_frame_time = 0.0

    def start(self):
        # Open video capture with DirectShow on Windows for fast startup
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Reads a frame and returns (success, bgr_image, timestamp_seconds).
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None, 0.0

        now = time.time()
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None, now

        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))

        return True, frame, now

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
