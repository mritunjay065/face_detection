"""
Audit Video Recorder for Proctoring Violations.
Automatically records a rolling 10-second video buffer in memory.
When a severe violation occurs (COLLUSION, UNAUTHORIZED OBJECT, AUDIO TALKING, SPOOFING):
Instantly dumps an auditable incident video clip to disk with timestamped evidence.
"""
from typing import List, Optional
import os
import time
from collections import deque
import cv2
import numpy as np


class ProctoringAuditRecorder:
    def __init__(self, output_dir: str = "data/audit_clips", buffer_seconds: int = 10, fps: int = 15):
        self.output_dir = output_dir
        self.buffer_size = buffer_seconds * fps
        self.fps = fps
        self.frame_buffer: deque[np.ndarray] = deque(maxlen=self.buffer_size)
        self.last_saved_time: float = 0.0
        self.cooldown_sec: float = 12.0  # Prevent spamming clip saves

        os.makedirs(self.output_dir, exist_ok=True)

    def record_frame(self, frame: np.ndarray):
        """Append frame to circular RAM buffer."""
        self.frame_buffer.append(frame.copy())

    def trigger_incident_clip(self, violation_label: str) -> Optional[str]:
        """
        Saves the last 10 seconds of video as an auditable MP4/AVI incident clip.
        """
        now = time.time()
        if (now - self.last_saved_time) < self.cooldown_sec:
            return None

        if len(self.frame_buffer) < 15:
            return None

        self.last_saved_time = now
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        clean_tag = violation_label.replace(" ", "_").replace(":", "").replace("/", "_")
        filename = f"INCIDENT_{timestamp_str}_{clean_tag}.avi"
        filepath = os.path.join(self.output_dir, filename)

        h, w = self.frame_buffer[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(filepath, fourcc, self.fps, (w, h))

        for f in list(self.frame_buffer):
            out.write(f)
        out.release()

        print(f"\n[AUDIT ALERT] Recorded evidence clip saved to: {filepath}")
        return filepath
