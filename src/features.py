"""
Temporal feature aggregation engine.
Maintains a rolling window (e.g. 5 seconds) to calculate sustained duration,
gaze shift frequency, blink rate, and on-target ratio.
"""
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass
class FrameFeature:
    timestamp: float
    face_visible: bool
    gaze_alignment: float
    gaze_x: float
    gaze_y: float
    head_alignment: float
    head_yaw: float
    head_pitch: float
    head_roll: float
    ear: float
    is_closed: bool
    is_blink: bool
    signal_quality: float


class TemporalFeatureEngine:
    def __init__(self, window_seconds: float = 5.0):
        self.window_seconds = window_seconds
        self.history: deque[FrameFeature] = deque()

    def add_frame(self, feat: FrameFeature):
        self.history.append(feat)
        # Prune frames older than window_seconds
        cutoff = feat.timestamp - self.window_seconds
        while self.history and self.history[0].timestamp < cutoff:
            self.history.popleft()

    def aggregate(self) -> Dict[str, float]:
        if not self.history:
            return {
                "face_visibility_ratio": 0.0,
                "on_target_ratio": 0.0,
                "longest_off_target_sec": 0.0,
                "gaze_shift_freq": 0.0,
                "downward_gaze_ratio": 0.0,
                "mean_head_yaw": 0.0,
                "mean_head_pitch": 0.0,
                "head_movement_std": 0.0,
                "blink_rate_bpm": 0.0,
                "eye_closure_ratio": 0.0,
                "signal_confidence": 0.0,
            }

        total_frames = len(self.history)
        visible_frames = [f for f in self.history if f.face_visible]
        face_visibility_ratio = len(visible_frames) / total_frames

        if not visible_frames:
            return {
                "face_visibility_ratio": 0.0,
                "on_target_ratio": 0.0,
                "longest_off_target_sec": self.window_seconds,
                "gaze_shift_freq": 0.0,
                "downward_gaze_ratio": 0.0,
                "mean_head_yaw": 0.0,
                "mean_head_pitch": 0.0,
                "head_movement_std": 0.0,
                "blink_rate_bpm": 0.0,
                "eye_closure_ratio": 1.0,
                "signal_confidence": 0.0,
            }

        # Valid frames metrics
        on_target_count = 0
        downward_count = 0
        closed_count = 0
        blink_count = 0

        yaws = []
        pitches = []
        gaze_xs = []
        gaze_ys = []
        qualities = []

        off_target_intervals: List[float] = []
        current_off_target_start: Optional[float] = None

        for f in visible_frames:
            # On-target criteria: both gaze and head within acceptable bounds
            is_on_target = (f.gaze_alignment >= 0.55) and (f.head_alignment >= 0.50) and not f.is_closed
            if is_on_target:
                on_target_count += 1
                if current_off_target_start is not None:
                    off_target_intervals.append(f.timestamp - current_off_target_start)
                    current_off_target_start = None
            else:
                if current_off_target_start is None:
                    current_off_target_start = f.timestamp

            if f.gaze_y > 0.65 or f.head_pitch < -12.0:
                downward_count += 1

            if f.is_closed:
                closed_count += 1
            if f.is_blink:
                blink_count += 1

            yaws.append(f.head_yaw)
            pitches.append(f.head_pitch)
            gaze_xs.append(f.gaze_x)
            gaze_ys.append(f.gaze_y)
            qualities.append(f.signal_quality)

        if current_off_target_start is not None and visible_frames:
            off_target_intervals.append(visible_frames[-1].timestamp - current_off_target_start)

        n_vis = len(visible_frames)
        duration_window = max(0.5, self.history[-1].timestamp - self.history[0].timestamp)

        # Gaze shift frequency (standard deviation or derivative changes)
        diff_gaze = np.diff(gaze_xs) if len(gaze_xs) > 1 else np.array([0.0])
        gaze_shifts = int(np.sum(np.abs(diff_gaze) > 0.08))
        gaze_shift_freq = float(gaze_shifts / duration_window)

        # Blink rate per minute
        blink_rate_bpm = float((blink_count / duration_window) * 60.0)

        longest_off_target = float(max(off_target_intervals)) if off_target_intervals else 0.0

        return {
            "face_visibility_ratio": float(face_visibility_ratio),
            "on_target_ratio": float(on_target_count / n_vis),
            "longest_off_target_sec": longest_off_target,
            "gaze_shift_freq": gaze_shift_freq,
            "downward_gaze_ratio": float(downward_count / n_vis),
            "mean_head_yaw": float(np.mean(yaws)),
            "mean_head_pitch": float(np.mean(pitches)),
            "head_movement_std": float(np.std(yaws) + np.std(pitches)),
            "blink_rate_bpm": blink_rate_bpm,
            "eye_closure_ratio": float(closed_count / n_vis),
            "signal_confidence": float(np.mean(qualities) * face_visibility_ratio),
        }
