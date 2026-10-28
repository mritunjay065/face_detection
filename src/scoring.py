"""
Rule-based scoring engine for visual engagement estimation.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class AttentionResult:
    score: Optional[float]
    state: str
    confidence: float
    gaze_alignment: float = 0.0
    head_alignment: float = 0.0
    on_target_ratio: float = 0.0


def estimate_engagement(
    gaze_alignment: float,
    head_alignment: float,
    on_target_ratio: float,
    signal_quality: float,
    face_visible: bool = True,
    eyes_closed: bool = False,
    closure_duration: float = 0.0,
    max_blink_duration: float = 0.4,
) -> AttentionResult:
    """
    Inputs are normalized to [0, 1].
    Estimates visual engagement and current behavioral state.
    """
    if not face_visible or signal_quality < 0.35:
        return AttentionResult(
            score=None,
            state="FACE_NOT_VISIBLE" if not face_visible else "UNKNOWN",
            confidence=0.0,
            gaze_alignment=0.0,
            head_alignment=0.0,
            on_target_ratio=0.0,
        )

    if eyes_closed and closure_duration > max_blink_duration:
        return AttentionResult(
            score=max(0.0, round(10.0 * signal_quality, 1)),
            state="EYES_CLOSED",
            confidence=round(signal_quality, 2),
            gaze_alignment=0.0,
            head_alignment=head_alignment,
            on_target_ratio=0.0,
        )

    g = max(0.0, min(1.0, gaze_alignment))
    h = max(0.0, min(1.0, head_alignment))
    d = max(0.0, min(1.0, on_target_ratio))
    q = max(0.0, min(1.0, signal_quality))

    # Weight distribution: Gaze (50%), Head Pose (20%), Temporal Target Ratio (30%)
    raw_score = 100.0 * (0.50 * g + 0.20 * h + 0.30 * d)

    # Behavior state classification
    if d >= 0.80 and g >= 0.65:
        state = "ON_TARGET"
    elif d >= 0.50 or g >= 0.45:
        state = "MIXED_GAZE"
    elif h < 0.40 and g < 0.40:
        state = "SUSTAINED_LOOK_AWAY"
    else:
        state = "BRIEF_GLANCE_AWAY"

    return AttentionResult(
        score=round(raw_score, 1),
        state=state,
        confidence=round(q, 2),
        gaze_alignment=round(g, 2),
        head_alignment=round(h, 2),
        on_target_ratio=round(d, 2),
    )


class ScoreSmoother:
    """Exponential Moving Average (EMA) smoother for continuous attention scores."""
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.smoothed_score: Optional[float] = None

    def update(self, new_score: Optional[float]) -> Optional[float]:
        if new_score is None:
            return self.smoothed_score
        if self.smoothed_score is None:
            self.smoothed_score = new_score
        else:
            self.smoothed_score = self.alpha * new_score + (1.0 - self.alpha) * self.smoothed_score
        return round(self.smoothed_score, 1)

    def reset(self):
        self.smoothed_score = None
