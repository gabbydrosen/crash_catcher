"""Decide, frame by frame, whether the driver's eyes are on the road.

1. Calibration: the driver looks at the road center for a few seconds.
   We record their typical head angle and iris position. This is their
   personal "road zone", and it also cancels out wherever the camera is mounted.
2. Every frame after that, the driver is "off road" if their head or eyes
   are outside that zone, their eyes are closed longer than a blink, or
   their face can't be found.
3. We time how long each off-road stretch has lasted. The alert policy
   compares that time with the limit for the current motion state.
"""

import math
from dataclasses import dataclass, asdict

import numpy as np

from .face_tracker import FaceReading


@dataclass
class RoadZone:
    yaw: float
    pitch: float
    gaze_x: float
    gaze_y: float

    def to_dict(self):
        return asdict(self)


class Calibrator:
    """Collects readings while the driver looks at the road center."""

    def __init__(self):
        self._samples = []

    def add(self, r: FaceReading):
        if r.found and not math.isnan(r.head_yaw):
            self._samples.append((r.head_yaw, r.head_pitch, r.gaze_x, r.gaze_y))

    @property
    def count(self):
        return len(self._samples)

    def finish(self) -> RoadZone:
        if len(self._samples) < 10:
            raise RuntimeError(
                f"Calibration only saw the face in {len(self._samples)} frames. "
                "Check the camera angle and lighting, then try again.")
        # The median ignores the odd bad frame better than the mean does.
        med = np.nanmedian(np.array(self._samples), axis=0)
        return RoadZone(*map(float, med))


@dataclass
class AttentionState:
    off_road: bool
    reason: str               # "", "head", "gaze", "eyes_closed", "no_face"
    off_road_duration: float  # seconds the current off-road stretch has lasted
    head_yaw_rel: float = math.nan    # head angle relative to the road zone
    head_pitch_rel: float = math.nan
    gaze_x_rel: float = math.nan
    gaze_y_rel: float = math.nan


class AttentionTracker:
    def __init__(self, zone: RoadZone, cfg):
        self.zone = zone
        self.cfg = cfg
        self._off_since = None
        self._closed_since = None
        self._missing_since = None

    def update(self, r: FaceReading, t: float) -> AttentionState:
        c = self.cfg
        rel = (math.nan,) * 4
        reason = ""

        if not r.found:
            if self._missing_since is None:
                self._missing_since = t
            self._closed_since = None
            # A face that drops out for a frame or two is just tracking
            # noise; one that stays gone means the head turned away.
            if t - self._missing_since >= c.no_face_grace_s:
                reason = "no_face"
        else:
            self._missing_since = None
            rel = (r.head_yaw - self.zone.yaw, r.head_pitch - self.zone.pitch,
                   r.gaze_x - self.zone.gaze_x, r.gaze_y - self.zone.gaze_y)
            yaw_rel, pitch_rel, gx_rel, gy_rel = rel

            blink = np.nanmean([r.blink_left, r.blink_right])
            if blink > c.blink_threshold:
                if self._closed_since is None:
                    self._closed_since = t
            else:
                self._closed_since = None

            if abs(yaw_rel) > c.head_yaw_margin_deg or abs(pitch_rel) > c.head_pitch_margin_deg:
                reason = "head"
            elif self._closed_since is not None and t - self._closed_since >= c.blink_grace_s:
                reason = "eyes_closed"
            # Iris position is unreliable while the eyes are half shut.
            elif self._closed_since is None and (
                    abs(gx_rel) > c.gaze_x_margin or abs(gy_rel) > c.gaze_y_margin):
                reason = "gaze"

        off = reason != ""
        if off:
            if self._off_since is None:
                # Back-date to when the face first went missing / eyes first
                # closed, so the grace periods count toward the duration.
                start = t
                if reason == "no_face":
                    start = self._missing_since
                elif reason == "eyes_closed":
                    start = self._closed_since
                self._off_since = start
            duration = t - self._off_since
        else:
            self._off_since = None
            duration = 0.0

        return AttentionState(off, reason, duration, *rel)
