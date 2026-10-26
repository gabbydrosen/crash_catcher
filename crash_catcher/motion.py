"""Turn IMU readings into one of four motion states.

  stopped       engine-idle vibration only, no acceleration or turning
  cruising      the default: moving, not turning, not braking hard
  turning       yaw rate (how fast the car is rotating) above a threshold
  hard_braking  strong forward deceleration

An IMU can't measure speed, and integrating acceleration drifts. So "stopped"
is detected from how much the acceleration jiggles (the vibration signature),
not from speed. The real-car validation drive checks this against GPS.

Everything here is plain math with no hardware, so it can be unit tested
and also run on simulator telemetry later.
"""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

STATES = ("stopped", "cruising", "turning", "hard_braking")


@dataclass
class VehicleSample:
    """One IMU sample already rotated into the car's frame."""
    t: float
    long_accel: float   # m/s^2, + = speeding up, - = braking
    yaw_rate: float     # deg/s, rotation around the vertical axis
    accel_mag: float    # m/s^2, magnitude of the raw acceleration (incl. gravity)


@dataclass
class MotionReading:
    state: str
    long_accel: float = 0.0     # smoothed
    yaw_rate: float = 0.0       # smoothed
    accel_std: float = math.nan


class MotionClassifier:
    def __init__(self, cfg):
        self.cfg = cfg
        self._window = deque()
        self._quiet_since = None
        self._brake_until = -math.inf
        self._turn_until = -math.inf
        self.reading = MotionReading("cruising")

    def update(self, s: VehicleSample) -> MotionReading:
        c = self.cfg
        self._window.append(s)
        while self._window and s.t - self._window[0].t > c.window_s:
            self._window.popleft()

        # Smooth over the last ~0.25 s to ignore bumps and vibration spikes.
        recent = [x for x in self._window if s.t - x.t <= 0.25]
        long_accel = float(np.mean([x.long_accel for x in recent]))
        yaw_rate = float(np.mean([x.yaw_rate for x in recent]))
        accel_std = float(np.std([x.accel_mag for x in self._window]))

        if long_accel <= -c.hard_brake_mps2:
            self._brake_until = s.t + c.brake_hold_s
        if abs(yaw_rate) >= c.turning_yaw_rate_dps:
            self._turn_until = s.t + c.turning_hold_s

        window_full = s.t - self._window[0].t >= 0.8 * c.window_s
        quiet = (window_full and accel_std < c.stopped_accel_std
                 and abs(yaw_rate) < c.turning_yaw_rate_dps / 2
                 and abs(long_accel) < 0.5)
        if quiet:
            if self._quiet_since is None:
                self._quiet_since = s.t
        else:
            self._quiet_since = None

        # Most dangerous state wins.
        if s.t < self._brake_until:
            state = "hard_braking"
        elif s.t < self._turn_until:
            state = "turning"
        elif self._quiet_since is not None and s.t - self._quiet_since >= c.stopped_min_duration_s:
            state = "stopped"
        elif self.reading.state == "stopped" and quiet:
            state = "stopped"  # stay stopped until something actually changes
        else:
            state = "cruising"

        self.reading = MotionReading(state, long_accel, yaw_rate, accel_std)
        return self.reading
