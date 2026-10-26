"""Session logging. Every frame becomes one row of frames.csv.

These files are the raw material for training your own model later, and for
scoring every alert policy offline on the same recordings. Log everything,
decide what to use later.
"""

import csv
import json
import math
import os
import time
from dataclasses import asdict, is_dataclass

COLUMNS = [
    "t", "frame", "face_found",
    "head_yaw", "head_pitch", "head_roll", "gaze_x", "gaze_y",
    "blink_left", "blink_right",
    "head_yaw_rel", "head_pitch_rel", "gaze_x_rel", "gaze_y_rel",
    "off_road", "off_reason", "off_duration",
    "motion_state", "long_accel", "yaw_rate", "accel_std",
    "alert_fixed", "alert_context", "alert_live",
    "event",
]


def _fmt(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.4f}"
    return v


class SessionLogger:
    def __init__(self, root: str, participant: str, session_type: str):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = os.path.join(root, f"{participant}_{session_type}_{stamp}")
        os.makedirs(self.dir, exist_ok=True)
        self._f = open(os.path.join(self.dir, "frames.csv"), "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=COLUMNS)
        self._w.writeheader()
        self._rows = 0

    def write_meta(self, **meta):
        def convert(o):
            if is_dataclass(o):
                return asdict(o)
            if hasattr(o, "to_dict"):
                return o.to_dict()
            return str(o)
        with open(os.path.join(self.dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=convert)

    def log(self, **row):
        self._w.writerow({k: _fmt(row.get(k, "")) for k in COLUMNS})
        self._rows += 1
        if self._rows % 30 == 0:  # don't lose much if the Pi loses power
            self._f.flush()

    def close(self):
        self._f.close()
