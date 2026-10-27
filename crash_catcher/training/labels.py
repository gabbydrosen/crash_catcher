"""Load recorded sessions and make the "right answers" (labels).

An onset is the moment the eyes leave the road: a frame where off_road
switches from 0 to 1. Very short off-road stretches (under 0.3 s) are usually
tracking flicker, so they are not counted as onsets.

Each frame then gets a label: 1 if an onset happens within the next
`horizon` seconds, otherwise 0. Frames where the driver is already off the
road are left out (the question is about the *start* of a glance), and so
are the last `horizon` seconds of a session (we can't see what came next).
"""

import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MIN_GLANCE_S = 0.3


@dataclass
class Session:
    path: str
    participant: str
    session_type: str
    frames: pd.DataFrame
    meta: dict = field(default_factory=dict)

    @property
    def name(self):
        return os.path.basename(os.path.normpath(self.path))


def load_session(path: str) -> Session:
    frames = pd.read_csv(os.path.join(path, "frames.csv"))
    meta = {}
    meta_path = os.path.join(path, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    # Folder names look like P03_main_20261027-193000, used if meta.json is missing.
    parts = os.path.basename(os.path.normpath(path)).split("_")
    participant = str(meta.get("participant", parts[0]))
    session_type = str(meta.get("session", parts[1] if len(parts) > 2 else "unknown"))
    frames = frames.sort_values("t").reset_index(drop=True)
    return Session(path, participant, session_type, frames, meta)


def find_sessions(root: str, session_types=("main", "pilot")) -> list[Session]:
    """Every session folder under root whose type is in session_types
    (pass None to take them all)."""
    sessions = []
    for csv_path in sorted(glob.glob(os.path.join(root, "*", "frames.csv"))):
        s = load_session(os.path.dirname(csv_path))
        if session_types is None or s.session_type in session_types:
            sessions.append(s)
    return sessions


def off_road_runs(frames: pd.DataFrame):
    """List of (start_time, end_time) for each stretch of off_road frames.
    end_time is the first frame back on the road (or the last frame)."""
    t = frames["t"].to_numpy(float)
    off = frames["off_road"].fillna(0).to_numpy().astype(bool)
    runs = []
    i, n = 0, len(off)
    while i < n:
        if off[i]:
            j = i
            while j < n and off[j]:
                j += 1
            runs.append((t[i], t[j] if j < n else t[n - 1]))
            i = j
        else:
            i += 1
    return runs


def find_onsets(frames: pd.DataFrame, min_glance_s: float = MIN_GLANCE_S) -> np.ndarray:
    """Times (seconds) when a real glance away started."""
    runs = off_road_runs(frames)
    # A run that starts on the very first frame has no visible start.
    first_t = frames["t"].iloc[0] if len(frames) else 0.0
    return np.array([s for s, e in runs if e - s >= min_glance_s and s > first_t])


def make_labels(frames: pd.DataFrame, horizon: float = 1.0,
                min_glance_s: float = MIN_GLANCE_S):
    """Returns (labels, mask) as numpy arrays, one entry per frame.
    Only use frames where mask is True for training and frame-level scoring."""
    t = frames["t"].to_numpy(float)
    onsets = find_onsets(frames, min_glance_s)
    # Index of the first onset strictly after each frame time.
    nxt = np.searchsorted(onsets, t, side="right")
    labels = np.zeros(len(t), dtype=int)
    has_next = nxt < len(onsets)
    labels[has_next] = (onsets[nxt[has_next]] - t[has_next] <= horizon).astype(int)

    on_road = ~frames["off_road"].fillna(0).to_numpy().astype(bool)
    future_known = t <= (t[-1] - horizon) if len(t) else np.zeros(0, bool)
    return labels, on_road & future_known
