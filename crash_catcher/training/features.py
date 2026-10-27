"""Features: numbers that describe the last 1-2 seconds of a driver's behavior.

Every feature for a frame is computed ONLY from that frame and earlier ones,
never from later ones. Live, the Pi only knows the past, so if training used
the future the model would look great in testing and fail in the car.

The same function runs in training (a whole recording at once) and live
(call it on a buffer of recent rows and take the last row).
"""

import numpy as np
import pandas as pd

from ..motion import STATES

SIGNALS = ["head_yaw_rel", "head_pitch_rel", "gaze_x_rel", "gaze_y_rel"]
PHONE_COLUMNS = ["phone_visible", "phone_area", "phone_area_rate",
                 "phone_brightness", "phone_near_face", "phone_in_use"]


def _numeric(df, col):
    if col not in df:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _velocity(t: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
    """Change per second, comparing each frame with the frame about dt
    seconds earlier (a single frame-to-frame difference is too noisy)."""
    j = np.searchsorted(t, t - dt, side="right") - 1
    j = np.clip(j, 0, None)
    span = t - t[j]
    with np.errstate(invalid="ignore", divide="ignore"):
        vel = (v - v[j]) / span
    vel[span <= 0] = np.nan
    return vel


def compute_features(frames: pd.DataFrame, window_s: float = 1.5,
                     vel_dt: float = 0.2) -> pd.DataFrame:
    """One row of features per input row, same order. Missing values become 0;
    face_missing_frac tells the model when the face data was unreliable."""
    t = frames["t"].to_numpy(float)
    idx = pd.to_timedelta(t, unit="s")
    win = f"{int(round(window_s * 1000))}ms"
    out = {}

    def rolling(series):
        return pd.Series(np.asarray(series, float), index=idx).rolling(win, min_periods=1)

    for col in SIGNALS:
        v = _numeric(frames, col).to_numpy(float)
        vel = _velocity(t, v, vel_dt)
        r = rolling(v)
        out[f"{col}"] = v
        out[f"{col}_mean"] = r.mean().to_numpy()
        out[f"{col}_min"] = r.min().to_numpy()
        out[f"{col}_max"] = r.max().to_numpy()
        out[f"{col}_vel"] = vel
        out[f"{col}_vel_absmax"] = rolling(np.abs(vel)).max().to_numpy()

    blink = pd.concat([_numeric(frames, "blink_left"), _numeric(frames, "blink_right")],
                      axis=1).mean(axis=1)
    out["blink_mean"] = rolling(blink).mean().to_numpy()

    face_found = _numeric(frames, "face_found").fillna(0).to_numpy(float)
    out["face_missing_frac"] = rolling(1.0 - face_found).mean().to_numpy()

    state = frames["motion_state"].astype(str) if "motion_state" in frames else \
        pd.Series("cruising", index=frames.index)
    for s in STATES:
        out[f"motion_{s}"] = (state == s).to_numpy(float)

    for col in PHONE_COLUMNS:
        out[col] = _numeric(frames, col).to_numpy(float)
    out["phone_visible_recent"] = rolling(
        _numeric(frames, "phone_visible").fillna(0)).max().to_numpy()

    feats = pd.DataFrame(out, index=frames.index)
    return feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def feature_names(window_s: float = 1.5) -> list[str]:
    """The column order the model is trained with."""
    dummy = pd.DataFrame({"t": [0.0], "motion_state": ["cruising"]})
    return list(compute_features(dummy, window_s).columns)
