import csv
import json
import os

import numpy as np
import pandas as pd
import pytest

from crash_catcher.logger import COLUMNS
from crash_catcher.training.evaluate import (rule_table, score_predictions,
                                             threshold_sweep, wilcoxon_compare)
from crash_catcher.training.features import compute_features, feature_names
from crash_catcher.training.labels import find_onsets, find_sessions, load_session, make_labels
from crash_catcher.training.train import build_dataset, cross_validate

FPS = 30


def synth_frames(seed=0, seconds=120, every=8.0, glance_s=2.5, ramp_s=0.5, decoys=True):
    """Fake recording: the head starts turning ramp_s before each glance
    onset (the precursor), then is off road for glance_s."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / FPS)
    yaw = rng.normal(0, 1.0, len(t))
    off = np.zeros(len(t), int)
    onsets = []
    o = 3.0
    while o + glance_s + 1 < seconds:
        o += rng.uniform(-1, 1)
        onsets.append(o)
        ramp = (t >= o - ramp_s) & (t < o)
        yaw[ramp] += 18 * (t[ramp] - (o - ramp_s)) / ramp_s
        glance = (t >= o) & (t < o + glance_s)
        yaw[glance] += 35
        off[glance] = 1
        o += every
    if decoys:  # small head moves that go nowhere, so the task isn't trivial
        for d in np.arange(7.0, seconds, 8.0):
            m = (t >= d) & (t < d + 0.3)
            yaw[m] += 5
    state = np.where((t // 20) % 2 == 0, "cruising", "stopped")
    alert_fixed = np.zeros(len(t), int)
    alert_context = np.zeros(len(t), int)
    for o in onsets:
        late = (t >= o + 2.0) & (t < o + glance_s)
        alert_fixed[late] = 1
        moving = late & (state == "cruising")
        alert_context[moving] = 1
    rows = []
    for i in range(len(t)):
        rows.append({"t": t[i], "frame": i, "face_found": 1, "head_yaw_rel": yaw[i],
                     "head_pitch_rel": 0.0, "gaze_x_rel": 0.0, "gaze_y_rel": 0.0,
                     "blink_left": 0.1, "blink_right": 0.1, "off_road": off[i],
                     "off_duration": 0.0, "motion_state": state[i],
                     "alert_fixed": alert_fixed[i], "alert_context": alert_context[i],
                     "event": 0})
    return rows, np.array(onsets)


def write_session(root, participant, session, seed, **kw):
    path = os.path.join(root, f"{participant}_{session}_20261027-120000")
    os.makedirs(path)
    rows, onsets = synth_frames(seed, **kw)
    with open(os.path.join(path, "frames.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump({"participant": participant, "session": session}, f)
    return path, onsets


@pytest.fixture
def sessions_dir(tmp_path):
    for i, pid in enumerate(["P01", "P02", "P03"]):
        write_session(tmp_path, pid, "main", seed=i)
    write_session(tmp_path, "P01", "practice", seed=9)  # should be skipped
    return str(tmp_path)


# ---------- labels ----------

def test_load_and_onsets(tmp_path):
    path, onsets = write_session(tmp_path, "P07", "main", seed=1)
    s = load_session(path)
    assert s.participant == "P07" and s.session_type == "main"
    found = find_onsets(s.frames)
    assert len(found) == len(onsets)
    assert np.allclose(found, onsets, atol=1.5 / FPS)


def test_short_flicker_is_not_an_onset():
    t = np.arange(0, 5, 1 / FPS)
    off = ((t >= 2.0) & (t < 2.1)).astype(int)
    assert len(find_onsets(pd.DataFrame({"t": t, "off_road": off}))) == 0


def test_labels_around_onset():
    t = np.arange(0, 6, 1 / FPS)
    off = ((t >= 3.0) & (t < 4.0)).astype(int)
    frames = pd.DataFrame({"t": t, "off_road": off})
    labels, mask = make_labels(frames, horizon=1.0)
    at = lambda x: np.argmin(np.abs(t - x))  # noqa: E731
    assert labels[at(2.5)] == 1 and labels[at(2.05)] == 1
    assert labels[at(1.9)] == 0
    assert not mask[at(3.5)]          # already off road
    assert not mask[at(5.5)]          # future not known
    assert mask[at(2.5)]


# ---------- features ----------

def test_features_have_no_lookahead():
    rows, _ = synth_frames(seed=3, seconds=20)
    df = pd.DataFrame(rows)
    base = compute_features(df)
    changed = df.copy()
    changed.loc[changed["t"] > 10.0, "head_yaw_rel"] += 100
    changed.loc[changed["t"] > 10.0, "motion_state"] = "turning"
    after = compute_features(changed)
    past = df["t"] <= 10.0
    pd.testing.assert_frame_equal(base[past], after[past])
    assert not base[~past].equals(after[~past])


def test_features_handle_missing_face_and_columns():
    df = pd.DataFrame({"t": np.arange(0, 2, 1 / FPS), "face_found": 0,
                       "motion_state": "cruising"})
    f = compute_features(df)
    assert list(f.columns) == feature_names()
    assert not f.isna().any().any()
    assert f["face_missing_frac"].iloc[-1] == 1.0
    assert (f["phone_visible"] == 0).all()


def test_velocity_catches_head_turn():
    t = np.arange(0, 2, 1 / FPS)
    yaw = np.where(t > 1.0, (t - 1.0) * 40.0, 0.0)  # turning at 40 deg/s
    f = compute_features(pd.DataFrame({"t": t, "head_yaw_rel": yaw, "face_found": 1,
                                       "motion_state": "cruising"}))
    assert f["head_yaw_rel_vel"].iloc[-1] == pytest.approx(40.0, rel=0.05)
    assert f["head_yaw_rel_vel"].iloc[10] == 0.0


# ---------- train + evaluate ----------

def test_train_lopo_beats_chance(sessions_dir):
    sessions = find_sessions(sessions_dir)
    assert {s.participant for s in sessions} == {"P01", "P02", "P03"}
    assert all(s.session_type == "main" for s in sessions)
    X, y, groups, info = build_dataset(sessions, horizon=1.0)
    preds, aucs = cross_validate(X, y, groups, info, n_trees=50)
    assert set(aucs) == {"P01", "P02", "P03"}
    # Frames 0.5-1.0 s before onset are labeled 1 but show no precursor yet,
    # so frame-level AUC can't reach 1.0 here; well above 0.5 is the point.
    assert min(aucs.values()) > 0.65
    sweep = threshold_sweep(preds, thresholds=[0.5], horizon=1.0)
    assert sweep["pct_warned_early"].iloc[0] > 50
    assert sweep["median_lead_ms"].iloc[0] > 0


def test_train_cli_end_to_end(sessions_dir, tmp_path, monkeypatch, capsys):
    import joblib
    from crash_catcher.training import train
    out = tmp_path / "models" / "predictor.joblib"
    monkeypatch.setattr("sys.argv", ["train", sessions_dir, "--trees", "30", "--out", str(out)])
    train.main()
    bundle = joblib.load(out)
    assert bundle["features"] == feature_names() and bundle["horizon"] == 1.0
    assert (out.parent / "cv_predictions.csv").exists()
    assert "Leave-one-participant-out" in capsys.readouterr().out


def test_lead_time_positive():
    t = np.arange(0, 10, 1 / FPS)
    off = ((t >= 5.0) & (t < 6.0)).astype(int)
    prob = np.where((t >= 4.6) & (t < 5.0), 0.9, 0.1)
    r = score_predictions(t, prob, off, threshold=0.5, horizon=1.0)
    assert r["onsets"] == 1 and r["warned"] == 1 and r["false_alarms"] == 0
    assert r["leads_s"][0] == pytest.approx(0.4, abs=1 / FPS)


def test_false_alarm_counted():
    t = np.arange(0, 10, 1 / FPS)
    off = np.zeros(len(t), int)
    prob = np.where((t >= 2.0) & (t < 2.2), 0.9, 0.1)
    r = score_predictions(t, prob, off, threshold=0.5)
    assert r["warned"] == 0 and r["false_alarms"] == 1


def test_rule_table_context_fewer_false_alarms(sessions_dir):
    table = rule_table(find_sessions(sessions_dir))
    wide = table.pivot(index="participant", columns="policy", values="fa_per_min")
    # Fixed alerts also fire while stopped, which the synthetic data counts
    # as not a real distraction; context alerts don't.
    assert (wide["context"] < wide["fixed"]).all()
    assert (table.loc[table.policy == "context", "fa_per_min_stopped"].fillna(0) == 0).all()
    fnr = table.set_index(["policy", "participant"])["false_negative_rate"]
    assert (fnr["fixed"] == 0).all()
    # A glance that starts while moving and runs into a stop can be missed by
    # the context rule; that trade-off is exactly what RQ1 measures.
    assert (fnr["context"] < 0.5).all()


def test_wilcoxon_effect_size_sign():
    r = wilcoxon_compare([1, 2, 3, 4, 5, 6], [0.5, 1.0, 2.0, 3.5, 4.0, 5.5])
    assert r["effect_size_r"] == 1.0 and r["p_value"] < 0.05
