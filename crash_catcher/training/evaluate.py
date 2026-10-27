"""Score models the way the research plan asks: by events, not by frames.

RQ2 (predictive model), for a probability threshold:
  - a *warning* is the moment the model's probability rises past the threshold
  - a glance counts as *warned early* if a warning came in the `horizon`
    seconds before its onset; lead time = onset time - first such warning
  - a *false alarm* is a warning with no onset in the `horizon` after it

RQ1 (rule-based alerts logged live, alert_fixed and alert_context):
  - a *real distraction* is an off-road stretch of at least 2 s while the
    car was moving (if a `true_distraction` column from hand-labeling the
    video exists, that is used instead)
  - a *false alarm* is an alert that is not during a real distraction
  - the *false negative rate* is real distractions with no alert at all

Usage:
  python -m crash_catcher.training.evaluate models/cv_predictions.csv --out results
  python -m crash_catcher.training.evaluate --rules data/sessions --out results
"""

import argparse
import os

import numpy as np
import pandas as pd

from .labels import MIN_GLANCE_S, find_onsets, find_sessions, off_road_runs


# ---------------------------------------------------------------- helpers

def rising_edges(t: np.ndarray, flag: np.ndarray) -> np.ndarray:
    """Times where a 0/1 signal switches from 0 to 1."""
    flag = np.asarray(flag, bool)
    prev = np.concatenate([[False], flag[:-1]])
    return np.asarray(t)[flag & ~prev]


def session_minutes(t: np.ndarray, max_gap_s: float = 0.5) -> float:
    """Recording length in minutes, ignoring any big gaps (paused recording)."""
    if len(t) < 2:
        return 0.0
    return float(np.clip(np.diff(t), 0, max_gap_s).sum()) / 60.0


# ---------------------------------------------------------------- RQ2

def warning_episodes(t, above, merge_gap_s=0.2):
    """(start, end) of each stretch where the model is warning. Gaps shorter
    than merge_gap_s are bridged so one flickering warning isn't counted
    as many."""
    eps = []
    for s, e in off_road_runs(pd.DataFrame({"t": t, "off_road": above.astype(int)})):
        if eps and s - eps[-1][1] < merge_gap_s:
            eps[-1] = (eps[-1][0], e)
        else:
            eps.append((s, e))
    return eps


def score_predictions(t, prob, off_road, threshold, horizon=1.0,
                      min_glance_s=MIN_GLANCE_S, merge_gap_s=0.2) -> dict:
    """Event-level score for one session. prob is the model's probability for
    each frame (it is ignored while the driver is already off the road).

    A glance is warned early if the probability was at or above the
    threshold at some point in the `horizon` seconds before onset; the lead
    time counts from the first such moment. A warning stretch is a false
    alarm if no glance starts during it or within `horizon` after it ends.
    """
    t = np.asarray(t, float)
    off = np.asarray(off_road, bool)
    p = np.where(off, 0.0, np.nan_to_num(np.asarray(prob, float)))
    above = p >= threshold
    frames = pd.DataFrame({"t": t, "off_road": off.astype(int)})
    onsets = find_onsets(frames, min_glance_s)

    leads = []
    for o in onsets:
        window = above & (t >= o - horizon) & (t < o)
        if window.any():
            leads.append(o - t[window][0])
    episodes = warning_episodes(t, above, merge_gap_s)
    false_alarms = sum(1 for s, e in episodes
                       if not np.any((onsets >= s) & (onsets <= e + horizon)))
    dt = np.clip(np.diff(t, append=t[-1]), 0, 0.5) if len(t) else np.zeros(0)
    return {"onsets": len(onsets), "warned": len(leads), "leads_s": leads,
            "warnings": len(episodes), "false_alarms": false_alarms,
            "minutes": session_minutes(t),
            # Guards against a model that "wins" by warning nonstop.
            "warning_minutes": float(dt[above].sum()) / 60.0}


def summarize(parts: list[dict]) -> dict:
    """Combine several session scores (e.g. all of one participant's)."""
    onsets = sum(p["onsets"] for p in parts)
    warned = sum(p["warned"] for p in parts)
    leads = [x for p in parts for x in p["leads_s"]]
    minutes = sum(p["minutes"] for p in parts)
    fa = sum(p["false_alarms"] for p in parts)
    return {
        "onsets": onsets,
        "pct_warned_early": 100.0 * warned / onsets if onsets else np.nan,
        "median_lead_ms": 1000.0 * float(np.median(leads)) if leads else np.nan,
        "false_alarms": fa,
        "fa_per_min": fa / minutes if minutes else np.nan,
        "pct_time_warning": 100.0 * sum(p["warning_minutes"] for p in parts) / minutes
        if minutes else np.nan,
        "minutes": minutes,
    }


def score_participants(preds: pd.DataFrame, threshold, horizon=1.0) -> pd.DataFrame:
    """One row per participant. preds needs columns participant, session, t,
    prob, off_road (train.py writes this as cv_predictions.csv)."""
    rows = []
    for pid, g in preds.groupby("participant"):
        parts = [score_predictions(s["t"], s["prob"], s["off_road"], threshold, horizon)
                 for _, s in g.sort_values("t").groupby("session")]
        rows.append({"participant": pid, **summarize(parts)})
    return pd.DataFrame(rows)


def threshold_sweep(preds: pd.DataFrame, thresholds=None, horizon=1.0) -> pd.DataFrame:
    """The lead time vs. false alarm trade-off, pooled over everyone."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.1, 0.96, 0.05), 2)
    rows = []
    for th in thresholds:
        parts = [score_predictions(s["t"], s["prob"], s["off_road"], th, horizon)
                 for _, s in preds.sort_values("t").groupby(["participant", "session"])]
        rows.append({"threshold": th, **summarize(parts)})
    return pd.DataFrame(rows)


def plot_tradeoff(sweep: pd.DataFrame, path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sweep["fa_per_min"], sweep["median_lead_ms"], marker="o")
    for _, r in sweep.iloc[::3].iterrows():
        ax.annotate(f"{r['threshold']:.2f}", (r["fa_per_min"], r["median_lead_ms"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("False alarms per minute")
    ax.set_ylabel("Median lead time (ms)")
    ax.set_title("Earlier warnings cost more false alarms")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


# ---------------------------------------------------------------- RQ1

def distraction_runs(frames: pd.DataFrame, min_s: float = 2.0):
    """(start, end, is_real) for each off-road stretch."""
    t = frames["t"].to_numpy(float)
    moving = (frames["motion_state"].astype(str) != "stopped").to_numpy()
    truth = frames["true_distraction"].fillna(0).to_numpy().astype(bool) \
        if "true_distraction" in frames else None
    runs = []
    for s, e in off_road_runs(frames):
        inside = (t >= s) & (t < e) if e > s else (t == s)
        if truth is not None:
            real = bool(truth[inside].any())
        else:
            real = (e - s) >= min_s and bool(moving[inside].any())
        runs.append((s, e, real))
    return runs


def score_rule_alerts(frames: pd.DataFrame, alert_col: str, min_s: float = 2.0) -> dict:
    t = frames["t"].to_numpy(float)
    alert = pd.to_numeric(frames[alert_col], errors="coerce").fillna(0).to_numpy().astype(bool)
    states = frames["motion_state"].astype(str).to_numpy()
    runs = distraction_runs(frames, min_s)
    alert_times = rising_edges(t, alert)

    def run_of(x):
        for s, e, real in runs:
            if s <= x <= e:
                return real
        return False

    false_by_state = {}
    false_alarms = 0
    for a in alert_times:
        if not run_of(a):
            false_alarms += 1
            st = states[np.searchsorted(t, a)]
            false_by_state[st] = false_by_state.get(st, 0) + 1

    real_runs = [(s, e) for s, e, real in runs if real]
    missed = sum(1 for s, e in real_runs if not alert[(t >= s) & (t <= e)].any())

    dt = np.clip(np.diff(t, append=t[-1]), 0, 0.5) if len(t) else np.zeros(0)
    minutes_by_state = {s: float(dt[states == s].sum()) / 60.0 for s in set(states)}
    result = {
        "alerts": len(alert_times),
        "false_alarms": false_alarms,
        "minutes": session_minutes(t),
        "real_distractions": len(real_runs),
        "missed": missed,
    }
    result["false_by_state"] = false_by_state
    result["minutes_by_state"] = minutes_by_state
    return result


def rule_table(sessions, alert_cols=("alert_fixed", "alert_context"), min_s=2.0) -> pd.DataFrame:
    """One row per participant per policy, with FA/min overall and per state."""
    rows = []
    by_pid = {}
    for s in sessions:
        by_pid.setdefault(s.participant, []).append(s)
    for pid, sess in sorted(by_pid.items()):
        for col in alert_cols:
            parts = [score_rule_alerts(s.frames, col, min_s) for s in sess if col in s.frames]
            if not parts:
                continue
            minutes = sum(p["minutes"] for p in parts)
            fa = sum(p["false_alarms"] for p in parts)
            real = sum(p["real_distractions"] for p in parts)
            missed = sum(p["missed"] for p in parts)
            row = {"participant": pid, "policy": col.replace("alert_", ""),
                   "fa_per_min": fa / minutes if minutes else np.nan,
                   "false_negative_rate": missed / real if real else np.nan,
                   "alerts": sum(p["alerts"] for p in parts), "minutes": minutes}
            for st in ("stopped", "cruising", "turning", "hard_braking"):
                f = sum(p["false_by_state"].get(st, 0) for p in parts)
                m = sum(p["minutes_by_state"].get(st, 0.0) for p in parts)
                row[f"fa_per_min_{st}"] = f / m if m else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- statistics

def wilcoxon_compare(a, b) -> dict:
    """Paired comparison across participants (same people, two systems).
    Returns the p-value plus an effect size (rank-biserial correlation,
    -1..1: +1 means a is higher for everyone)."""
    from scipy.stats import rankdata, wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    d = a - b
    nz = d[d != 0]
    out = {"n": int(len(d)), "median_a": float(np.median(a)) if len(a) else np.nan,
           "median_b": float(np.median(b)) if len(b) else np.nan,
           "median_diff": float(np.median(d)) if len(d) else np.nan}
    if len(nz) == 0:
        return {**out, "statistic": np.nan, "p_value": 1.0, "effect_size_r": 0.0}
    res = wilcoxon(a, b)
    ranks = rankdata(np.abs(nz))
    w_plus, w_minus = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    return {**out, "statistic": float(res.statistic), "p_value": float(res.pvalue),
            "effect_size_r": float((w_plus - w_minus) / (w_plus + w_minus))}


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description="Score Crash Catcher models and alert rules")
    ap.add_argument("predictions", nargs="?", help="cv_predictions.csv from train.py")
    ap.add_argument("--horizon", type=float, default=None,
                    help="prediction horizon in s (default: the one in the CSV, else 1.0)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="threshold for the per-participant table")
    ap.add_argument("--rules", metavar="SESSIONS_DIR",
                    help="also score alert_fixed vs alert_context (RQ1)")
    ap.add_argument("--types", default="main",
                    help="session types for --rules, comma separated, or 'all'")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    pd.set_option("display.width", 140)

    if args.predictions:
        preds = pd.read_csv(args.predictions)
        horizon = args.horizon or (float(preds["horizon"].iloc[0]) if "horizon" in preds else 1.0)
        sweep = threshold_sweep(preds, horizon=horizon)
        sweep.to_csv(os.path.join(args.out, "threshold_sweep.csv"), index=False)
        print(f"== RQ2 threshold sweep (horizon {horizon} s) ==")
        print(sweep.round(2).to_string(index=False))
        per = score_participants(preds, args.threshold, horizon)
        per.to_csv(os.path.join(args.out, "per_participant.csv"), index=False)
        print(f"\n== Per participant at threshold {args.threshold} ==")
        print(per.round(2).to_string(index=False))
        if plot_tradeoff(sweep, os.path.join(args.out, "lead_vs_false_alarms.png")):
            print(f"\nSaved plot: {args.out}/lead_vs_false_alarms.png")

    if args.rules:
        types = None if args.types == "all" else tuple(args.types.split(","))
        table = rule_table(find_sessions(args.rules, types))
        table.to_csv(os.path.join(args.out, "rq1_rules.csv"), index=False)
        print("\n== RQ1: fixed vs context alerts ==")
        print(table.round(3).to_string(index=False))
        if not table.empty:
            wide = table.pivot(index="participant", columns="policy", values="fa_per_min")
            if {"fixed", "context"} <= set(wide.columns):
                print("\nFalse alarms/min, context vs fixed (Wilcoxon):",
                      wilcoxon_compare(wide["context"], wide["fixed"]))

    if not args.predictions and not args.rules:
        ap.print_help()


if __name__ == "__main__":
    main()
