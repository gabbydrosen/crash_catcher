"""Train the predictive model: "will an eyes-off-road glance start in the next
`horizon` seconds?"

  python -m crash_catcher.training.train data/sessions --horizon 1.0 \
      --out models/predictor.joblib

Steps:
  1. load every main/pilot session, make labels and features for each frame
  2. leave-one-participant-out: for each person, train on everyone else and
     test on them (so the model is never tested on someone it has seen)
  3. print how each held-out person scored, and save every held-out
     prediction to cv_predictions.csv for evaluate.py
  4. train one final model on everyone and save it for the Pi
"""

import argparse
import os

import numpy as np
import pandas as pd

from .evaluate import score_participants
from .features import compute_features
from .labels import find_sessions, make_labels


def build_dataset(sessions, horizon=1.0, window_s=1.5):
    """Returns (X, y, groups, info). X has features for EVERY frame; train
    only on rows where info['mask'] is True. info also carries what
    evaluation needs (participant, session, t, off_road)."""
    xs, infos = [], []
    for s in sessions:
        f = s.frames
        labels, mask = make_labels(f, horizon)
        xs.append(compute_features(f, window_s))
        infos.append(pd.DataFrame({
            "participant": s.participant, "session": s.name,
            "t": f["t"].to_numpy(float),
            "off_road": f["off_road"].fillna(0).astype(int).to_numpy(),
            "label": labels, "mask": mask}))
    if not xs:
        raise SystemExit("No sessions found. Check the folder and --types.")
    X = pd.concat(xs, ignore_index=True)
    info = pd.concat(infos, ignore_index=True)
    return X, info["label"].to_numpy(), info["participant"].to_numpy(), info


def make_model(n_trees=300, seed=0):
    from sklearn.ensemble import RandomForestClassifier
    # class_weight="balanced": glance onsets are rare, so without it the
    # model could score well by always answering "no".
    return RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=5,
                                  class_weight="balanced", n_jobs=-1,
                                  random_state=seed)


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def cross_validate(X, y, groups, info, n_trees=300, seed=0):
    """Leave-one-participant-out. Returns info with a 'prob' column holding
    each frame's prediction from the model that never saw that person."""
    from sklearn.model_selection import LeaveOneGroupOut
    mask = info["mask"].to_numpy()
    prob = np.full(len(X), np.nan)
    aucs = {}
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        tr = train_idx[mask[train_idx]]
        if len(np.unique(y[tr])) < 2:
            continue
        model = make_model(n_trees, seed).fit(X.iloc[tr], y[tr])
        prob[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
        te = test_idx[mask[test_idx]]
        aucs[groups[test_idx[0]]] = _auc(y[te], prob[te])
    out = info.copy()
    out["prob"] = prob
    return out, aucs


def main():
    ap = argparse.ArgumentParser(description="Train the Crash Catcher predictor")
    ap.add_argument("sessions_dir", help="e.g. data/sessions")
    ap.add_argument("--horizon", type=float, default=1.0, help="seconds ahead to predict")
    ap.add_argument("--window", type=float, default=1.5, help="seconds of history per feature")
    ap.add_argument("--types", default="main,pilot",
                    help="session types to use, comma separated, or 'all'")
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="warning threshold for the printed summary")
    ap.add_argument("--out", default="models/predictor.joblib")
    args = ap.parse_args()

    import joblib

    types = None if args.types == "all" else tuple(args.types.split(","))
    sessions = find_sessions(args.sessions_dir, types)
    participants = sorted({s.participant for s in sessions})
    print(f"{len(sessions)} sessions from {len(participants)} participants: "
          f"{', '.join(participants)}")

    X, y, groups, info = build_dataset(sessions, args.horizon, args.window)
    mask = info["mask"].to_numpy()
    print(f"{mask.sum()} usable frames, {y[mask].sum()} labeled 'glance coming' "
          f"({100 * y[mask].mean():.1f}%)")

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    if len(participants) >= 2:
        preds, aucs = cross_validate(X, y, groups, info, args.trees)
        preds["horizon"] = args.horizon
        cv_path = os.path.join(out_dir, "cv_predictions.csv")
        preds.to_csv(cv_path, index=False)
        table = score_participants(preds.dropna(subset=["prob"]), args.threshold, args.horizon)
        table["frame_auc"] = table["participant"].map(aucs)
        print(f"\n== Leave-one-participant-out, threshold {args.threshold} ==")
        print(table.round(2).to_string(index=False))
        print(f"\nMedian AUC across participants: {np.nanmedian(list(aucs.values())):.3f} "
              "(0.5 = guessing, 1.0 = perfect)")
        print(f"Held-out predictions saved to {cv_path}; run evaluate.py on it "
              "for the threshold sweep.")
    else:
        print("Only one participant: skipping leave-one-out testing "
              "(scores on training data would be misleading).")

    model = make_model(args.trees).fit(X[mask], y[mask])
    joblib.dump({"model": model, "features": list(X.columns), "horizon": args.horizon,
                 "window_s": args.window, "participants": participants}, args.out)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(
        ascending=False)
    print("\nMost useful features:")
    print(importances.head(10).round(3).to_string())
    print(f"\nSaved final model (trained on everyone) to {args.out}")


if __name__ == "__main__":
    main()
