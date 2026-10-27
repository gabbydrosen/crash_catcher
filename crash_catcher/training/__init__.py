"""Training and evaluation for the predictive model (Novelty 2) and offline
scoring of the rule-based alerts (Novelty 1). Runs on a laptop, not the Pi.

  labels.py    load recorded sessions, find glance onsets, make the 0/1 answers
  features.py  turn the last 1-2 s of readings into numbers the model can use
  train.py     train a random forest with leave-one-participant-out testing
  evaluate.py  event-level scoring, threshold sweep, Wilcoxon comparisons
"""
