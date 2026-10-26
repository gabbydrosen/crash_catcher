# Training your own model: the whole picture

This explains what "training an AI" means for Crash Catcher, from the data
the Pi already records to a model running live. Nothing here is magic. It's a
pipeline of steps you can check one at a time.

## 1. What training actually is

**Supervised learning:** you show the computer many examples, each one with
the right answer, and it finds a rule that maps examples to answers.

For Crash Catcher, one example is "the last 1–2 seconds of the driver's
head and eye movement, at this moment". The answer is "did an eyes-off-road
glance start within the next second? yes (1) or no (0)".

After training, the model gets a new moment it has never seen and outputs a
probability, e.g. 0.82. If that's above a threshold you choose, it warns.
That's Novelty 2: a warning *before* the glance instead of after it.

You don't write the rule yourself. The fixed 2 s policy is a rule *you* wrote;
the model *learns* its rule from data. That's the whole difference.

## 2. The data: you're already collecting it

Every run of `crash_catcher.main` writes `frames.csv`, one row per camera
frame (~20–30 per second):

| Column | Meaning |
|---|---|
| `t` | time in seconds |
| `head_yaw_rel`, `head_pitch_rel` | head angle relative to the driver's calibrated road center |
| `gaze_x_rel`, `gaze_y_rel` | iris position relative to road center |
| `blink_left/right` | how closed each eye is, 0..1 |
| `face_found` | 0 if MediaPipe lost the face |
| `off_road`, `off_reason` | the rule-based "eyes off road" decision |
| `motion_state` | stopped / cruising / turning / hard_braking |
| `event` | 1 on frames where you pressed `e` / the button (a prompt was given) |

**Rule #1: every participant's data must be recorded with alerts OFF in the
main drive**, so their behavior is natural and every model is scored on the same
recordings.

## 3. Labels: the "right answers"

Labels come from the data itself:

1. **Onset** = a frame where `off_road` switches from 0 to 1. That's the moment
   the eyes left the road.
2. **Clean it up:** ignore glances shorter than ~0.3 s (tracking flicker), and
   spot-check onsets against the reference video. You don't have to check every one;
   checking a random 50 and counting how many are right gives you an accuracy number for
   your labels, which judges like.
3. **Label each frame:** `label = 1` if an onset happens within the next 1.0 s
   (the *prediction horizon*), else 0. Frames where the driver is *already*
   off road are left out of training, because the question is about the *start* of a glance.

The notification prompts help here. The Pi knows the exact moment it buzzed the test
phone, so you get a clean stimulus → reaction → glance sequence.

## 4. Features: what the model gets to see

A model can't take "a video" as input. You give it numbers that describe the
last 1–2 seconds. These are **features**. For each frame, compute over the
previous window:

- **head yaw/pitch velocity** (how fast the head is turning): change in angle
  ÷ change in time. This is the head *starting* to turn, which is the main precursor.
- **gaze velocity** toward the lap/console (gaze_y increasing = eyes moving down)
- current, mean, min and max of each angle over the window
- blink level, whether the face was lost recently
- motion state (as 4 yes/no columns; this connects Novelty 1 and Novelty 2)
- *later:* phone features from YOLO (phone appears, gets bigger, screen brightens)

Velocity matters more than position. By the time the *position* is far
off-center, the glance has already started. The *speed* shows it beginning.

## 5. Splitting: the one rule that protects your results

If frames from the same person are in both the training set and the test set,
the model can memorize that person, and your score will look much better than it
really is. Judges who know ML will ask about this.

**Leave-one-participant-out cross-validation:** with 12 participants, train on 11,
test on the 12th, and repeat 12 times so each person is the test person once.
Report the average and the spread. scikit-learn does this in one line
(`LeaveOneGroupOut`).

## 6. Models, simplest first

1. **Fixed rule** (baseline): 2.0 s. Already logged as `alert_fixed`.
2. **Context rule**: already logged as `alert_context`.
3. **Random forest** (scikit-learn): hundreds of small decision trees that
   vote. Fast to train on a laptop, works well with a few thousand
   examples, and can tell you *which features mattered most*, which makes a great
   graph for the board. **Start here.**
4. **Small GRU / 1D-CNN** (neural networks over the raw 1–2 s sequence):
   only if the random forest works and there's time left. With 12–15 people
   it may not beat the forest. If so, that's a legitimate finding.

Training happens on a **laptop**, not the Pi. Only the finished model goes back
to the Pi.

## 7. Scoring

For each held-out person, replay their recording through each model and compute:

- **% of glances warned before onset**, and the **median lead time** in ms
  (onset time − first warning time; positive = early)
- **false alarms per minute** (warnings with no glance following)
- for RQ1: false alarms per minute and missed glances, per motion state

Then sweep the warning threshold (0.3, 0.4, ... 0.9) and the horizon (0.5, 1.0,
1.5 s). Plot **lead time vs. false alarms**. That curve is likely your strongest
graph. Compare models across participants with the Wilcoxon signed-rank test.

**Set your targets (H1: 30%, H2: 50% and 300 ms) before looking at the results**, and
write them down with a date.

## 8. Back onto the Pi

The trained forest is saved to a file (`joblib.dump`). On the Pi, the main loop
keeps the last 1–2 s of readings, computes the same features each frame, and
calls `model.predict_proba`. A random forest takes well under a millisecond, so
it won't slow anything down. The feature code must be **shared** between
training and live use; if they differ even slightly, the model gets inputs it never saw.

## 9. What to do now

1. Get the hardware working (`scripts/hardware_check.py`).
2. **Pilot on the team:** run several 10–15 minute sessions each, with deliberate
   glances at a phone, the lap and the "radio". This tests the whole pipeline and gives
   the first training data, so the training code can be built *before* participants exist.
3. Tune the margins and limits in `config.py` on pilot data, then **freeze them**
   (the end of week 3 in the plan).
4. Next code steps: a labeling + feature script, a training script with
   leave-one-out, and a replay/scoring script. These run on the CSVs this code
   already produces.
