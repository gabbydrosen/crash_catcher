import math

import numpy as np
import pytest

from crash_catcher.attention import AttentionTracker, Calibrator, RoadZone
from crash_catcher.config import AttentionConfig, MotionConfig, PolicyConfig
from crash_catcher.face_tracker import FaceReading, rotation_to_euler
from crash_catcher.imu import G, VehicleFrame
from crash_catcher.motion import MotionClassifier, VehicleSample
from crash_catcher.policy import ContextPolicy, FixedPolicy

FPS = 30


def face(yaw=0.0, pitch=0.0, gx=0.0, gy=0.0, blink=0.1):
    return FaceReading(True, yaw, pitch, 0.0, gx, gy, blink, blink)


# ---------- head pose ----------

def test_rotation_to_euler_yaw():
    a = math.radians(30)
    r = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    yaw, pitch, roll = rotation_to_euler(r)
    assert yaw == pytest.approx(30, abs=1e-6)
    assert pitch == pytest.approx(0, abs=1e-6) and roll == pytest.approx(0, abs=1e-6)


# ---------- attention ----------

def make_tracker():
    return AttentionTracker(RoadZone(5.0, -3.0, 0.0, 0.0), AttentionConfig())


def test_calibration_uses_median():
    cal = Calibrator()
    for _ in range(20):
        cal.add(face(yaw=10, pitch=-5))
    cal.add(face(yaw=80))  # one bad frame
    zone = cal.finish()
    assert zone.yaw == 10 and zone.pitch == -5


def test_calibration_fails_without_face():
    with pytest.raises(RuntimeError):
        Calibrator().finish()


def test_on_road_relative_to_zone():
    tr = make_tracker()
    s = tr.update(face(yaw=15, pitch=-3), 0.0)  # 10 deg from zone: fine
    assert not s.off_road and s.head_yaw_rel == 10


def test_head_turn_duration():
    tr = make_tracker()
    tr.update(face(yaw=5), 0.0)
    s = None
    for i in range(1, 31):
        s = tr.update(face(yaw=40), i / FPS)
    assert s.off_road and s.reason == "head"
    assert s.off_road_duration == pytest.approx(29 / FPS)
    assert tr.update(face(yaw=5), 2.0).off_road_duration == 0.0


def test_blink_is_not_off_road_but_long_closure_is():
    tr = make_tracker()
    assert not tr.update(face(blink=0.9), 0.0).off_road
    assert not tr.update(face(blink=0.9), 0.2).off_road  # still a blink
    s = tr.update(face(blink=0.9), 0.5)
    assert s.off_road and s.reason == "eyes_closed"
    assert s.off_road_duration == pytest.approx(0.5)  # counted from first closed frame


def test_gaze_down_at_lap():
    s = make_tracker().update(face(gy=0.6), 0.0)
    assert s.off_road and s.reason == "gaze"


def test_missing_face_grace_then_off_road():
    tr = make_tracker()
    tr.update(face(), 0.0)
    assert not tr.update(FaceReading(False), 0.1).off_road
    s = tr.update(FaceReading(False), 0.5)
    assert s.off_road and s.reason == "no_face"
    assert s.off_road_duration == pytest.approx(0.4)


# ---------- motion ----------

def run_motion(samples):
    mc = MotionClassifier(MotionConfig())
    return [mc.update(s).state for s in samples]


def series(seconds, start=0.0, hz=100, long_accel=0.0, yaw=0.0, vib=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * hz)
    return [VehicleSample(start + i / hz, long_accel, yaw,
                          G + rng.normal(0, vib)) for i in range(n)]


def test_idle_vibration_becomes_stopped():
    states = run_motion(series(3, vib=0.05))
    assert states[50] == "cruising"   # not enough evidence yet
    assert states[-1] == "stopped"


def test_road_vibration_is_cruising():
    assert run_motion(series(3, vib=0.8))[-1] == "cruising"


def test_turning_and_braking():
    turn = series(1, vib=0.8, yaw=15.0)
    assert run_motion(turn)[-1] == "turning"
    brake = series(1, vib=0.8, long_accel=-5.0, yaw=15.0)
    assert run_motion(brake)[-1] == "hard_braking"  # braking beats turning


def test_braking_is_held_briefly():
    s = series(0.5, vib=0.8, long_accel=-5.0) + series(0.5, start=0.5, vib=0.8)
    assert run_motion(s)[-1] == "hard_braking"


def test_vehicle_frame_tilted_mount():
    # Board mounted tilted 30 deg about its x axis; car turning at 20 deg/s.
    a = math.radians(30)
    up = np.array([0, math.sin(a), math.cos(a)])
    frame = VehicleFrame(up * G, np.array([0.5, -0.2, 0.1]), "+x")
    s = frame.convert(0.0, up * G + np.array([-3.0, 0, 0]),
                      up * 20 + np.array([0.5, -0.2, 0.1]))
    assert s.yaw_rate == pytest.approx(20)
    assert s.long_accel == pytest.approx(-3.0)


# ---------- policies ----------

def test_fixed_policy():
    p = FixedPolicy(2.0)
    assert not p.should_alert(True, 1.9, "turning")
    assert p.should_alert(True, 2.0, "stopped")


def test_context_limits():
    p = ContextPolicy(PolicyConfig().context_limits_s)
    assert not p.should_alert(True, 1.5, "cruising")
    assert p.should_alert(True, 1.5, "turning")
    assert p.should_alert(True, 0.6, "hard_braking")


def test_context_stopped_then_pull_away():
    p = ContextPolicy(PolicyConfig().context_limits_s)
    assert not p.should_alert(True, 5.0, "stopped")    # glance at a red light
    assert p.should_alert(True, 5.1, "cruising")       # pulls away still looking
    assert p.should_alert(True, 5.2, "cruising")       # keeps alerting
    assert not p.should_alert(False, 0.0, "cruising")  # eyes back: stops
    assert not p.should_alert(True, 0.1, "cruising")   # new glance starts fresh
