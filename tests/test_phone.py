import numpy as np
import pytest

from crash_catcher.config import PhoneConfig
from crash_catcher.phone import (COCO80_CELL_PHONE, Detection, PhonePolicy, PhoneUseTracker,
                                 letterbox, near, parse_hailo_nms)

FRAME = np.full((480, 640, 3), 60, dtype=np.uint8)


def test_letterbox_and_hailo_box_mapping():
    img, scale, px, py = letterbox(FRAME, (640, 640))
    assert img.shape == (640, 640, 3) and scale == 1.0 and (px, py) == (0, 80)
    # A phone at frame pixels x 100..200, y 50..150 is at model y 130..230.
    out = [np.zeros((0, 5)) for _ in range(80)]
    out[COCO80_CELL_PHONE] = np.array([[130 / 640, 100 / 640, 230 / 640, 200 / 640, 0.8],
                                       [0.1, 0.1, 0.2, 0.2, 0.1]])  # below threshold
    dets = parse_hailo_nms(out, (640, 640), (480, 640), scale, px, py, 0.35)
    assert len(dets) == 1
    assert dets[0].box == pytest.approx((100, 50, 200, 150))


def test_hailo_batch_dimension():
    out = [[np.zeros((0, 5)) for _ in range(80)]]
    assert parse_hailo_nms(out, (640, 640), (480, 640), 1.0, 0, 80, 0.35) == []


def test_near_face():
    face = (300, 100, 400, 220)
    assert near((390, 120, 450, 230), face, 0.3)       # at the ear
    assert not near((50, 350, 150, 470), face, 0.3)    # down in the lap


def feed(tracker, times, det_box=(100, 300, 180, 420), seen=lambda t: True,
         off_road=False, fbox=None):
    s = None
    for t in times:
        dets = [Detection(0.7, det_box)] if seen(t) else []
        s = tracker.update(t, t, dets, FRAME, fbox, off_road)
    return s


def test_phone_becomes_in_use_after_min_time():
    tr = PhoneUseTracker(PhoneConfig())
    s = feed(tr, np.arange(0, 0.9, 0.1))
    assert s.visible and not s.in_use
    s = feed(tr, np.arange(0.9, 1.3, 0.1))
    assert s.in_use and s.use_reason == "holding"


def test_missed_detections_are_bridged():
    tr = PhoneUseTracker(PhoneConfig())
    # Detector misses every other run: still one continuous "in use".
    s = feed(tr, np.arange(0, 1.6, 0.1), seen=lambda t: round(t * 10) % 2 == 0)
    assert s.in_use


def test_phone_gone_resets():
    tr = PhoneUseTracker(PhoneConfig())
    feed(tr, np.arange(0, 1.5, 0.1))
    s = feed(tr, np.arange(1.5, 2.5, 0.1), seen=lambda t: False)
    assert not s.visible and not s.in_use


def test_use_reasons():
    tr = PhoneUseTracker(PhoneConfig())
    assert feed(tr, np.arange(0, 1.3, 0.1), off_road=True).use_reason == "looking"
    tr = PhoneUseTracker(PhoneConfig())
    s = feed(tr, np.arange(0, 1.3, 0.1), det_box=(390, 120, 450, 230),
             fbox=(300, 100, 400, 220))
    assert s.use_reason == "to_ear" and s.near_face


def test_lit_screen_is_brighter():
    frame = FRAME.copy()
    frame[300:420, 100:180] = 230
    tr = PhoneUseTracker(PhoneConfig())
    s = tr.update(0.0, 0.0, [Detection(0.7, (100, 300, 180, 420))], frame, None, False)
    assert s.brightness > 2


def test_phone_policy_ignores_stopped():
    tr = PhoneUseTracker(PhoneConfig())
    s = feed(tr, np.arange(0, 2.1, 0.1))
    p = PhonePolicy(1.0)
    assert p.should_alert(s, "cruising")
    assert not p.should_alert(s, "stopped")
