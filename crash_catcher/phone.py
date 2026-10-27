"""Phone detection and "phone in use" tracking.

Two detector backends, both looking for the COCO "cell phone" class:
  hailo      YOLOv8 on the AI HAT+ (fast, frees the CPU for face tracking)
  mediapipe  Google MediaPipe EfficientDet-Lite0 on the CPU (fallback, slower)

The detector runs in its own thread on the newest camera frame, so a slow
detection never holds up face tracking. PhoneUseTracker then turns raw
detections into "is a phone being used right now, and how?"

Limits of a ready-made detector: it finds a phone that is held up or lit
in front of the camera, and often misses one that is mostly hidden in a hand,
low in the lap, or out of the camera's view. Fine-tune it on pilot footage
later if needed.
"""

import glob
import math
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

COCO80_CELL_PHONE = 67  # index of "cell phone" in YOLO's 80-class list


@dataclass
class Detection:
    score: float
    box: tuple  # (x0, y0, x1, y1) in frame pixels


# ---------------------------------------------------------------- backends

def letterbox(frame: np.ndarray, size_hw: tuple):
    """Resize keeping the aspect ratio, padding to the model's input size.
    Returns the padded image plus (scale, pad_x, pad_y) to map boxes back."""
    import cv2
    th, tw = size_hw
    h, w = frame.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    pad_x, pad_y = (tw - nw) // 2, (th - nh) // 2
    out = np.full((th, tw, 3), 114, dtype=np.uint8)
    out[pad_y:pad_y + nh, pad_x:pad_x + nw] = cv2.resize(frame, (nw, nh))
    return out, scale, pad_x, pad_y


def parse_hailo_nms(output, model_hw, frame_hw, scale, pad_x, pad_y,
                    threshold, class_id=COCO80_CELL_PHONE):
    """Hailo's YOLO models with built-in NMS return one array per class,
    each row [y0, x0, y1, x1, score] normalized to the model input."""
    if len(output) == 1 and len(output[0]) == 80:  # batch dimension
        output = output[0]
    mh, mw = model_hw
    fh, fw = frame_hw
    dets = []
    for row in output[class_id]:
        y0, x0, y1, x1, score = (float(v) for v in row[:5])
        if score < threshold:
            continue
        bx0 = np.clip((x0 * mw - pad_x) / scale, 0, fw)
        by0 = np.clip((y0 * mh - pad_y) / scale, 0, fh)
        bx1 = np.clip((x1 * mw - pad_x) / scale, 0, fw)
        by1 = np.clip((y1 * mh - pad_y) / scale, 0, fh)
        dets.append(Detection(score, (float(bx0), float(by0), float(bx1), float(by1))))
    return dets


def _hailo_arch():
    try:
        out = subprocess.run(["hailortcli", "fw-control", "identify"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in out.splitlines():
        if "Architecture" in line:
            return line.split(":")[-1].strip().upper()  # HAILO8 or HAILO8L
    return None


def find_hailo_model(path="auto"):
    if path != "auto":
        return [path]
    arch = _hailo_arch()
    files = sorted(glob.glob("/usr/share/hailo-models/yolov8*_h8*.hef"))
    if arch == "HAILO8L":
        files = [f for f in files if f.endswith("_h8l.hef")] or files
    elif arch == "HAILO8":
        files = [f for f in files if f.endswith("_h8.hef")] + \
                [f for f in files if f.endswith("_h8l.hef")]  # h8l also runs on h8
    # Prefer the "s" (small) model: accurate enough and plenty fast on the HAT.
    files.sort(key=lambda f: "yolov8s" not in f)
    return files


class HailoBackend:
    name = "hailo"

    def __init__(self, cfg):
        from picamera2.devices import Hailo
        candidates = find_hailo_model(cfg.hailo_model)
        if not candidates:
            raise RuntimeError("no YOLOv8 .hef found in /usr/share/hailo-models "
                               "(install with: sudo apt install hailo-all)")
        last_err = None
        for path in candidates:
            try:
                self.hailo = Hailo(path)
                self.model_path = path
                break
            except Exception as e:  # wrong chip for this file, try the next
                last_err = e
        else:
            raise RuntimeError(f"could not load any Hailo model: {last_err}")
        h, w, _ = self.hailo.get_input_shape()
        self.model_hw = (h, w)
        self.threshold = cfg.score_threshold

    def detect(self, frame_rgb):
        img, scale, px, py = letterbox(frame_rgb, self.model_hw)
        out = self.hailo.run(img)
        return parse_hailo_nms(out, self.model_hw, frame_rgb.shape[:2],
                               scale, px, py, self.threshold)

    def close(self):
        self.hailo.close()


class MediaPipeBackend:
    name = "mediapipe"

    def __init__(self, cfg):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        self.model_path = cfg.mediapipe_model
        self.det = vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=cfg.mediapipe_model),
            score_threshold=cfg.score_threshold,
            category_allowlist=["cell phone"]))

    def detect(self, frame_rgb):
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(frame_rgb))
        dets = []
        for d in self.det.detect(img).detections:
            b = d.bounding_box
            dets.append(Detection(d.categories[0].score,
                                  (b.origin_x, b.origin_y,
                                   b.origin_x + b.width, b.origin_y + b.height)))
        return dets

    def close(self):
        self.det.close()


def open_backend(cfg):
    if cfg.backend in ("auto", "hailo"):
        try:
            b = HailoBackend(cfg)
            print(f"[phone] using AI HAT+ ({b.model_path})")
            return b
        except Exception as e:
            if cfg.backend == "hailo":
                raise
            print(f"[phone] AI HAT+ not available ({e}); using MediaPipe on the CPU")
    b = MediaPipeBackend(cfg)
    print(f"[phone] using MediaPipe ({b.model_path})")
    return b


# ---------------------------------------------------------------- thread

class PhoneDetectorThread:
    """Runs the detector on the newest frame, as fast as the backend allows
    (capped at max_hz so the CPU backend doesn't starve face tracking)."""

    def __init__(self, cfg, backend=None):
        self.backend = backend or open_backend(cfg)
        self.max_hz = cfg.max_hz_hailo if self.backend.name == "hailo" else cfg.max_hz_cpu
        self._frame = None
        self._result = (None, [], 0.0)  # (frame time, detections, latency s)
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame_rgb, t):
        with self._lock:
            self._frame = (frame_rgb, t)
        self._new.set()

    def result(self):
        """(time of the frame it was run on, detections, seconds it took)."""
        with self._lock:
            return self._result

    def _run(self):
        min_period = 1.0 / self.max_hz
        while not self._stop.is_set():
            if not self._new.wait(timeout=0.2):
                continue
            self._new.clear()
            with self._lock:
                frame, t = self._frame
            start = time.monotonic()
            try:
                dets = self.backend.detect(frame)
            except Exception as e:
                print(f"[phone] detector error: {e}")
                dets = []
            took = time.monotonic() - start
            with self._lock:
                self._result = (t, dets, took)
            time.sleep(max(0.0, min_period - took))

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self.backend.close()


# ---------------------------------------------------------------- use tracking

@dataclass
class PhoneState:
    visible: bool = False
    score: float = math.nan
    box: tuple | None = None      # pixels, for drawing
    area: float = math.nan        # box area / frame area
    cx: float = math.nan          # box center, 0..1 across the frame
    cy: float = math.nan
    area_rate: float = math.nan   # change in area per second (+ = being lifted toward camera)
    brightness: float = math.nan  # box brightness / frame brightness (lit screen > 1)
    near_face: bool = False       # phone at the ear / in front of the face
    in_use: bool = False
    use_reason: str = ""          # "looking", "to_ear", "holding"
    use_duration: float = 0.0


def face_box(landmarks):
    if landmarks is None:
        return None
    x0, y0 = landmarks.min(axis=0)
    x1, y1 = landmarks.max(axis=0)
    return float(x0), float(y0), float(x1), float(y1)


def near(box, fbox, margin_frac):
    """True if box overlaps the face box grown by margin_frac of the face width."""
    if box is None or fbox is None:
        return False
    m = (fbox[2] - fbox[0]) * margin_frac
    return not (box[2] < fbox[0] - m or box[0] > fbox[2] + m or
                box[3] < fbox[1] - m or box[1] > fbox[3] + m)


class PhoneUseTracker:
    """A phone counts as "in use" once it has stayed in view for
    use_min_s. Short flickers (missed detections) are bridged by hold_s."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._last_seen = -math.inf
        self._last_det = None
        self._visible_since = None
        self._history = deque()  # (t, area) for area_rate
        self._last_frame_t = None
        self.state = PhoneState()

    def update(self, t, det_time, dets, frame_rgb, fbox, eyes_off_road) -> PhoneState:
        c = self.cfg
        fresh = det_time is not None and det_time != self._last_frame_t
        if fresh:
            self._last_frame_t = det_time
            best = max(dets, key=lambda d: d.score) if dets else None
            if best is not None:
                self._last_seen = det_time
                self._last_det = best
                fh, fw = frame_rgb.shape[:2]
                x0, y0, x1, y1 = best.box
                area = max(0.0, (x1 - x0) * (y1 - y0)) / (fw * fh)
                self._history.append((det_time, area))
        while self._history and t - self._history[0][0] > 1.0:
            self._history.popleft()

        visible = t - self._last_seen <= c.hold_s
        if not visible:
            self._visible_since = None
            self._last_det = None
            self.state = PhoneState()
            return self.state
        if self._visible_since is None:
            self._visible_since = self._last_seen

        d = self._last_det
        fh, fw = frame_rgb.shape[:2]
        x0, y0, x1, y1 = d.box
        area = max(0.0, (x1 - x0) * (y1 - y0)) / (fw * fh)
        area_rate = math.nan
        if len(self._history) >= 2:
            (ta, aa), (tb, ab) = self._history[0], self._history[-1]
            if tb > ta:
                area_rate = (ab - aa) / (tb - ta)

        brightness = math.nan
        ix0, iy0, ix1, iy1 = (int(max(0, v)) for v in (x0, y0, x1, y1))
        crop = frame_rgb[iy0:iy1, ix0:ix1]
        if crop.size:
            frame_mean = float(frame_rgb.mean()) or 1.0
            brightness = float(crop.mean()) / frame_mean

        close_to_face = near(d.box, fbox, c.near_face_margin)
        duration = t - self._visible_since
        in_use = duration >= c.use_min_s
        reason = ""
        if in_use:
            if eyes_off_road:
                reason = "looking"
            elif close_to_face:
                reason = "to_ear"
            else:
                reason = "holding"

        self.state = PhoneState(True, d.score, d.box, area,
                                (x0 + x1) / 2 / fw, (y0 + y1) / 2 / fh,
                                area_rate, brightness, close_to_face,
                                in_use, reason, duration if in_use else 0.0)
        return self.state


class PhonePolicy:
    """Alert when a phone has been in use for alert_s while the car is moving."""
    name = "phone"

    def __init__(self, alert_s: float):
        self.alert_s = alert_s

    def should_alert(self, phone: PhoneState, motion_state: str) -> bool:
        return phone.in_use and motion_state != "stopped" and phone.use_duration >= self.alert_s
