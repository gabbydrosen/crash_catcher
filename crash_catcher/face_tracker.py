"""Face tracking with Google MediaPipe Face Landmarker.

For each camera frame this returns:
  - head yaw / pitch / roll in degrees (which way the head points)
  - gaze_x / gaze_y: where the iris sits inside each eye, -1..1
  - eye blink scores 0..1 (from MediaPipe's blendshapes)

MediaPipe runs on the Pi 5's CPU. The AI HAT+ is not used here: it only runs
models compiled for its Hailo chip, and MediaPipe can't target it.
"""

import math
from dataclasses import dataclass

import numpy as np

# Landmark indices in MediaPipe's 478-point face mesh.
# "Left"/"right" are the person's own left and right.
RIGHT_EYE_OUTER, RIGHT_EYE_INNER = 33, 133
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 159, 145
LEFT_EYE_INNER, LEFT_EYE_OUTER = 362, 263
LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 386, 374
RIGHT_IRIS = 468  # center of the right iris (468-472)
LEFT_IRIS = 473   # center of the left iris (473-477)


@dataclass
class FaceReading:
    found: bool
    head_yaw: float = math.nan
    head_pitch: float = math.nan
    head_roll: float = math.nan
    gaze_x: float = math.nan
    gaze_y: float = math.nan
    blink_left: float = math.nan
    blink_right: float = math.nan
    landmarks: np.ndarray | None = None  # (478, 2) pixel coords, for drawing


def rotation_to_euler(r: np.ndarray) -> tuple[float, float, float]:
    """Convert a 3x3 rotation matrix to (yaw, pitch, roll) in degrees."""
    sy = math.hypot(r[0, 0], r[1, 0])
    if sy > 1e-6:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    else:  # gimbal lock, can't happen with a real head
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def _ratio(p, a, b) -> float:
    """Where p lies along segment a->b, mapped to -1..1 (0 = middle)."""
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-9:
        return math.nan
    t = float(np.dot(p - a, ab)) / denom
    return 2.0 * t - 1.0


def gaze_from_landmarks(pts: np.ndarray) -> tuple[float, float]:
    """Average iris position inside both eyes. pts is (478, 2) in pixels.

    gaze_x goes from the person's right (-1) to their left (+1);
    gaze_y goes from up (-1) to down (+1).
    """
    rx = _ratio(pts[RIGHT_IRIS], pts[RIGHT_EYE_OUTER], pts[RIGHT_EYE_INNER])
    lx = _ratio(pts[LEFT_IRIS], pts[LEFT_EYE_INNER], pts[LEFT_EYE_OUTER])
    ry = _ratio(pts[RIGHT_IRIS], pts[RIGHT_EYE_TOP], pts[RIGHT_EYE_BOTTOM])
    ly = _ratio(pts[LEFT_IRIS], pts[LEFT_EYE_TOP], pts[LEFT_EYE_BOTTOM])
    return float(np.nanmean([rx, lx])), float(np.nanmean([ry, ly]))


class FaceTracker:
    def __init__(self, cfg):
        # Imported here so the rest of the package (and the tests) work
        # without MediaPipe installed.
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=cfg.model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=cfg.min_detection_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts_ms = -1

    def process(self, frame_rgb: np.ndarray, timestamp_s: float) -> FaceReading:
        """frame_rgb: HxWx3 uint8 in RGB order."""
        ts_ms = int(timestamp_s * 1000)
        if ts_ms <= self._last_ts_ms:  # MediaPipe needs increasing timestamps
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms

        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(frame_rgb))
        result = self._landmarker.detect_for_video(image, ts_ms)
        if not result.face_landmarks:
            return FaceReading(found=False)

        h, w = frame_rgb.shape[:2]
        pts = np.array([(lm.x * w, lm.y * h) for lm in result.face_landmarks[0]])

        yaw = pitch = roll = math.nan
        if result.facial_transformation_matrixes:
            m = np.asarray(result.facial_transformation_matrixes[0])
            yaw, pitch, roll = rotation_to_euler(m[:3, :3])

        blink_l = blink_r = math.nan
        if result.face_blendshapes:
            scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
            blink_l = scores.get("eyeBlinkLeft", math.nan)
            blink_r = scores.get("eyeBlinkRight", math.nan)

        gx, gy = gaze_from_landmarks(pts) if len(pts) >= 478 else (math.nan, math.nan)
        return FaceReading(True, yaw, pitch, roll, gx, gy, blink_l, blink_r, pts)

    def close(self):
        self._landmarker.close()
