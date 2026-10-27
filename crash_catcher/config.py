"""All tunable settings in one place.

Change values here, not inside the other modules. The eyes-off-road limits
come straight from the research plan; tune them on pilot data from the team,
then freeze them before participant sessions.
"""

from dataclasses import dataclass, field


@dataclass
class CameraConfig:
    # "picamera" for a Raspberry Pi camera module (ribbon cable),
    # "usb" for a USB webcam, or a path to a video file for testing.
    source: str = "picamera"
    width: int = 640
    height: int = 480
    fps: int = 30
    usb_index: int = 0


@dataclass
class FaceConfig:
    model_path: str = "models/face_landmarker.task"
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class AttentionConfig:
    # How long the calibration step lasts. The plan says 60 s for participants;
    # use something short like 5 s while developing.
    calibration_seconds: float = 10.0
    # How far from the calibrated "road center" the head can turn
    # before it counts as eyes-off-road (degrees).
    head_yaw_margin_deg: float = 20.0
    head_pitch_margin_deg: float = 15.0
    # How far the eyes can move inside the eye socket (0 = calibrated
    # center, 1 = all the way to the eye corner).
    gaze_x_margin: float = 0.35
    gaze_y_margin: float = 0.35
    # Eyes closed for longer than a blink also counts as off-road.
    blink_threshold: float = 0.55
    blink_grace_s: float = 0.4
    # If the face disappears (turned far away, looking down at the lap),
    # count it as off-road after this long.
    no_face_grace_s: float = 0.3


@dataclass
class PhoneConfig:
    # "auto" tries the AI HAT+ first and falls back to the CPU.
    # Also "hailo", "mediapipe", or "off".
    backend: str = "auto"
    # "auto" picks a YOLOv8 model in /usr/share/hailo-models that matches the chip.
    hailo_model: str = "auto"
    mediapipe_model: str = "models/efficientdet_lite0.tflite"
    # Phones are small and often half hidden, so accept fairly low scores.
    score_threshold: float = 0.35
    max_hz_hailo: float = 30.0
    max_hz_cpu: float = 8.0      # leave CPU time for face tracking
    # Bridge missed detections shorter than this (seconds).
    hold_s: float = 0.6
    # A phone in view this long counts as "in use".
    use_min_s: float = 1.0
    # "Near the face" = within this fraction of the face width (phone to ear).
    near_face_margin: float = 0.3


@dataclass
class IMUConfig:
    enabled: bool = True
    i2c_bus: int = 1
    address: int = 0x68
    sample_hz: float = 100.0
    # Which way the sensor's printed axis points in the car. One of
    # "+x", "-x", "+y", "-y", "+z", "-z". Only needed for braking; turning
    # and stopped work no matter how the board is mounted.
    forward_axis: str = "+x"
    # Hold the sensor still for this long at startup to measure gravity
    # and gyro bias.
    calibration_seconds: float = 2.0


@dataclass
class MotionConfig:
    window_s: float = 1.0
    # "Stopped": accel magnitude barely varies (engine idle only).
    stopped_accel_std: float = 0.15       # m/s^2
    stopped_min_duration_s: float = 1.5
    # "Turning": yaw rate (after smoothing) above this.
    turning_yaw_rate_dps: float = 8.0
    turning_hold_s: float = 0.5
    # "Hard braking": forward deceleration beyond this (about 0.35 g).
    hard_brake_mps2: float = 3.5
    brake_hold_s: float = 1.0


@dataclass
class PolicyConfig:
    # Fixed baseline: one limit everywhere.
    fixed_limit_s: float = 2.0
    # Context-aware limits per motion state (from the research plan table).
    # "stopped" is None: no alert while stopped, but alert at once if the
    # driver is still looking away when the car starts moving.
    context_limits_s: dict = field(default_factory=lambda: {
        "stopped": None,
        "cruising": 2.0,
        "turning": 1.0,
        "hard_braking": 0.5,
    })
    # Which policy drives the LED and buzzer live: "context" or "fixed".
    # Both are always computed and logged.
    live_policy: str = "context"
    # Alert when a phone has been in use this long while the car is moving.
    phone_alert_s: float = 1.0
    # Whether phone alerts also drive the LED and buzzer (always logged).
    phone_alerts_live: bool = True


@dataclass
class PinConfig:
    # BCM GPIO numbers (not physical pin numbers). See README wiring table.
    led_red: int = 17
    led_green: int = 27
    led_blue: int = 22
    # True if the RGB LED's long leg goes to 3.3V (common anode),
    # False if it goes to GND (common cathode).
    led_common_anode: bool = False
    buzzer: int = 18
    buzzer_hz: int = 2000
    # Optional push button the experimenter presses when a prompt is given.
    event_button: int | None = None


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    phone: PhoneConfig = field(default_factory=PhoneConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    imu: IMUConfig = field(default_factory=IMUConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    pins: PinConfig = field(default_factory=PinConfig)
    data_dir: str = "data/sessions"
