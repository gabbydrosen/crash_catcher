"""MPU-6050 IMU driver, plus conversion into the car's frame of reference.

The MPU-6050 talks over I2C (SDA = GPIO2, SCL = GPIO3). We read it about
100 times a second in a background thread, because the camera loop is slower.

At startup the sensor must be held still for a couple of seconds:
  - gravity tells us which way is "down", so turning (yaw) is measured
    correctly however the board is tilted
  - the gyro's zero offset (bias) is measured and subtracted
"""

import math
import threading
import time

import numpy as np

from .motion import MotionClassifier, MotionReading, VehicleSample

G = 9.80665

# MPU-6050 registers
_PWR_MGMT_1 = 0x6B
_CONFIG = 0x1A
_GYRO_CONFIG = 0x1B
_ACCEL_CONFIG = 0x1C
_ACCEL_XOUT_H = 0x3B
_WHO_AM_I = 0x75

_ACCEL_LSB_PER_G = 8192.0   # +/-4 g range
_GYRO_LSB_PER_DPS = 65.5    # +/-500 deg/s range

_AXES = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0),
         "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}


class MPU6050:
    def __init__(self, bus: int = 1, address: int = 0x68):
        from smbus2 import SMBus
        self.address = address
        self.bus = SMBus(bus)
        who = self.bus.read_byte_data(address, _WHO_AM_I)
        if who not in (0x68, 0x70, 0x72, 0x98):  # 0x68 genuine; clones vary
            print(f"[imu] warning: unexpected WHO_AM_I 0x{who:02x}, continuing")
        self.bus.write_byte_data(address, _PWR_MGMT_1, 0x01)   # wake, gyro clock
        time.sleep(0.05)
        self.bus.write_byte_data(address, _CONFIG, 0x03)       # ~44 Hz low-pass filter
        self.bus.write_byte_data(address, _GYRO_CONFIG, 0x08)  # +/-500 deg/s
        self.bus.write_byte_data(address, _ACCEL_CONFIG, 0x08) # +/-4 g

    def read(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (accel m/s^2, gyro deg/s), each a 3-vector in sensor axes."""
        raw = self.bus.read_i2c_block_data(self.address, _ACCEL_XOUT_H, 14)
        vals = [int.from_bytes(bytes(raw[i:i + 2]), "big", signed=True)
                for i in range(0, 14, 2)]
        accel = np.array(vals[0:3]) / _ACCEL_LSB_PER_G * G
        gyro = np.array(vals[4:7]) / _GYRO_LSB_PER_DPS  # vals[3] is temperature
        return accel, gyro

    def close(self):
        self.bus.close()


class VehicleFrame:
    """Rotates raw sensor readings into car terms (forward accel, yaw rate)."""

    def __init__(self, gravity: np.ndarray, gyro_bias: np.ndarray, forward_axis: str):
        self.gravity = gravity
        self.up = gravity / np.linalg.norm(gravity)  # accelerometer reads +1 g "up" at rest
        self.gyro_bias = gyro_bias
        fwd = np.array(_AXES[forward_axis], dtype=float)
        fwd = fwd - np.dot(fwd, self.up) * self.up  # flatten onto the horizontal plane
        n = np.linalg.norm(fwd)
        if n < 0.3:
            raise ValueError(
                f"forward_axis {forward_axis} points almost straight up or down. "
                "Pick the sensor axis that points toward the front of the car.")
        self.forward = fwd / n

    @classmethod
    def calibrate(cls, samples, forward_axis: str):
        accels = np.array([a for a, _ in samples])
        gyros = np.array([g for _, g in samples])
        if accels.std(axis=0).max() > 0.5:
            print("[imu] warning: sensor moved during calibration; turning detection may be off")
        return cls(accels.mean(axis=0), gyros.mean(axis=0), forward_axis)

    def convert(self, t: float, accel: np.ndarray, gyro: np.ndarray) -> VehicleSample:
        linear = accel - self.gravity
        yaw_rate = float(np.dot(gyro - self.gyro_bias, self.up))
        return VehicleSample(t, float(np.dot(linear, self.forward)), yaw_rate,
                             float(np.linalg.norm(accel)))

    def to_dict(self):
        return {"gravity": self.gravity.tolist(), "gyro_bias": self.gyro_bias.tolist(),
                "forward": self.forward.tolist()}


class IMUThread:
    """Reads the IMU in the background and keeps the latest motion state."""

    def __init__(self, imu_cfg, motion_cfg, sensor=None):
        self.cfg = imu_cfg
        self.sensor = sensor or MPU6050(imu_cfg.i2c_bus, imu_cfg.address)
        self.classifier = MotionClassifier(motion_cfg)
        self.frame = None
        self.latest_sample = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def calibrate(self):
        print(f"[imu] hold still for {self.cfg.calibration_seconds:.0f} s ...")
        samples = []
        end = time.monotonic() + self.cfg.calibration_seconds
        while time.monotonic() < end:
            samples.append(self.sensor.read())
            time.sleep(1.0 / self.cfg.sample_hz)
        self.frame = VehicleFrame.calibrate(samples, self.cfg.forward_axis)
        print(f"[imu] calibrated: |g| = {np.linalg.norm(self.frame.gravity):.2f} m/s^2")

    def start(self):
        if self.frame is None:
            self.calibrate()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        period = 1.0 / self.cfg.sample_hz
        next_t = time.monotonic()
        while not self._stop.is_set():
            try:
                accel, gyro = self.sensor.read()
            except OSError as e:  # loose wire, I2C glitch: keep going
                print(f"[imu] read error: {e}")
                time.sleep(0.1)
                continue
            t = time.monotonic()
            sample = self.frame.convert(t, accel, gyro)
            with self._lock:
                self.latest_sample = sample
                self.classifier.update(sample)
            next_t += period
            time.sleep(max(0.0, next_t - time.monotonic()))

    def reading(self) -> MotionReading:
        with self._lock:
            return self.classifier.reading

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        if hasattr(self.sensor, "close"):
            self.sensor.close()


class FakeSensor:
    """Pretends to be an MPU-6050 sitting still, for testing without hardware."""

    def read(self):
        noise = np.random.normal(0, 0.03, 3)
        return np.array([0.0, 0.0, G]) + noise, np.random.normal(0, 0.1, 3)


class ManualMotion:
    """Stand-in for the IMU when there is none: the motion state is set by
    hand (keyboard keys in the preview window) or by simulator telemetry."""

    def __init__(self, state="cruising"):
        self.state = state

    def start(self):
        pass

    def reading(self) -> MotionReading:
        return MotionReading(self.state, math.nan, math.nan, math.nan)

    def stop(self):
        pass
