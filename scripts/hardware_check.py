"""Test each part on its own before running the full system.

  python scripts/hardware_check.py led
  python scripts/hardware_check.py buzzer
  python scripts/hardware_check.py imu      # tilt/rotate the board and watch the numbers
  python scripts/hardware_check.py camera
  python scripts/hardware_check.py face     # needs the model downloaded
  python scripts/hardware_check.py phone    # hold a phone up to the camera
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from crash_catcher.config import Config  # noqa: E402


def check_led(cfg):
    from crash_catcher.alerts import COLORS, GPIOAlerts
    a = GPIOAlerts(cfg.pins)
    for name in ["red", "green", "blue", "yellow", "purple"]:
        print(f"LED should be {name.upper()}")
        a.led.color = COLORS[name]
        time.sleep(1.5)
    print("If red and blue look swapped, swap led_red/led_blue in config.py.")
    print("If the LED was on when it should be off, set led_common_anode=True.")
    a.close()


def check_buzzer(cfg):
    from crash_catcher.alerts import GPIOAlerts
    a = GPIOAlerts(cfg.pins)
    for hz in [1000, 2000, 3000]:
        print(f"Buzzer at {hz} Hz")
        a.buzzer.frequency = hz
        a.beep(0.5)
        time.sleep(0.3)
    print("Pick the loudest frequency and put it in config.py as buzzer_hz.")
    a.close()


def check_imu(cfg):
    from crash_catcher.imu import MPU6050, VehicleFrame
    try:
        imu = MPU6050(cfg.imu.i2c_bus, cfg.imu.address)
    except OSError as e:
        print(f"Can't talk to the MPU-6050 ({e}).")
        print("Check wiring, enable I2C (sudo raspi-config > Interface Options > I2C),")
        print("and run `i2cdetect -y 1`; you should see 68 in the grid.")
        return
    print("Hold the board still for 2 seconds...")
    samples = [imu.read() for _ in range(200) if not time.sleep(0.01)]
    frame = VehicleFrame.calibrate(samples, cfg.imu.forward_axis)
    print("Now: rotate it flat on the table (yaw rate should change),")
    print(f"and slide it along its {cfg.imu.forward_axis} axis (forward accel should change). Ctrl+C to stop.")
    try:
        while True:
            accel, gyro = imu.read()
            s = frame.convert(time.monotonic(), accel, gyro)
            print(f"\rraw accel {accel.round(2)}  forward {s.long_accel:+6.2f} m/s2  "
                  f"yaw {s.yaw_rate:+7.1f} deg/s   ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    imu.close()


def check_camera(cfg):
    import cv2
    from crash_catcher.camera import open_camera
    cam = open_camera(cfg.camera)
    t0, n = time.monotonic(), 0
    while time.monotonic() - t0 < 3:
        frame = cam.read()
        n += 1
    print(f"Camera OK: {frame.shape[1]}x{frame.shape[0]}, about {n / 3:.0f} fps")
    cv2.imwrite("camera_test.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print("Saved camera_test.jpg. Open it: colors should look natural.")
    cam.close()


def check_face(cfg):
    from crash_catcher.camera import open_camera
    from crash_catcher.face_tracker import FaceTracker
    cam = open_camera(cfg.camera)
    tracker = FaceTracker(cfg.face)
    print("Look at the camera, then turn your head. Ctrl+C to stop.")
    try:
        t0, n = time.monotonic(), 0
        while True:
            r = tracker.process(cam.read(), time.monotonic())
            n += 1
            fps = n / (time.monotonic() - t0)
            if r.found:
                print(f"\ryaw {r.head_yaw:+6.1f}  pitch {r.head_pitch:+6.1f}  "
                      f"gaze {r.gaze_x:+.2f},{r.gaze_y:+.2f}  blink {r.blink_left:.2f}  "
                      f"{fps:4.1f} fps   ", end="", flush=True)
            else:
                print(f"\rno face{' ' * 60}", end="", flush=True)
    except KeyboardInterrupt:
        print()
    tracker.close()
    cam.close()


def check_phone(cfg):
    from crash_catcher.camera import open_camera
    from crash_catcher.phone import open_backend
    cam = open_camera(cfg.camera)
    backend = open_backend(cfg.phone)
    print("Hold a phone up in front of the camera. Ctrl+C to stop.")
    try:
        while True:
            frame = cam.read()
            t0 = time.monotonic()
            dets = backend.detect(frame)
            ms = (time.monotonic() - t0) * 1000
            if dets:
                best = max(dets, key=lambda d: d.score)
                print(f"\rPHONE  score {best.score:.2f}  box {[int(v) for v in best.box]}  "
                      f"{ms:5.0f} ms   ", end="", flush=True)
            else:
                print(f"\rno phone{' ' * 40}{ms:5.0f} ms   ", end="", flush=True)
    except KeyboardInterrupt:
        print()
    backend.close()
    cam.close()


CHECKS = {"led": check_led, "buzzer": check_buzzer, "imu": check_imu,
          "camera": check_camera, "face": check_face, "phone": check_phone}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in CHECKS:
        print(__doc__)
        sys.exit(1)
    CHECKS[sys.argv[1]](Config())
