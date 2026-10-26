"""Run Crash Catcher live.

Examples (run from the project folder):
  python -m crash_catcher.main --participant P01 --session main --alerts off
  python -m crash_catcher.main --participant dev --preview          # testing
  python -m crash_catcher.main --camera usb --no-imu --no-gpio --preview  # laptop

Preview window keys:
  q quit   e mark an event   1-4 set motion state when running --no-imu
"""

import argparse
import signal
import time

import cv2

from .alerts import GPIOAlerts, NullAlerts
from .attention import AttentionTracker, Calibrator
from .camera import open_camera
from .config import Config
from .face_tracker import FaceTracker
from .imu import FakeSensor, IMUThread, ManualMotion
from .logger import SessionLogger
from .policy import ContextPolicy, FixedPolicy

STATE_KEYS = {ord("1"): "stopped", ord("2"): "cruising",
              ord("3"): "turning", ord("4"): "hard_braking"}


def parse_args():
    p = argparse.ArgumentParser(description="Crash Catcher driver monitor")
    p.add_argument("--participant", default="dev", help="e.g. P01")
    p.add_argument("--session", default="dev",
                   help="practice, main, alerts_on, pilot, dev ...")
    p.add_argument("--alerts", choices=["on", "off"], default="on",
                   help="off = LED and buzzer stay dark (main drive); still logged")
    p.add_argument("--policy", choices=["context", "fixed"], help="which policy drives live alerts")
    p.add_argument("--camera", help="picamera, usb, or a video file path")
    p.add_argument("--no-imu", action="store_true", help="no MPU-6050; set state with keys 1-4")
    p.add_argument("--fake-imu", action="store_true", help="simulated still IMU, for testing")
    p.add_argument("--no-gpio", action="store_true", help="no LED/buzzer (e.g. on a laptop)")
    p.add_argument("--preview", action="store_true", help="show the camera window (needs a desktop)")
    p.add_argument("--record", action="store_true", help="also save the camera video")
    p.add_argument("--calib-seconds", type=float, help="road-center calibration length")
    return p.parse_args()


def draw_overlay(frame_bgr, face, att, motion, live_alert, fps):
    if face.landmarks is not None:
        for x, y in face.landmarks[::4]:
            cv2.circle(frame_bgr, (int(x), int(y)), 1, (0, 255, 0), -1)
    color = (0, 0, 255) if live_alert else ((0, 200, 255) if att.off_road else (0, 200, 0))
    lines = [
        f"{'ALERT' if live_alert else ('OFF ROAD: ' + att.reason if att.off_road else 'eyes on road')}"
        f"  {att.off_road_duration:.1f}s",
        f"motion: {motion.state}   fps: {fps:.0f}",
        f"yaw {att.head_yaw_rel:+.0f}  pitch {att.head_pitch_rel:+.0f}  "
        f"gaze {att.gaze_x_rel:+.2f},{att.gaze_y_rel:+.2f}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(frame_bgr, text, (10, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color if i == 0 else (255, 255, 255), 2)
    return frame_bgr


def main():
    args = parse_args()
    cfg = Config()
    if args.camera:
        cfg.camera.source = args.camera
    if args.calib_seconds:
        cfg.attention.calibration_seconds = args.calib_seconds
    if args.policy:
        cfg.policy.live_policy = args.policy

    stop = False

    def handle_sigint(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, handle_sigint)

    camera = open_camera(cfg.camera)
    tracker = FaceTracker(cfg.face)
    alerts = NullAlerts() if args.no_gpio else GPIOAlerts(cfg.pins)
    live_alerts = alerts if args.alerts == "on" else NullAlerts()

    if args.no_imu:
        motion_src = ManualMotion()
    else:
        sensor = FakeSensor() if args.fake_imu else None
        motion_src = IMUThread(cfg.imu, cfg.motion, sensor=sensor)
    motion_src.start()  # IMU calibrates here: keep it still

    event_flag = {"pending": False}
    button = None
    if cfg.pins.event_button is not None and not args.no_gpio:
        from gpiozero import Button
        button = Button(cfg.pins.event_button, bounce_time=0.05)
        button.when_pressed = lambda: event_flag.update(pending=True)

    log = SessionLogger(cfg.data_dir, args.participant, args.session)
    writer = None
    print(f"[main] logging to {log.dir}")

    # ---- Road-center calibration --------------------------------------
    print(f"[main] CALIBRATION: look at the center of the road for "
          f"{cfg.attention.calibration_seconds:.0f} s")
    alerts.show("calibrating")  # calibration light shows even with alerts off
    cal = Calibrator()
    t_end = time.monotonic() + cfg.attention.calibration_seconds
    while time.monotonic() < t_end and not stop:
        frame = camera.read()
        if frame is None:
            break
        cal.add(tracker.process(frame, time.monotonic()))
        if args.preview:
            cv2.imshow("Crash Catcher", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
    zone = cal.finish()
    print(f"[main] road zone: yaw {zone.yaw:.1f}, pitch {zone.pitch:.1f} "
          f"({cal.count} frames)")
    alerts.show("off")
    alerts.beep(0.1)

    attention = AttentionTracker(zone, cfg.attention)
    fixed = FixedPolicy(cfg.policy.fixed_limit_s)
    context = ContextPolicy(cfg.policy.context_limits_s)
    log.write_meta(participant=args.participant, session=args.session,
                   alerts=args.alerts, live_policy=cfg.policy.live_policy,
                   started=time.strftime("%Y-%m-%d %H:%M:%S"), road_zone=zone,
                   imu_frame=getattr(motion_src, "frame", None), config=cfg)

    # ---- Main loop -----------------------------------------------------
    frame_idx, fps, fps_t, fps_n = 0, 0.0, time.monotonic(), 0
    try:
        while not stop:
            frame = camera.read()
            if frame is None:
                break
            t = time.monotonic()
            face = tracker.process(frame, t)
            att = attention.update(face, t)
            motion = motion_src.reading()

            a_fixed = fixed.should_alert(att.off_road, att.off_road_duration, motion.state)
            a_context = context.should_alert(att.off_road, att.off_road_duration, motion.state)
            live = a_context if cfg.policy.live_policy == "context" else a_fixed

            if live:
                live_alerts.show("alert")
            elif att.reason == "no_face":
                live_alerts.show("no_face")
            elif att.off_road:
                live_alerts.show("warn")
            else:
                live_alerts.show("ok")

            key = -1
            if args.preview or args.record:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                if args.record:
                    if writer is None:
                        h, w = bgr.shape[:2]
                        writer = cv2.VideoWriter(f"{log.dir}/video.mp4",
                                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                                 cfg.camera.fps, (w, h))
                    writer.write(bgr)  # raw frames: frame N = CSV row N
                if args.preview:
                    cv2.imshow("Crash Catcher", draw_overlay(bgr, face, att, motion, live, fps))
                    key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("e"):
                event_flag["pending"] = True
            if key in STATE_KEYS and isinstance(motion_src, ManualMotion):
                motion_src.state = STATE_KEYS[key]

            event = int(event_flag["pending"])
            event_flag["pending"] = False

            log.log(t=t, frame=frame_idx, face_found=face.found,
                    head_yaw=face.head_yaw, head_pitch=face.head_pitch,
                    head_roll=face.head_roll, gaze_x=face.gaze_x, gaze_y=face.gaze_y,
                    blink_left=face.blink_left, blink_right=face.blink_right,
                    head_yaw_rel=att.head_yaw_rel, head_pitch_rel=att.head_pitch_rel,
                    gaze_x_rel=att.gaze_x_rel, gaze_y_rel=att.gaze_y_rel,
                    off_road=att.off_road, off_reason=att.reason,
                    off_duration=att.off_road_duration,
                    motion_state=motion.state, long_accel=motion.long_accel,
                    yaw_rate=motion.yaw_rate, accel_std=motion.accel_std,
                    alert_fixed=a_fixed, alert_context=a_context, alert_live=live,
                    event=event)
            frame_idx += 1

            fps_n += 1
            if t - fps_t >= 5.0:
                fps = fps_n / (t - fps_t)
                fps_t, fps_n = t, 0
                print(f"[main] {fps:.1f} fps | motion {motion.state} | "
                      f"{'OFF ROAD ' + att.reason if att.off_road else 'on road'}")
    finally:
        print(f"[main] stopping, {frame_idx} frames saved to {log.dir}")
        log.close()
        if writer:
            writer.release()
        motion_src.stop()
        alerts.close()
        tracker.close()
        camera.close()
        if button:
            button.close()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
