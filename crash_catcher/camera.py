"""Camera input. Every source returns RGB frames (what MediaPipe expects)."""

import time

import cv2


class PiCamera:
    """Raspberry Pi camera module (ribbon cable) via Picamera2."""

    def __init__(self, cfg):
        from picamera2 import Picamera2
        self.cam = Picamera2()
        # Picamera2's "RGB888" actually stores pixels in B, G, R order.
        config = self.cam.create_video_configuration(
            main={"size": (cfg.width, cfg.height), "format": "RGB888"},
            controls={"FrameRate": cfg.fps})
        self.cam.configure(config)
        self.cam.start()
        time.sleep(0.5)  # let exposure settle

    def read(self):
        bgr = self.cam.capture_array()
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self):
        self.cam.stop()
        self.cam.close()


class OpenCVCamera:
    """USB webcam (by index) or a recorded video file (by path)."""

    def __init__(self, source, cfg, realtime=True):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera/video: {source}")
        self.is_file = isinstance(source, str)
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
            self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        self.file_fps = self.cap.get(cv2.CAP_PROP_FPS) or cfg.fps
        self.realtime = realtime
        self._next = None

    def read(self):
        ok, bgr = self.cap.read()
        if not ok:
            return None
        if self.is_file and self.realtime:  # play files back at their real speed
            now = time.monotonic()
            self._next = (self._next or now) + 1.0 / self.file_fps
            time.sleep(max(0.0, self._next - now))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def close(self):
        self.cap.release()


def open_camera(cfg):
    if cfg.source == "picamera":
        return PiCamera(cfg)
    if cfg.source == "usb":
        return OpenCVCamera(cfg.usb_index, cfg)
    return OpenCVCamera(cfg.source, cfg)
