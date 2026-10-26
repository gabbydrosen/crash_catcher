#!/usr/bin/env bash
# One-time setup on the Raspberry Pi 5. Run from the project folder:
#   bash setup_pi.sh
set -euo pipefail

echo "== System packages (camera, GPIO, I2C) =="
sudo apt-get update
sudo apt-get install -y python3-picamera2 python3-gpiozero python3-lgpio \
    python3-smbus2 i2c-tools python3-venv libegl1 libgles2

echo "== Turning on I2C for the MPU-6050 =="
sudo raspi-config nonint do_i2c 0

echo "== Python environment =="
# --system-site-packages lets the venv use the Pi's own picamera2 and gpiozero,
# which can't be installed with pip.
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "== Face model =="
mkdir -p models
if [ ! -f models/face_landmarker.task ]; then
    curl -L -o models/face_landmarker.task \
        https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
fi

echo "== Checking imports =="
.venv/bin/python - <<'EOF'
import numpy, mediapipe, cv2
print("numpy", numpy.__version__, "| mediapipe", mediapipe.__version__, "| opencv", cv2.__version__)
try:
    import picamera2  # noqa: F401
    print("picamera2 OK")
except Exception as e:
    print("picamera2 FAILED:", e)
    print("Usually a numpy version clash. Fix: .venv/bin/pip uninstall -y numpy")
import gpiozero  # noqa: F401
print("gpiozero OK")
EOF

echo
echo "Done. Next: source .venv/bin/activate, then python scripts/hardware_check.py led"
echo "(I2C was just enabled; if the IMU isn't found, reboot once.)"
