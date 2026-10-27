# Crash Catcher

Driver distraction detection on a Raspberry Pi 5. It is context-aware: the
eyes-off-road limit depends on whether the car is stopped, cruising, turning or
braking hard. It is built to grow into a predictive system that warns
*before* the eyes leave the road.

## What's here

| File | What it does |
|---|---|
| `crash_catcher/face_tracker.py` | Google MediaPipe Face Landmarker: head angle, iris position, blinks |
| `crash_catcher/attention.py` | Road-zone calibration; decides "eyes on/off road" and times each glance |
| `crash_catcher/imu.py` | MPU-6050 driver; converts readings into forward acceleration and yaw rate |
| `crash_catcher/motion.py` | Classifies motion: stopped / cruising / turning / hard braking |
| `crash_catcher/policy.py` | Alert rules: fixed 2 s baseline and context-aware limits |
| `crash_catcher/alerts.py` | RGB LED and passive buzzer |
| `crash_catcher/logger.py` | Saves every frame to `data/sessions/.../frames.csv` (your training data) |
| `crash_catcher/main.py` | Runs everything together |
| `crash_catcher/config.py` | **All settings and pin numbers** |
| `scripts/hardware_check.py` | Tests each part on its own |
| `docs/TRAINING.md` | How training your own model works, step by step |

## Wiring

Pin numbers in the config are **GPIO (BCM) numbers**; the table also lists
the physical pin on the 40-pin header. Wire with the Pi **off**.

| Part | Part pin | Pi GPIO | Physical pin |
|---|---|---|---|
| MPU-6050 | VCC | 3.3V | 1 |
| | GND | GND | 9 |
| | SDA | GPIO2 (SDA) | 3 |
| | SCL | GPIO3 (SCL) | 5 |
| | AD0, INT, XDA, XCL | leave unconnected | |
| RGB LED (common cathode) | Red leg, through 220 Ω | GPIO17 | 11 |
| | Green leg, through 220 Ω | GPIO27 | 13 |
| | Blue leg, through 220 Ω | GPIO22 | 15 |
| | Long leg (common) | GND | 14 |
| Passive buzzer | + (or S / I/O) | GPIO18 | 12 |
| | − | GND | 20 |
| | VCC (3-pin modules only) | 3.3V | 17 |
| Camera | ribbon cable | CAM/DISP port | |

Notes:
- **MPU-6050 on 3.3V, not 5V.** Most breakout boards accept either, but 3.3V
  keeps the I2C lines at a level that's safe for the Pi.
- **Common-anode LED?** If the long leg is meant for + instead of −, connect it to
  3.3V (pin 17) and set `led_common_anode = True` in `config.py`.
- **Bare buzzer (two legs, no circuit board)?** Put a 100 Ω resistor in series so it doesn't draw too much
  current from the GPIO pin. For a louder buzzer, drive it through an NPN transistor
  (e.g. 2N2222).
- **AI HAT+:** it doesn't need to come off. It connects through the PCIe
  port and passes the GPIO header through, so everything above still connects to the
  header pins on top. It isn't used yet (see below).
- **IMU mounting:** mount it flat and firm (double-sided foam tape), and note which
  printed axis (X or Y) points to the **front of the car**. Set `forward_axis`
  in `config.py` (e.g. `"+x"`, or `"-y"` if the Y arrow points backward).

## Setup on the Pi

```bash
git clone https://github.com/gabbydrosen/crash_catcher.git crash-catcher
cd crash-catcher
bash setup_pi.sh
source .venv/bin/activate
```

Then test one part at a time:

```bash
python scripts/hardware_check.py led      # cycles colors
python scripts/hardware_check.py buzzer   # three tones
python scripts/hardware_check.py imu      # live numbers; rotate and slide the board
python scripts/hardware_check.py camera   # saves camera_test.jpg
python scripts/hardware_check.py face     # live head angle and gaze
python -m pytest tests                    # logic tests, no hardware needed
```

## Running

```bash
# Development, with a monitor attached:
python -m crash_catcher.main --preview --calib-seconds 5

# Desk testing (IMU not in a car): keys 1-4 set the motion state by hand
python -m crash_catcher.main --preview --no-imu

# Participant main drive: alerts off, full calibration, save video too
python -m crash_catcher.main --participant P03 --session main --alerts off \
    --calib-seconds 60 --record

# Participant alerts-on drive
python -m crash_catcher.main --participant P03 --session alerts_on --calib-seconds 60
```

Startup sequence:
1. **IMU calibration (2 s):** keep the unit still.
2. **Road calibration (blue pulsing LED):** the driver looks at the center of the road.
3. **Beep:** monitoring starts.

LED: green = eyes on road, yellow = looking away (under the limit), flashing
red + beeping = alert, purple = face not found.

Every run creates `data/sessions/<participant>_<session>_<time>/` with
`frames.csv` (one row per frame), `meta.json` (calibration and settings) and
optionally `video.mp4`. The **fixed and context alerts are both logged on every frame**, whichever
one is driving the LED, so you can compare them on identical data.

In the preview window, press `e` to mark an event, such as when you give a distraction
prompt. You can also wire a push button between a free GPIO and GND and set
`event_button` in `config.py`.

## Do I need the AI HAT+?

**Not for this part.** MediaPipe runs on the Pi 5's CPU (expect roughly 15–30 fps
at 640×480; the `face` check prints the real number). The AI HAT+'s Hailo chip only runs
models compiled specifically for it, and MediaPipe can't use it.

It will be useful for **phone detection with YOLO** (one of the precursor cues in
Novelty 2). YOLO on the CPU alongside MediaPipe would drop the frame rate a lot,
and Raspberry Pi provides ready-made YOLO models for the Hailo chip. That's the next
integration step after this one.

## Using the simulator

In simulator sessions there's no real motion, so the motion state will come from the
game's telemetry instead of the IMU. `ManualMotion` in `imu.py` is the placeholder
for that. Once you pick the game, a small telemetry reader sets its `state`.
