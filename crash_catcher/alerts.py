"""RGB LED and passive buzzer.

LED colors:
  blue (pulsing)   calibrating
  green            eyes on road
  yellow           looking away, still under the limit
  red (flashing)   ALERT (buzzer beeps too)
  purple           face not found

A *passive* buzzer has no built-in tone, so it needs a square wave (PWM) at
an audible frequency. A duty cycle of 0.5 gives the loudest tone; 1.0 is
silent, so gpiozero's on()/blink() won't work for it.
"""

import threading
import time

COLORS = {
    "off": (0, 0, 0),
    "green": (0, 1, 0),
    "yellow": (1, 0.6, 0),
    "red": (1, 0, 0),
    "blue": (0, 0, 1),
    "purple": (0.6, 0, 1),
}


class GPIOAlerts:
    def __init__(self, pins):
        from gpiozero import PWMOutputDevice, RGBLED
        self.led = RGBLED(pins.led_red, pins.led_green, pins.led_blue,
                          active_high=not pins.led_common_anode)
        self.buzzer = PWMOutputDevice(pins.buzzer, frequency=pins.buzzer_hz, initial_value=0)
        self._mode = None
        self._alarm = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._beeper, daemon=True)
        self._thread.start()

    def show(self, mode: str):
        """mode: 'calibrating', 'ok', 'warn', 'alert', 'no_face', or 'off'."""
        if mode == self._mode:
            return
        self._mode = mode
        if mode == "alert":
            self.led.blink(on_time=0.15, off_time=0.15, on_color=COLORS["red"])
            self._alarm.set()
            return
        self._alarm.clear()
        if mode == "calibrating":
            self.led.pulse(fade_in_time=0.5, fade_out_time=0.5, on_color=COLORS["blue"])
        else:
            color = {"ok": "green", "warn": "yellow", "no_face": "purple"}.get(mode, "off")
            self.led.color = COLORS[color]

    def _beeper(self):
        while not self._stop.is_set():
            if self._alarm.wait(timeout=0.1):
                self.buzzer.value = 0.5
                time.sleep(0.15)
                self.buzzer.value = 0
                time.sleep(0.15)

    def beep(self, seconds=0.2):
        self.buzzer.value = 0.5
        time.sleep(seconds)
        self.buzzer.value = 0

    def close(self):
        self._stop.set()
        self._alarm.clear()
        self._thread.join(timeout=1)
        self.buzzer.value = 0
        self.led.off()
        self.led.close()
        self.buzzer.close()


class NullAlerts:
    """No LED or buzzer: used on a laptop, and for alerts-OFF drives
    (the system still logs when it *would* have alerted)."""

    def show(self, mode: str):
        pass

    def beep(self, seconds=0.2):
        pass

    def close(self):
        pass
