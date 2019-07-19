# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPIO Controller for L298"""
import RPi.GPIO as GPIO
import time


# Left (BCM)
LEFT_IN3 = 14
LEFT_IN4 = 15
LEFT_ENB = 18

# Right (BCM)
RIGHT_ENA = 17
RIGHT_IN1 = 27
RIGHT_IN2 = 22

PINS = {"L": {"EN": LEFT_ENB, "IN1": LEFT_IN3, "IN2": LEFT_IN4},
        "R": {"EN": RIGHT_ENA, "IN1": RIGHT_IN1, "IN2": RIGHT_IN2}}


class MotorController(object):
    """GPIO Controller for L298."""

    def __init__(self):
        super(MotorController, self).__init__()

        GPIO.setmode(GPIO.BCM)
        for k in PINS.keys():
            GPIO.setup(PINS[k]["EN"],  GPIO.OUT)
            GPIO.setup(PINS[k]["IN1"], GPIO.OUT)
            GPIO.setup(PINS[k]["IN2"], GPIO.OUT)
        self.pwm = {"L": GPIO.PWM(PINS["L"]["EN"], 20),
                    "R": GPIO.PWM(PINS["R"]["EN"], 20)}

    def __del__(self):
        self.stop()
        self.cleanup()

    def stop(self):
        for k in PINS.keys():
            GPIO.output(PINS[k]["EN"],  GPIO.LOW)
            GPIO.output(PINS[k]["IN1"], GPIO.LOW)
            GPIO.output(PINS[k]["IN2"], GPIO.LOW)

    def forcebreak(self):
        for k in PINS.keys():
            GPIO.output(PINS[k]["EN"],  GPIO.HIGH)
            GPIO.output(PINS[k]["IN1"], GPIO.LOW)
            GPIO.output(PINS[k]["IN2"], GPIO.LOW)

    def _forward(self, k, dc=100):
        GPIO.output(PINS[k]["IN1"], GPIO.HIGH)
        GPIO.output(PINS[k]["IN2"], GPIO.LOW)
        self.pwm[k].start(dc)

    def _reverse(self, k, dc=100):
        GPIO.output(PINS[k]["IN1"], GPIO.LOW)
        GPIO.output(PINS[k]["IN2"], GPIO.HIGH)
        self.pwm[k].start(dc)

    def forward(self, duration=None):
        self._forward("L")
        self._forward("R")
        if duration:
            time.sleep(duration)

    def reverse(self, duration=None):
        self._reverse("L")
        self._reverse("R")
        if duration:
            time.sleep(duration)

    def turn_l(self, radius=0, duration=None):
        if radius > 0:
            self._forward("L", radius)
            self._forward("R")
        else:
            self._reverse("L")
            self._forward("R")
        if duration:
            time.sleep(duration)

    def turn_r(self, radius=0, duration=None):
        if radius > 0:
            self._forward("L")
            self._forward("R", radius)
        else:
            self._forward("L")
            self._reverse("R")
        if duration:
            time.sleep(duration)

    def cleanup(self):
        GPIO.cleanup()
