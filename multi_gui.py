#!/usr/bin/env python3

import cv2
import numpy as np
import serial
import time
import sys

from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer


class App(QMainWindow):

    def __init__(self):
        super().__init__()

        # =====================================================
        # LOAD UI
        # =====================================================
        uic.loadUi("gui.ui", self)

        # =====================================================
        # SETTINGS
        # =====================================================
        self.COOLDOWN = 10
        self.last_sent = {}

        self.lap_count = {
            "red": 0,
            "green": 0,
            "blue": 0,
            "purple": 0
        }

        self.total_laps = 5
        self.race_running = False

        self.letter_to_color = {
            "R": "red",
            "G": "green",
            "B": "blue",
            "P": "purple"
        }

        # Individual pixel thresholds
        self.pixel_thresholds = {
            "red":    self.slider_c_red.value(),
            "green":  self.slider_c_green.value(),
            "blue":   self.slider_c_blue.value(),
            "purple": self.slider_c_purple.value()
        }

        # Individual HSV brightness values
        self.hsv_values = {
            "red":    self.slider_b_red.value(),
            "green":  self.slider_b_green.value(),
            "blue":   self.slider_b_blue.value(),
            "purple": self.slider_b_purple.value()
        }

        # =====================================================
        # CONNECT GUI SIGNALS
        # =====================================================

        # Pixel threshold sliders
        self.slider_c_red.valueChanged.connect(
            lambda value: self.update_pixel_threshold("red", value)
        )
        self.slider_c_green.valueChanged.connect(
            lambda value: self.update_pixel_threshold("green", value)
        )
        self.slider_c_blue.valueChanged.connect(
            lambda value: self.update_pixel_threshold("blue", value)
        )
        self.slider_c_purple.valueChanged.connect(
            lambda value: self.update_pixel_threshold("purple", value)
        )

        # HSV sliders
        self.slider_b_red.valueChanged.connect(
            lambda value: self.update_hsv("red", value)
        )
        self.slider_b_green.valueChanged.connect(
            lambda value: self.update_hsv("green", value)
        )
        self.slider_b_blue.valueChanged.connect(
            lambda value: self.update_hsv("blue", value)
        )
        self.slider_b_purple.valueChanged.connect(
            lambda value: self.update_hsv("purple", value)
        )

        # Buttons
        self.start_button.clicked.connect(self.start_pressed)
        self.stop_button.clicked.connect(self.stop_pressed)
        self.reset_button.clicked.connect(self.reset_pressed)

        # =====================================================
        # SERIAL / UART
        # =====================================================
        try:
            self.arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
            time.sleep(2)
            print("Arduino connected.")

        except:
            self.arduino = None
            print("Arduino not found.")

        # =====================================================
        # CAMERA
        # =====================================================
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(1)

        if not self.cap.isOpened():
            print("Cannot open camera")
            sys.exit()

        self.cap.set(3, 640)
        self.cap.set(4, 480)

        # =====================================================
        # TIMER
        # =====================================================
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # =========================================================
    # GUI UPDATE FUNCTIONS
    # =========================================================

    def update_pixel_threshold(self, color, value):
        self.pixel_thresholds[color] = value
        print(f"{color} pixel threshold updated: {value}")

    def update_hsv(self, color, value):
        self.hsv_values[color] = value
        print(f"{color} HSV value updated: {value}")

    # =========================================================
    # UART FUNCTIONS
    # =========================================================

    def send_uart(self, message):

        if self.arduino:
            self.arduino.write((message + "\n").encode())
            print(f"Sent: {message}")

    def start_pressed(self):

        self.total_laps = self.lap_counter.value()

        self.lap_count = {
            "red": 0,
            "green": 0,
            "blue": 0,
            "purple": 0
        }

        self.last_sent = {}

        self.race_running = True

        self.send_uart(f"setlaps {self.total_laps}")

        time.sleep(0.1)

        self.send_uart("racestart")

        print(f"START pressed | laps = {self.total_laps}")

    def stop_pressed(self):

        self.race_running = False

        self.send_uart("finish")

        print("STOP pressed")

    def reset_pressed(self):

        self.race_running = False

        self.lap_count = {
            "red": 0,
            "green": 0,
            "blue": 0,
            "purple": 0
        }

        self.last_sent = {}

        self.send_uart("reset")

        print("RESET pressed")

    def send_color_signal(self, letter):

        if not self.race_running:
            return

        now = time.time()

        color = self.letter_to_color[letter]

        if (
            letter not in self.last_sent
            or now - self.last_sent[letter] > self.COOLDOWN
        ):

            if self.lap_count[color] >= self.total_laps:
                return

            self.send_uart(f"{color} stop")

            self.lap_count[color] += 1

            self.last_sent[letter] = now

            if self.lap_count[color] < self.total_laps:

                time.sleep(0.05)

                self.send_uart(f"{color} start")

            else:
                print(f"{color} finished all laps!")

    # =========================================================
    # IMAGE HELPERS
    # =========================================================

    def tint_mask(self, mask, bgr_color):

        colored = np.zeros(
            (mask.shape[0], mask.shape[1], 3),
            dtype=np.uint8
        )

        colored[mask > 0] = bgr_color

        return colored

    # =========================================================
    # MAIN CAMERA LOOP
    # =========================================================

    def update_frame(self):

        # Read serial messages
        if self.arduino and self.arduino.in_waiting > 0:

            msg = self.arduino.readline().decode().strip()

            print(f"Arduino says: {msg}")

        # Read camera frame
        ret, frame = self.cap.read()

        if not ret:
            print("Cannot receive frame")
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # =====================================================
        # HSV VALUES FROM SLIDERS
        # =====================================================

        red_v    = self.hsv_values["red"]
        green_v  = self.hsv_values["green"]
        blue_v   = self.hsv_values["blue"]
        purple_v = self.hsv_values["purple"]

        # =====================================================
        # COLOR MASKS
        # =====================================================

        mask_red = (
            cv2.inRange(
                hsv,
                np.array([0, 150, red_v]),
                np.array([10, 255, 255])
            )
            |
            cv2.inRange(
                hsv,
                np.array([170, 150, red_v]),
                np.array([180, 255, 255])
            )
        )

        mask_green = cv2.inRange(
            hsv,
            np.array([40, 70, green_v]),
            np.array([80, 255, 255])
        )

        mask_blue = cv2.inRange(
            hsv,
            np.array([100, 150, blue_v]),
            np.array([140, 255, 255])
        )

        mask_purple = cv2.inRange(
            hsv,
            np.array([125, 80, purple_v]),
            np.array([165, 255, 255])
        )

        # =====================================================
        # COLOR DATA
        # =====================================================

        color_data = [
            ("R", "red", mask_red, (0, 0, 255), "Red"),
            ("G", "green", mask_green, (0, 255, 0), "Green"),
            ("B", "blue", mask_blue, (255, 0, 0), "Blue"),
            ("P", "purple", mask_purple, (255, 0, 255), "Purple")
        ]

        # =====================================================
        # CAMERA STREAM
        # =====================================================

        stream = frame.copy()

        for uart_letter, color_name, mask, box_color, label in color_data:

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE
            )

            total_area = 0

            for cnt in contours:

                area = cv2.contourArea(cnt)

                if area > 500:

                    total_area += area

                    x, y, w, h = cv2.boundingRect(cnt)

                    cv2.rectangle(
                        stream,
                        (x, y),
                        (x + w, y + h),
                        box_color,
                        2
                    )

                    cv2.putText(
                        stream,
                        f"{label}: {int(total_area)}",
                        (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        box_color,
                        2
                    )

            if total_area > self.pixel_thresholds[color_name]:
                self.send_color_signal(uart_letter)

        cv2.imshow("Camera Stream", stream)

        # =====================================================
        # FILTER PANELS
        # =====================================================

        panel_red = self.tint_mask(mask_red, (0, 0, 255))
        panel_green = self.tint_mask(mask_green, (0, 255, 0))
        panel_blue = self.tint_mask(mask_blue, (255, 0, 0))
        panel_purple = self.tint_mask(mask_purple, (255, 0, 255))

        for img, label, color in [
            (panel_red, "Red", (0, 0, 255)),
            (panel_green, "Green", (0, 255, 0)),
            (panel_blue, "Blue", (255, 0, 0)),
            (panel_purple, "Purple", (255, 0, 255))
        ]:

            cv2.putText(
                img,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

        top = np.hstack([panel_red, panel_green])
        bottom = np.hstack([panel_blue, panel_purple])

        grid = np.vstack([top, bottom])

        grid = cv2.resize(grid, (1280, 720))

        cv2.imshow("Color Filters", grid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    # =========================================================
    # CLOSE EVENT
    # =========================================================

    def closeEvent(self, event):

        self.timer.stop()

        self.cap.release()

        if self.arduino:
            self.arduino.close()

        cv2.destroyAllWindows()

        event.accept()


# =============================================================
# START APPLICATION
# =============================================================

app = QApplication(sys.argv)

window = App()

window.show()

sys.exit(app.exec_())
