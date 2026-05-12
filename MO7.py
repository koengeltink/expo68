import sys
import serial
import threading
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.uic import loadUi

# ── UART config ──────────────────────────────────────
UART_PORT  = '/dev/ttyUSB1'  
BAUDRATE   = 9600
UI_FILE    = 'MO7.ui'
# ─────────────────────────────────────────────────────

class AckReceiver(QObject):
    ack_received = pyqtSignal(str)

    def __init__(self, ser):
        super().__init__()
        self.ser     = ser
        self.running = True

    def listen(self):
        while self.running:
            try:
                line = self.ser.readline().decode().strip()
                if line:
                    self.ack_received.emit(line)
            except Exception:
                pass

    def stop(self):
        self.running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi(UI_FILE, self)

        # open UART
        self.ser = serial.Serial(UART_PORT, baudrate=BAUDRATE, timeout=1)

        # connect slider
        self.verticalSlider.valueChanged.connect(self.on_slider_change)

        # start background receive thread
        self.receiver = AckReceiver(self.ser)
        self.receiver.ack_received.connect(self.on_ack)
        self.thread = threading.Thread(target=self.receiver.listen, daemon=True)
        self.thread.start()

    def on_slider_change(self, value):
        message = f"G {value},"
        self.ser.write(message.encode())
        print(f"Sent: {message}")          # shows in terminal for debugging         

    def on_ack(self, message):
        print(f"From PYQN: {message}")   # shows in terminal for debugging

    def closeEvent(self, event):
        self.receiver.stop()
        self.ser.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
