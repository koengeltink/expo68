import cv2
import numpy as np
import serial
import time

# --- Arduino Serial Setup ---
try:
    arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    time.sleep(2)
    print("Arduino connected.")
except:
    arduino = None
    print("Arduino not found. Running without serial output.")

# --- Camera Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open any camera")
    exit()

cap.set(3, 640)
cap.set(4, 480)

# Pixel area threshold to trigger Arduino signal
PIXEL_THRESHOLD = 10000

# Cooldown to avoid spamming Arduino (seconds)
COOLDOWN = 1.5
last_sent = {}

def send_signal(letter):
    now = time.time()
    if arduino and (letter not in last_sent or now - last_sent[letter] > COOLDOWN):
        arduino.write(letter.encode())
        last_sent[letter] = now
        print(f"Sent to Arduino: {letter}")
        time.sleep(0.05)  # small delay to let Arduino respond
        if arduino.in_waiting > 0:
            confirmation = arduino.readline().decode().strip()
            print(f"Arduino confirmed: {confirmation}")

def tint_mask(mask, bgr_color):
    colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colored[mask > 0] = bgr_color
    return colored

while True:
    # Read any unsolicited messages from Arduino
    if arduino and arduino.in_waiting > 0:
        msg = arduino.readline().decode().strip()
        print(f"Arduino says: {msg}")

    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- Color masks ---
    mask_red = (
        cv2.inRange(hsv, np.array([0,   150, 50]), np.array([10,  255, 255])) |
        cv2.inRange(hsv, np.array([170, 150, 50]), np.array([180, 255, 255]))
    )
    mask_blue   = cv2.inRange(hsv, np.array([100, 150,  50]), np.array([140, 255, 255]))
    mask_green  = cv2.inRange(hsv, np.array([40,   70,  50]), np.array([80,  255, 255]))
    mask_yellow = cv2.inRange(hsv, np.array([20,  150,  50]), np.array([35,  255, 255]))

    color_data = [
        ('R', mask_red,    (0,   0,   255), 'Red'),
        ('B', mask_blue,   (255, 0,   0  ), 'Blue'),
        ('G', mask_green,  (0,   255, 0  ), 'Green'),
        ('Y', mask_yellow, (0,   255, 255), 'Yellow'),
    ]

    # --- Window 1: Live camera stream with all bounding boxes ---
    stream = frame.copy()

    for letter, mask, box_color, label in color_data:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        total_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                total_area += area
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(stream, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(stream, label, (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

        # Send to Arduino if total detected area exceeds threshold
        if total_area > PIXEL_THRESHOLD:
            send_signal(letter)

    cv2.imshow('Camera Stream', stream)

    # --- Window 2: 2x2 filtered color masks ---
    panel_red    = tint_mask(mask_red,    (0,   0,   255))
    panel_blue   = tint_mask(mask_blue,   (255, 0,   0  ))
    panel_green  = tint_mask(mask_green,  (0,   255, 0  ))
    panel_yellow = tint_mask(mask_yellow, (0,   255, 255))

    for img, label, color in [
        (panel_red,    'Red',    (0,   0,   255)),
        (panel_blue,   'Blue',   (255, 0,   0  )),
        (panel_green,  'Green',  (0,   255, 0  )),
        (panel_yellow, 'Yellow', (0,   255, 255)),
    ]:
        cv2.putText(img, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    top    = np.hstack([panel_red,   panel_blue])
    bottom = np.hstack([panel_green, panel_yellow])
    grid   = np.vstack([top, bottom])
    grid   = cv2.resize(grid, (1280, 720))

    cv2.imshow('Color Filters (Red | Blue / Green | Yellow)', grid)

    # Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if arduino:
    arduino.close()
cv2.destroyAllWindows()
