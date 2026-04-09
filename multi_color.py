import cv2
import numpy as np

# Try Pi Camera first (index 0), then USB (index 1)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open any camera")
    exit()

cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- Color ranges in HSV ---

    # Red (wraps around hue, so two ranges)
    lower_red1 = np.array([0, 150, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 150, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

    # Blue
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Green
    lower_green = np.array([40, 70, 50])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Yellow
    lower_yellow = np.array([20, 150, 50])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # --- Draw bounding boxes for each color on separate frames ---

    colors = {
        'Red':    (mask_red,    (0, 0, 255)),
        'Blue':   (mask_blue,   (255, 0, 0)),
        'Green':  (mask_green,  (0, 255, 0)),
        'Yellow': (mask_yellow, (0, 255, 255)),
    }

    frames = {}
    for name, (mask, box_color) in colors.items():
        f = frame.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(f, (x, y), (x + w, y + h), box_color, 2)
        # Add label in top-left corner
        cv2.putText(f, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)
        frames[name] = f

    # --- Build 2x2 grid ---
    # Top-left: Red | Top-right: Blue
    # Bottom-left: Green | Bottom-right: Yellow
    top_row    = np.hstack([frames['Red'],   frames['Blue']])
    bottom_row = np.hstack([frames['Green'], frames['Yellow']])
    grid       = np.vstack([top_row, bottom_row])

    # Scale down so it fits on screen (each quadrant is 640x480 → grid is 1280x960)
    grid_small = cv2.resize(grid, (1280, 720))

    cv2.imshow('Color Detection (Red | Blue / Green | Yellow)', grid_small)

    # Koen was here
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()