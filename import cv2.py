import cv2
import numpy as np

# Try Pi Camera first (index 0), then USB (index 1)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Cannot open any camera")
    exit()

# Optional: reduce resolution for performance
cap.set(3, 640)  # width
cap.set(4, 480)  # height

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    # Convert BGR frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define blue color range (adjust if needed)
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])

    # Create mask for blue
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours of the blue areas
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:  # ignore tiny areas
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the original frame and mask
    cv2.imshow('Camera Stream', frame)
    cv2.imshow('Mask', mask)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()