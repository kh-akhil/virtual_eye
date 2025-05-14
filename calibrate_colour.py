import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    center = frame[frame.shape[0] // 2, frame.shape[1] // 2]
    hsv_pixel = cv2.cvtColor(np.uint8([[center]]), cv2.COLOR_BGR2HSV)[0][0]
    print("Center HSV:", hsv_pixel)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
