import cv2
import numpy as np

color_ranges = {
    "Red":      [([0, 70, 50], [10, 255, 255]), ([170, 70, 50], [180, 255, 255])],
    "Orange":   [([11, 100, 100], [20, 255, 255])],
    "Yellow":   [([21, 100, 100], [30, 255, 255])],
    "Green":    [([40, 40, 40], [80, 255, 255])],
    "Blue":     [([90, 50, 50], [130, 255, 255])],
    "Purple":   [([130, 50, 50], [160, 255, 255])],
    "White":    [([0, 0, 200], [180, 30, 255])],
    "Black":    [([0, 0, 0], [180, 255, 50])]
}

def get_dominant_color(hsv_roi):
    pixel_counts = {}
    for color, ranges in color_ranges.items():
        mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            lower_np = np.array(lower)
            upper_np = np.array(upper)
            mask |= cv2.inRange(hsv_roi, lower_np, upper_np)
        count = cv2.countNonZero(mask)
        pixel_counts[color] = count

    dominant_color = max(pixel_counts, key=pixel_counts.get)
    if pixel_counts[dominant_color] > 50:  
        return dominant_color
    else:
        return "Unknown"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    cx, cy = width // 2, height // 2
    box_size = 100
    x1, y1 = cx - box_size, cy - box_size
    x2, y2 = cx + box_size, cy + box_size

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    roi = frame[y1:y2, x1:x2]
    blurred_roi = cv2.GaussianBlur(roi, (7, 7), 0)
    hsv_roi = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2HSV)

    detected_color = get_dominant_color(hsv_roi)

    # Optional: Display center HSV value for debugging
    center_hsv = hsv_roi[box_size, box_size]
    cv2.putText(frame, f"HSV: {center_hsv.tolist()}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(frame, f"Detected: {detected_color}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "Place object in the box - Press 'q' to quit", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.imshow("High Accuracy Color Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):  # Press Space to detect color
        print(f"Detected Color: {detected_color}")
        
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

