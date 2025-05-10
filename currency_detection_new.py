import cv2
from ultralytics import YOLO
import logging

logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Load the trained model
model = YOLO("best.pt")

# OpenCV video capture from webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    results = model(frame)

    # Draw results on the frame
    annotated_frame = results[0].plot()

    # Display the frame
    cv2.imshow("Live Detection", annotated_frame)
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            object_name = model.names[class_id]
            print(f"- {object_name} ({confidence:.2f} confidence)")

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release and destroy
cap.release()
cv2.destroyAllWindows()
