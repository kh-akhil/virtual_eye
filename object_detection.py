import cv2
import logging
import keyboard
from ultralytics import YOLO

logging.getLogger("ultralytics").setLevel(logging.ERROR)

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("Error: Cannot access the camera.")

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Object Detection", frame)
    else:
        print("Error reading frame")
        break

    results = model(frame)
    frame_annotated = results[0].plot()
    cv2.imshow("Object Detection", frame_annotated)
    
    if keyboard.is_pressed(" "):
        print("\nObjects in Frame:")
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                object_name = model.names[class_id]
                print(f"- {object_name} ({confidence:.2f} confidence)")
                
    if keyboard.is_pressed("q"):
        break

cap.release()
cv2.destroyAllWindows()