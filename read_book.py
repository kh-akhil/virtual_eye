import cv2
import pytesseract
import pyttsx3

# Optional: Set Tesseract path for Windows
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# TTS Engine
# engine = pyttsx3.init()

# Start camera
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("📷 Press 's' to scan and read, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("⚠️ Failed to grab frame")
        continue

    height, width, _ = frame.shape

    # Define central box
    box_w, box_h = 300, 150  # size of the box (you can modify)
    cx, cy = width // 2, height // 2
    x1, y1 = cx - box_w // 2, cy - box_h // 2
    x2, y2 = cx + box_w // 2, cy + box_h // 2

    # Draw the rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Align text in the box - Press 's' to scan", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):
        print("🔍 Scanning...")

        # Crop ROI from frame
        roi = frame[y1:y2, x1:x2]

        # Preprocess image
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Resize for better OCR accuracy
        resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # OCR
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(resized, config=custom_config)

        print("📝 Detected Text:\n", text.strip())

        # Speak the text (if enabled)
        # if text.strip():
        #     engine.say(text)
        #     engine.runAndWait()
        # else:
        #     print("❗ No readable text detected.")

    elif key == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
