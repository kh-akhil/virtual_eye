import cv2
import pytesseract
import pyttsx3

# Optional: Set Tesseract path (for Windows users)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize TTS engine
engine = pyttsx3.init()

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

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):
        print("🔍 Scanning...")

        # Preprocess image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)

        # Adaptive thresholding for better OCR
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Resize image to improve OCR accuracy
        resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Optional: Save processed image to debug OCR
        # cv2.imwrite("processed_frame.jpg", resized)

        # OCR with better config
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(resized, config=custom_config)

        print("📝 Detected Text:\n", text.strip())

        # Speak the text
        if text.strip():
            engine.say(text)
            engine.runAndWait()
        else:
            print("❗ No readable text detected.")

    elif key == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
