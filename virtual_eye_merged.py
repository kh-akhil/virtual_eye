import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import logging
from google import genai
from dotenv import load_dotenv
import os
from paddleocr import PaddleOCR

load_dotenv()
key=os.getenv("API_KEY")

client = genai.Client(api_key=key)

# Load YOLO and currency detection models
yolo_model = YOLO("yolov8n.pt")
yolo_currency = YOLO("yolo_currency.pt")
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# TTS (optional)
tts_engine = pyttsx3.init()

ocr = PaddleOCR(use_angle_cls=True, lang='en')
logging.getLogger('ppocr').setLevel(logging.ERROR)

# Logging
logging.basicConfig(level=logging.INFO)

# Color Ranges
color_ranges = {
    "Red":      [([0, 70, 50], [10, 255, 255]), ([170, 70, 50], [180, 255, 255])],
    "Orange":   [([11, 100, 100], [20, 255, 255])],
    "Yellow":   [([21, 100, 100], [30, 255, 255])],
    "Green":    [([40, 40, 40], [80, 255, 255])],
    "Blue":     [([90, 50, 50], [130, 255, 255])],
    "Purple":   [([130, 50, 50], [160, 255, 255])],
    "White":    [([0, 0, 200], [180, 30, 255])],
    "Black":    [([0, 0, 0], [180, 255, 50])],

    # Additional colors
    "Pink":     [([140, 50, 50], [170, 255, 255])],
    "Brown":    [([10, 100, 20], [20, 255, 200])],
    "Gray":     [([0, 0, 50], [180, 30, 200])],
    "Cyan":     [([80, 50, 50], [100, 255, 255])],
    "Magenta":  [([140, 50, 50], [160, 255, 255])],
    "Beige":    [([20, 10, 100], [40, 255, 200])],
    "Lavender": [([120, 20, 20], [160, 255, 255])],
    "Turquoise": [([80, 50, 50], [100, 255, 255])],

    # Adding more shades
    "Light Blue": [([90, 50, 50], [120, 255, 255])],
    "Dark Blue":  [([110, 100, 50], [130, 255, 255])],
    "Light Green": [([40, 50, 50], [70, 255, 255])],
    "Dark Green":  [([35, 70, 50], [85, 255, 255])],
    "Light Red":   [([0, 70, 50], [5, 255, 255])],
    "Dark Red":    [([160, 70, 50], [180, 255, 255])],
    "Light Yellow": [([20, 50, 50], [35, 255, 255])],
    "Dark Yellow":  [([30, 100, 100], [40, 255, 255])],
    "Light Pink":   [([145, 50, 50], [165, 255, 255])]
}

# Define the fixed ROI
ROI_WIDTH = 224
ROI_HEIGHT = 224

# Coordinates for the center of the screen (adjust this if needed)
ROI_CENTER_X = 640  # Horizontal center of the frame (you can adjust it)
ROI_CENTER_Y = 360  # Vertical center of the frame (you can adjust it)

# ROI corner coordinates
ROI_TOP_LEFT = (ROI_CENTER_X - ROI_WIDTH // 2, ROI_CENTER_Y - ROI_HEIGHT // 2)
ROI_BOTTOM_RIGHT = (ROI_CENTER_X + ROI_WIDTH // 2, ROI_CENTER_Y + ROI_HEIGHT // 2)

def preprocess_image(image, size=(224, 224)):
    image = cv2.resize(image, size)
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def get_dominant_color(hsv_roi):
    pixel_counts = {}
    for color, ranges in color_ranges.items():
        mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask |= cv2.inRange(hsv_roi, np.array(lower), np.array(upper))
        pixel_counts[color] = cv2.countNonZero(mask)

    dominant_color = max(pixel_counts, key=pixel_counts.get)
    return dominant_color if pixel_counts[dominant_color] > 50 else "Unknown"

def handle_object_detection(frame):
    results = yolo_model(frame)
    frame_annotated = results[0].plot()
    detected_objects = []
    detected_objects_conf = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            object_name = yolo_model.names[class_id]
            detected_objects.append(object_name)
            detected_objects_conf.append(f"{object_name} ({confidence:.2f})")
    return frame_annotated, detected_objects, detected_objects_conf

def handle_color_detection(frame):
    roi = frame[ROI_TOP_LEFT[1]:ROI_BOTTOM_RIGHT[1], ROI_TOP_LEFT[0]:ROI_BOTTOM_RIGHT[0]]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    dominant_color = get_dominant_color(hsv_roi)
    return frame, dominant_color

def handle_currency_detection(frame):
    results = yolo_currency(frame)
    frame_annotated = results[0].plot()
    detected_curr = []
    detected_curr_conf = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            object_name = yolo_currency.names[class_id]
            detected_curr.append(object_name)
            detected_curr_conf.append(f"{object_name} ({confidence:.2f})")
    return frame_annotated, detected_curr, detected_curr_conf


def get_line_center_y(box):
    return (box[0][1] + box[2][1]) / 2 

def handle_book_reading(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = ocr.ocr(image_rgb, cls=True)
    if result and result[0] is not None and len(result[0]) > 0:
            lines_with_boxes = result[0]

            # Step 1: Sort lines by vertical position (top to bottom)
            lines_with_boxes.sort(key=lambda x: get_line_center_y(x[0]))

            full_text = ""
            last_y = None

            for i, (box, (text, _)) in enumerate(lines_with_boxes):
                curr_y = get_line_center_y(box)

                if last_y is not None:
                    # If the current line is vertically far from the last line → new paragraph
                    if abs(curr_y - last_y) > 40:  # Adjust this threshold as needed
                        full_text += "\n\n"
                    else:
                        full_text += " "
                full_text += text
                last_y = curr_y
    return frame, full_text

import speech_recognition as sr

def listen_and_transcribe():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Listening... (Speak now)")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return None


    

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    print("Press 1-5 to change mode. 'q' to quit.")
    mode = "object"

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_display = frame.copy()

        # Draw the fixed ROI as a rectangle on the camera feed
        cv2.rectangle(frame_display, ROI_TOP_LEFT, ROI_BOTTOM_RIGHT, (0, 255, 0), 2)

        cv2.imshow("Virtual Eye", frame_display)

        key = cv2.waitKey(1) & 0xFF  # Handle key press

        if key == ord('1'):
            mode = "object"
            print("Object detection mode")
            tts_engine.say("Object detection mode")
            tts_engine.runAndWait()
        elif key == ord('2'):
            mode = "color"
            print("Color detection mode")
            tts_engine.say("Color detection mode")
            tts_engine.runAndWait()
        elif key == ord('3'):
            mode = "currency"
            print("Currency detection mode")
            tts_engine.say("Currency detection mode")
            tts_engine.runAndWait()
        elif key == ord('4'):
            mode = "book"
            print("Book reading mode")
            tts_engine.say("Book reading mode")
            tts_engine.runAndWait()
        elif key == ord('5'):
            mode = "AI assistant"
            print("AI assistant mode")
            tts_engine.say("AI assistant mode")
            tts_engine.runAndWait()
        elif key == ord('d'):
            if mode == "object":
                frame, objs, objs_conf = handle_object_detection(frame)
                print("Objects Detected:", ", ".join(objs_conf) if objs_conf else "None")
                tts_engine.say(f'The detected objects are: {", ".join(objs) if objs else "None"}')
                tts_engine.runAndWait()
            elif mode == "color":
                _, color = handle_color_detection(frame)
                print("Detected Color:", color)
                tts_engine.say(f'The detected color is: {color}')
                tts_engine.runAndWait()
            elif mode == "currency":
                frame, note, note_conf = handle_currency_detection(frame)
                print("Currency Detected:", ", ".join(note_conf) if note_conf else "None")
                tts_engine.say(f'The detected notes are: {", ".join(note) if note else "None"}')
                tts_engine.runAndWait()
            elif mode == "book":
                _, text = handle_book_reading(frame)
                print("Detected Text:\n", text)
                tts_engine.say(f'The detected text is: {text}')
                tts_engine.runAndWait()
            elif mode == "AI assistant":    
                text = listen_and_transcribe()
                if text:
                    response = client.models.generate_content(model="gemini-2.0-flash", contents=text)
                    print("AI Assistant Response:", response.text)
                    tts_engine.say(response.text)
                    tts_engine.runAndWait()
                else:
                    print("No valid input for AI assistant.")
                    tts_engine.say("No valid input for AI assistant.")
                    tts_engine.runAndWait()
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
