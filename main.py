import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
import pyttsx3
import logging
import google.generativeai as genai
from dotenv import load_dotenv
import os
from gpiozero import Button
import time
import speech_recognition as sr

os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"

load_dotenv()
key=os.getenv("API_KEY")
genai.configure(api_key=key)
model = genai.GenerativeModel(model_name="gemini-2.0-flash")

yolo_model = YOLO("yolov8n.pt")
yolo_currency = YOLO("best.pt")
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# TTS 
tts_engine = pyttsx3.init()

logging.basicConfig(level=logging.INFO)

MODE_BUTTON_PIN = 17     # Touch sensor 1
CONFIRM_BUTTON_PIN = 27  # Touch sensor 2

mode_button = Button(MODE_BUTTON_PIN, pull_up=False)
confirm_button = Button(CONFIRM_BUTTON_PIN, pull_up=False)

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


def handle_book_reading(frame):
    roi = frame[ROI_TOP_LEFT[1]:ROI_BOTTOM_RIGHT[1], ROI_TOP_LEFT[0]:ROI_BOTTOM_RIGHT[0]]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(resized, config='--oem 3 --psm 6')
    return frame, text.strip()


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
    tts_engine.say("Welcome to the Virtual Eye. Please select a mode.")
    tts_engine.runAndWait()
    print("Virtual Eye started.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    modes = ["Object Detection", "Color Detection", "Currency Detection", "Book Reading", "AI Assistance"]
    current_mode_index = 0
    previous_mode_index = -1
    active = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_display = frame.copy()

            # Draw the fixed ROI as a rectangle on the camera feed
            cv2.rectangle(frame_display, ROI_TOP_LEFT, ROI_BOTTOM_RIGHT, (0, 255, 0), 2)
            #cv2.imshow("Virtual Eye", frame_display)
            #key = cv2.waitKey(1) & 0xFF  # Handle key press
            
            if mode_button.is_pressed:
                current_mode_index = (current_mode_index + 1) % len(modes)
                time.sleep(0.3)  # debounce
                
            if confirm_button.is_pressed:
                active = True
                time.sleep(0.3)  # debounce
            if current_mode_index != previous_mode_index:
                if current_mode_index == 0:
                    print("Object detection mode")
                    tts_engine.say("Object detection mode")
                    tts_engine.runAndWait()
                elif current_mode_index == 1:
                    print("Color detection mode")
                    tts_engine.say("Color detection mode")
                    tts_engine.runAndWait()
                elif current_mode_index == 2:
                    print("Currency detection mode")
                    tts_engine.say("Currency detection mode")
                    tts_engine.runAndWait()
                elif current_mode_index == 3:
                    print("Book reading mode")
                    tts_engine.say("Book reading mode")
                    tts_engine.runAndWait()
                elif current_mode_index == 4:
                    print("AI assistance mode")
                    tts_engine.say("AI assistance mode")
                    tts_engine.runAndWait()
                previous_mode_index = current_mode_index
                
            if active:
                if current_mode_index == 0:
                    frame, objs, objs_conf = handle_object_detection(frame)
                    print("Objects Detected:", ", ".join(objs_conf) if objs_conf else "None")
                    tts_engine.say(f'The detected objects are: {", ".join(objs) if objs else "None"}')
                    tts_engine.runAndWait()
                    time.sleep(1)
                    active = False
                elif current_mode_index == 1:
                    _, color = handle_color_detection(frame)
                    print("Detected Color:", color)
                    tts_engine.say(f'The detected color is: {color}')
                    tts_engine.runAndWait()
                    time.sleep(1)
                    active = False
                elif current_mode_index == 2:
                    frame, note, note_conf = handle_currency_detection(frame)
                    print("Currency Detected:", ", ".join(note_conf) if note_conf else "None")
                    tts_engine.say(f'The detected notes are: {", ".join(note) if note else "None"}')
                    time.sleep(1)
                    tts_engine.runAndWait()
                    active = False
                elif current_mode_index == 3:
                    _, text = handle_book_reading(frame)
                    print("Detected Text:\n", text)
                    tts_engine.say(f'The detected text is: {text}')
                    tts_engine.runAndWait()
                    time.sleep(1)
                    active = False
                elif current_mode_index == 4:    
                    text = listen_and_transcribe()
                    print("Sending to AI Assistant:", text)
                    if text:
                        response = model.generate_content(text)
                        print("AI Assistant Response:", response.text)
                        tts_engine.say(response.text)
                        tts_engine.runAndWait()
                        time.sleep(1)
                        active = False
                    else:
                        print("No valid input for AI assistant.")
                        tts_engine.say("Sorry I did not catch that. Please try again.")
                        tts_engine.runAndWait()
                        time.sleep(1)
                        active = False
                        
    except KeyboardInterrupt:
        print("\nExiting...")
        tts_engine.say("Exiting the Virtual Eye.")
        tts_engine.runAndWait()
    finally:    
        cap.release()
        cv2.destroyAllWindows()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
