import cv2
import numpy as np
from ultralytics import YOLO
import logging
import google.generativeai as genai
from dotenv import load_dotenv
import os
from gpiozero import Button
import time
import speech_recognition as sr
from paddleocr import PaddleOCR

os.environ["GPIOZERO_PIN_FACTORY"] = "lgpio"

load_dotenv()
key=os.getenv("API_KEY")
genai.configure(api_key=key)
model = genai.GenerativeModel(model_name="gemini-2.0-flash")

yolo_model = YOLO("yolov8n.pt")
yolo_currency = YOLO("yolo_currency.pt")
logging.getLogger("ultralytics").setLevel(logging.ERROR)

ocr = PaddleOCR(use_angle_cls=True, lang='en')
logging.getLogger('ppocr').setLevel(logging.ERROR)

def speak(text, limit=150):
    chunks = [text[i:i+limit] for i in range(0, len(text), limit)]
    for chunk in chunks:
        os.system(f"espeak '{chunk}'")

logging.basicConfig(level=logging.INFO)

MODE_BUTTON_PIN = 17     # Touch sensor 1
CONFIRM_BUTTON_PIN = 27  # Touch sensor 2

mode_button = Button(MODE_BUTTON_PIN, pull_up=False)
confirm_button = Button(CONFIRM_BUTTON_PIN, pull_up=False)

# Color Ranges
color_ranges = {
    "Red":      [([0, 20, 20], [10, 255, 255]), ([170, 20, 20], [180, 255, 255])],
    "Orange":   [([11, 20, 20], [20, 255, 255])],
    "Yellow":   [([21, 20, 20], [30, 255, 255])],
    "Green":    [([40, 20, 20], [80, 255, 255])],
    "Blue":     [([90, 20, 20], [130, 255, 255])],
    "Purple":   [([130, 20, 20], [160, 255, 255])],
    "White":    [([0, 0, 200], [180, 30, 255])],
    "Black":    [([0, 0, 0], [180, 255, 50])],
    "Pink":     [([140, 20, 20], [170, 255, 255])],
    "Brown":    [([10, 20, 20], [20, 255, 200])],
    "Gray":     [([0, 0, 50], [180, 30, 200])],
    "Cyan":     [([80, 20, 20], [100, 255, 255])],
    "Magenta":  [([140, 20, 20], [160, 255, 255])],
    "Beige":    [([20, 10, 20], [40, 255, 200])],
    "Lavender": [([120, 20, 20], [160, 255, 255])],
    "Turquoise":[([80, 20, 20], [100, 255, 255])],
    "Light Blue": [([90, 20, 20], [120, 255, 255])],
    "Dark Blue":  [([110, 20, 20], [130, 255, 255])],
    "Light Green": [([40, 20, 20], [70, 255, 255])],
    "Dark Green":  [([35, 20, 20], [85, 255, 255])],
    "Light Red":   [([0, 20, 20], [5, 255, 255])],
    "Dark Red":    [([160, 20, 20], [180, 255, 255])],
    "Light Yellow": [([20, 20, 20], [35, 255, 255])],
    "Dark Yellow":  [([30, 20, 20], [40, 255, 255])],
    "Light Pink":   [([145, 20, 20], [165, 255, 255])]
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
    full_text = ""
    if result and result[0] is not None and len(result[0]) > 0:
        lines_with_boxes = result[0]

        # Step 1: Sort lines by vertical position (top to bottom)
        lines_with_boxes.sort(key=lambda x: get_line_center_y(x[0]))

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
    speak("Welcome to the Virtual Eye. Please select a mode.")
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
                    speak("Object detection mode")
                elif current_mode_index == 1:
                    print("Color detection mode")
                    speak("Color detection mode")
                elif current_mode_index == 2:
                    print("Currency detection mode")
                    speak("Currency detection mode")
                elif current_mode_index == 3:
                    print("Book reading mode")
                    speak("Book reading mode")
                elif current_mode_index == 4:
                    print("AI assistance mode")
                    speak("AI assistance mode")
                previous_mode_index = current_mode_index
                
            if active:
                if current_mode_index == 0:
                    frame, objs, objs_conf = handle_object_detection(frame)
                    objects_text = ", ".join(objs) if objs else "None"
                    print("Objects Detected:", ", ".join(objs_conf) if objs_conf else "None")
                    speak(f'The detected objects are {objects_text}')
                    active = False
                elif current_mode_index == 1:
                    _, color = handle_color_detection(frame)
                    print("Detected Color:", color)
                    speak(f'The detected color is {color}')
                    active = False
                elif current_mode_index == 2:
                    frame, note, note_conf = handle_currency_detection(frame)
                    notes_text = ", ".join(note) if note else "None"
                    print("Currency Detected:", ", ".join(note_conf) if note_conf else "None")
                    speak(f'The detected notes are {notes_text}')
                    active = False
                elif current_mode_index == 3:
                    _, text = handle_book_reading(frame)
                    print("Detected Text:\n", text)
                    if text:
                        speak(f'The detected text is {text}')
                    else:
                        speak("No text detected")
                    active = False
                elif current_mode_index == 4:    
                    text = listen_and_transcribe()
                    print("Sending to AI Assistant:", text)
                    if text:
                        response = model.generate_content(text)
                        print("AI Assistant Response:", response.text)
                        speak(response.text)
                    else:
                        print("No valid input for AI assistant.")
                        speak("Sorry I did not catch that. Please try again.")
                    active = False
                        
    except KeyboardInterrupt:
        print("\nExiting...")
        speak("Exiting the Virtual Eye.")
    finally:    
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()