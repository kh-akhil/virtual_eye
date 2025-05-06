import tensorflow as tf
import cv2
import numpy as np

# Load model
model = tf.keras.models.load_model("currency_detection.h5")

# Class names
class_names = ['1Hundrednote', '2Hundrednote', '5Hundrednote', '2Thousandnote', 'Fiftynote', 'Tennote', 'TwentyNote', 'NoNote']

# Preprocessing function
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # normalize
    img = np.expand_dims(img, axis=0)  # shape becomes (1, 224, 224, 3)
    return img

# Check if ROI has sufficient color variation (helps in detecting "no note" scenarios)
def is_blank_image(roi):
    # Convert to grayscale and calculate variance
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < 100  # This threshold might need adjustment based on your camera conditions

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define region of interest (ROI)
    cv2.rectangle(frame, (100, 100), (324, 324), (0, 255, 0), 2)
    roi = frame[100:324, 100:324]  # Crop the region of interest

    # Check if there's a note in the ROI (simple threshold for detecting blank/no-note)
    if is_blank_image(roi):
        predicted_class = 'NoNote'  # If no note is detected, label it as "NoNote"
    else:
        # Preprocess image and predict
        input_img = preprocess_image(roi)
        pred = model.predict(input_img)
        # If the highest prediction probability is very low, classify as "NoNote"
        if np.max(pred) < 0.3:  # Adjust threshold based on your model's confidence
            predicted_class = 'NoNote'
        else:
            predicted_class = class_names[np.argmax(pred)]

    # Display the prediction on the frame
    cv2.putText(frame, f"Rs. {predicted_class}", (100, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the result
    cv2.imshow("Currency Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
