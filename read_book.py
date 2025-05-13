import cv2
from paddleocr import PaddleOCR
import pyttsx3

ocr = PaddleOCR(use_angle_cls=True, lang='en')
#engine = pyttsx3.init()
#engine.setProperty('rate', 150)

cap = cv2.VideoCapture(0)
print("Press 's' to scan and read, 'q' to quit.")

def get_line_center_y(box):
    return (box[0][1] + box[2][1]) / 2  # Average y of top-left and bottom-right

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Live Feed - Book Reader", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
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

            print("\nFormatted Text:\n", full_text)
            #engine.say(full_text)
            #engine.runAndWait()

        else:
            print("No text detected.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
