import cv2
import numpy as np
import math
import pyttsx3
from cvzone.HandTrackingModule import HandDetector
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=r"E:\Sign Language-SIH\Model\model_unquant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
_, input_h, input_w, _ = input_details[0]['shape']

# Labels for your model
labels = ["Hello! We are team sigNOVA", "I Love You!", "Thank You!", "Help! call 112"]

# Camera setup
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Text-to-Speech setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Constants
offset = 20
imgSize = 300
sentence = ""
current_label = ""

while True:
    success, img = cap.read()
    if not success:
        continue

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Safe cropping
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size > 0:
            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Resize to model expected shape
            imgInput = cv2.resize(imgWhite, (input_w, input_h))
            imgInput = cv2.cvtColor(imgInput, cv2.COLOR_BGR2RGB)
            imgInput = np.expand_dims(imgInput, axis=0).astype(np.float32) / 255.0

            # Run inference
            interpreter.set_tensor(input_details[0]['index'], imgInput)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0]
            index = int(np.argmax(prediction))
            current_label = labels[index]

            # Draw predicted phrase
            cv2.putText(imgOutput, current_label, (x, y - 26),
                        cv2.FONT_HERSHEY_COMPLEX, 1.2, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)

    # Show sentence with current label
    display_text = sentence + (" " if sentence else "") + current_label
    cv2.putText(imgOutput, display_text, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Image", imgOutput)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == 13:  # Enter key confirms the predicted phrase
        sentence += (current_label + " ")
    elif key == 32:  # Spacebar adds space
        sentence += " "
    elif key == ord("a"):  # Read whatever is currently displayed
        engine.say(display_text.strip())
        engine.runAndWait()
    elif key == ord("c"):  # Clear the sentence
        sentence = ""
