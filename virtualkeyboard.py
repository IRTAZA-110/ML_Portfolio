import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time
import pyautogui
import pyttsx3
import threading

# Initialize
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Keyboard layout
keys = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
]

finalText = ""
delayCounter = 0
lastKeyPressTime = 0
delayBetweenKeys = 1  # in seconds

# Speak letter using pyttsx3 in a thread
def speak_letter(letter):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(letter)
    engine.runAndWait()
    engine.stop()

# Draw keyboard keys
def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, (x, y), (x + w, y + h), (100, 100, 100), cv2.FILLED)
        cv2.putText(img, button.text, (x + 20, y + 65),
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return img

# Button class
class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text

# Create buttons
buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        posX = j * 100 + 50
        posY = i * 100 + 50
        buttonList.append(Button([posX, posY], key))

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img)

    img = drawAll(img, buttonList)

    if hands:
        lmList = hands[0]['lmList']
        if lmList:
            index_tip = lmList[8]
            middle_tip = lmList[12]

            for button in buttonList:
                x, y = button.pos
                w, h = button.size

                # Check if finger is on the button
                if x < index_tip[0] < x + w and y < index_tip[1] < y + h:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 20, y + 65),
                                cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

                    # Measure distance between index and middle finger
                    length, _, _ = detector.findDistance((index_tip[0], index_tip[1]), (middle_tip[0], middle_tip[1]), img)


                    if length < 40 and time.time() - lastKeyPressTime > delayBetweenKeys:
                        pyautogui.press(button.text.lower())
                        threading.Thread(target=speak_letter, args=(button.text,), daemon=True).start()
                        lastKeyPressTime = time.time()

    cv2.imshow("Virtual Keyboard", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
