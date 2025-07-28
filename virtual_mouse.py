import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import time
import pyautogui
import pyttsx3
import threading

# --- INITIALIZATION ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Screen and Frame size for mapping
screen_width, screen_height = pyautogui.size()
frame_reduction = 100  # A frame margin to make it easier to reach screen edges

# Keyboard layout
keys = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
]

# --- VARIABLES ---
# Timers for preventing rapid/multiple actions
last_action_time = 0
action_delay = 0.5  # 500ms delay between any click or key press

# Mouse movement smoothing
smoothening = 7 # Value between 5-7 as requested
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0


# --- FUNCTIONS ---
# Speak letter using pyttsx3 in a separate thread
def speak_letter(letter):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(letter)
    engine.runAndWait()
    engine.stop()


# Button class for keyboard keys
class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text


# Create keyboard buttons
buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        posX = j * 100 + 50
        posY = i * 100 + 50
        buttonList.append(Button([posX, posY], key))


# Draw keyboard keys
def draw_keyboard(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, (x, y), (x + w, y + h), (100, 100, 100), cv2.FILLED)
        cv2.putText(img, button.text, (x + 20, y + 65),
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return img


# --- MAIN LOOP ---
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    img = draw_keyboard(img, buttonList)

    if hands:
        lmList = hands[0]['lmList']
        if lmList:
            # Get finger tip coordinates
            index_tip = lmList[8]
            middle_tip = lmList[12]

            # --- UNIFIED CONTROL LOGIC ---

            # 1. HOVERING AND MOUSE MOVEMENT
            is_on_key = False
            hovered_button = None

            for button in buttonList:
                x, y = button.pos
                w, h = button.size
                if x < index_tip[0] < x + w and y < index_tip[1] < y + h:
                    is_on_key = True
                    hovered_button = button
                    # Highlight the key being hovered over
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 20, y + 65),
                                cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)

            # If not on a key, treat as mouse movement
            if not is_on_key:
                # Indicate mouse mode with a circle on the index finger
                cv2.circle(img, (index_tip[0], index_tip[1]), 15, (0, 255, 0), cv2.FILLED)

                # Map coordinates and smoothen movement
                x_mapped = np.interp(index_tip[0], (frame_reduction, 1280 - frame_reduction), (0, screen_width))
                y_mapped = np.interp(index_tip[1], (frame_reduction, 720 - frame_reduction), (0, screen_height))
                curr_x = prev_x + (x_mapped - prev_x) / smoothening
                curr_y = prev_y + (y_mapped - prev_y) / smoothening
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y


            # 2. CLICKING / TYPING ACTION
            # Action is triggered by index and middle finger coming together
            action_dist, _, _ = detector.findDistance(index_tip[0:2], middle_tip[0:2], img)

            if action_dist < 40 and time.time() - last_action_time > action_delay:
                # If on a key, type the key
                if is_on_key and hovered_button:
                    pyautogui.press(hovered_button.text.lower())
                    threading.Thread(target=speak_letter, args=(hovered_button.text,), daemon=True).start()
                    # Visual feedback for typing
                    cv2.rectangle(img, hovered_button.pos,
                                  (hovered_button.pos[0] + hovered_button.size[0], hovered_button.pos[1] + hovered_button.size[1]),
                                  (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, hovered_button.text, (hovered_button.pos[0] + 20, hovered_button.pos[1] + 65),
                                cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)

                # If not on a key, perform a mouse click
                else:
                    pyautogui.click()
                    # Visual feedback for clicking
                    cv2.circle(img, (index_tip[0], index_tip[1]), 15, (0, 0, 255), cv2.FILLED)

                last_action_time = time.time() # Reset the timer after any action


    # --- DISPLAY ---
    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()