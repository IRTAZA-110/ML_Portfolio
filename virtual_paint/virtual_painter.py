import cv2
import numpy as np
import os
import mediapipe as mp

# --- Configuration Parameters ---
BRUSH_THICKNESS = 15
ERASER_THICKNESS = 100
WINDOW_NAME = "AI Virtual Painter"

# Define a standard size for all icons
ICON_WIDTH, ICON_HEIGHT = 70, 70
ICON_SIZE = (ICON_WIDTH, ICON_HEIGHT)

# --- Webcam and Hand Tracking Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# --- ROBUSTNESS FIX: Ensure the correct MediaPipe class name is used ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# --- Dynamic Initialization Based on Actual Frame Size ---
success, temp_frame = cap.read()
if not success:
    raise IOError("Could not read a frame from the webcam.")
SCREEN_HEIGHT, SCREEN_WIDTH, _ = temp_frame.shape
print(f"--- Webcam running at resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT} ---")

# --- UI Setup ---
header_path = "Header"
if not os.path.exists(header_path):
    raise FileNotFoundError(
        f"The 'Header' folder was not found. Please create it and add 'blue.png', 'green.png', etc.")

# Load, resize, and store icon images
icons = {}
for im_path in os.listdir(header_path):
    key = os.path.splitext(im_path)[0]
    image = cv2.imread(f'{header_path}/{im_path}')
    if image is None: continue
    icons[key] = cv2.resize(image, ICON_SIZE)

if not icons:
    raise FileNotFoundError("No images were loaded from the 'Header' folder.")

# Create header using the actual screen width
header = np.full((ICON_HEIGHT + 20, SCREEN_WIDTH, 3), 220, dtype=np.uint8)

# Icon positions. Eraser is positioned dynamically on the right side.
icon_positions = {
    'blue': 50,
    'green': 170,
    'red': 290,
    'eraser': SCREEN_WIDTH - ICON_WIDTH - 30
}

# Place icons onto the header bar
for key, x_pos in icon_positions.items():
    if key in icons and (x_pos + ICON_WIDTH) <= SCREEN_WIDTH:
        header[10:10 + ICON_HEIGHT, x_pos:x_pos + ICON_WIDTH] = icons[key]

# Define BGR colors
BLUE, GREEN, RED = (255, 100, 0), (0, 255, 0), (0, 0, 255)
ERASER_COLOR = (0, 0, 0)
SELECTION_COLOR = (255, 255, 255)

# Initialize canvas using the actual dimensions from the camera
draw_color = BLUE
xp, yp = 0, 0
img_canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), np.uint8)


# --- Helper Functions (Unchanged) ---
def find_hand_landmarks(img, draw=True):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    landmark_list = []
    if results.multi_hand_landmarks:
        my_hand = results.multi_hand_landmarks[0]
        for id, lm in enumerate(my_hand.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmark_list.append([id, cx, cy])
        if draw:
            mp_draw.draw_landmarks(img, my_hand, mp_hands.HAND_CONNECTIONS)
    return landmark_list


def fingers_up(landmark_list):
    fingers = []
    tip_ids = [4, 8, 12, 16, 20]
    if landmark_list[tip_ids[0]][1] > landmark_list[tip_ids[0] - 1][1]:
        fingers.append(1)
    else:
        fingers.append(0)
    for id in range(1, 5):
        if landmark_list[tip_ids[id]][2] < landmark_list[tip_ids[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers.count(1)


# --- Main Application Loop ---
while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)

    lm_list = find_hand_landmarks(img, draw=False)

    # --- ROBUSTNESS FIX: Only proceed if a FULL hand is detected ---
    if len(lm_list) >= 21:
        x1, y1 = lm_list[8][1:]  # Index finger tip
        x2, y2 = lm_list[12][1:]  # Middle finger tip
        num_fingers = fingers_up(lm_list)

        if num_fingers == 2:  # Selection Mode
            xp, yp = 0, 0
            if y1 < header.shape[0]:  # Check if finger is in header
                # Simplified and safer selection logic
                if icon_positions['blue'] < x1 < icon_positions['blue'] + ICON_WIDTH:
                    draw_color = BLUE
                elif icon_positions['green'] < x1 < icon_positions['green'] + ICON_WIDTH:
                    draw_color = GREEN
                elif icon_positions['red'] < x1 < icon_positions['red'] + ICON_WIDTH:
                    draw_color = RED
                elif icon_positions['eraser'] < x1 < icon_positions['eraser'] + ICON_WIDTH:
                    draw_color = ERASER_COLOR
            cv2.rectangle(img, (x1 - 10, y1 - 15), (x2 + 10, y2 + 25), SELECTION_COLOR, cv2.FILLED)

        elif num_fingers == 1 and lm_list[8][2] < lm_list[6][2]:  # Drawing Mode
            cursor_color = (100, 100, 100) if draw_color == ERASER_COLOR else draw_color
            cv2.circle(img, (x1, y1), int(BRUSH_THICKNESS / 2), cursor_color, cv2.FILLED)
            if xp == 0 and yp == 0: xp, yp = x1, y1

            thickness = ERASER_THICKNESS if draw_color == ERASER_COLOR else BRUSH_THICKNESS
            cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, thickness)

            xp, yp = x1, y1
        else:
            xp, yp = 0, 0  # Reset if in other gesture
    else:
        xp, yp = 0, 0  # Reset if no hand is detected

    # Rendering logic (should be safe now)
    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, img_canvas)

    # Display the header
    img[0:header.shape[0], 0:SCREEN_WIDTH] = header

    # Add visual feedback border
    color_map = {'blue': BLUE, 'green': GREEN, 'red': RED, 'eraser': ERASER_COLOR}
    for key, x_pos in icon_positions.items():
        if color_map.get(key) == draw_color:
            cv2.rectangle(img, (x_pos, 10), (x_pos + ICON_WIDTH, 10 + ICON_HEIGHT), (0, 255, 0), 4)

    cv2.imshow(WINDOW_NAME, img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()