import cv2
import mediapipe as mp
import random
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Rock, Paper, Scissors labels
rps_gestures = {
    "rock": [0, 0, 0, 0, 0],
    "paper": [1, 1, 1, 1, 1],
    "scissors": [0, 1, 1, 0, 0]
}

# Simple finger state detector
def get_finger_states(landmarks):
    finger_states = []

    # Thumb
    finger_states.append(int(landmarks[4].x > landmarks[3].x))

    # 4 fingers
    tips = [8, 12, 16, 20]
    pip_joints = [6, 10, 14, 18]

    for tip, pip in zip(tips, pip_joints):
        finger_states.append(int(landmarks[tip].y < landmarks[pip].y))

    return finger_states

# Compare gesture to known ones
def classify_gesture(states):
    for gesture, pattern in rps_gestures.items():
        if states == pattern:
            return gesture
    return "unknown"

# Get computer move
def get_computer_move():
    return random.choice(["rock", "paper", "scissors"])

# Decide winner
def get_winner(player, computer):
    if player == computer:
        return "Draw"
    elif (player == "rock" and computer == "scissors") or \
         (player == "scissors" and computer == "paper") or \
         (player == "paper" and computer == "rock"):
        return "You Win!"
    else:
        return "Computer Wins!"

# OpenCV webcam
cap = cv2.VideoCapture(0)

last_move_time = time.time()
delay = 3  # seconds
game_ready = False
result = ""
computer_move = ""

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Mirror the frame

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_hand = hands.process(image_rgb)

    if result_hand.multi_hand_landmarks:
        for hand_landmarks in result_hand.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            finger_states = get_finger_states(hand_landmarks.landmark)
            user_move = classify_gesture(finger_states)

            current_time = time.time()
            if current_time - last_move_time > delay and user_move != "unknown":
                computer_move = get_computer_move()
                result = get_winner(user_move, computer_move)
                game_ready = True
                last_move_time = current_time

            if game_ready:
                cv2.putText(frame, f"You: {user_move}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Computer: {computer_move}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Result: {result}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        game_ready = False
        result = ""
        computer_move = ""

    cv2.imshow("Rock Paper Scissors", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
