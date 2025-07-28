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

# Emoji mapping
emoji_map = {
    "rock": "✊",
    "paper": "✋",
    "scissors": "✌️",
    "unknown": "❓"
}
result_emoji = {
    "You Win!": "🏆",
    "Computer Wins!": "😢",
    "Draw": "🤝"
}

# Simple finger state detector
def get_finger_states(landmarks):
    finger_states = []
    finger_states.append(int(landmarks[4].x > landmarks[3].x))

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

cap = cv2.VideoCapture(0)

last_move_time = time.time()
delay = 3
countdown_duration = 3
result = ""
computer_move = ""
user_move = ""
game_ready = False

player_score = 0
computer_score = 0

display_countdown = True
start_count_time = time.time()

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result_hand = hands.process(image_rgb)

    # Countdown
    if display_countdown:
        remaining = countdown_duration - int(time.time() - start_count_time)
        if remaining > 0:
            cv2.putText(frame, f"Get Ready: {remaining}", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)
        else:
            display_countdown = False
            last_move_time = time.time()

    elif result_hand.multi_hand_landmarks:
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
                start_count_time = current_time
                display_countdown = True

                if result == "You Win!":
                    player_score += 1
                elif result == "Computer Wins!":
                    computer_score += 1

            if game_ready:
                color = (255, 255, 255)
                if result == "You Win!":
                    color = (0, 255, 0)
                elif result == "Computer Wins!":
                    color = (0, 0, 255)
                elif result == "Draw":
                    color = (255, 0, 0)

                cv2.putText(frame, f"You: {emoji_map[user_move]} ({user_move})", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                cv2.putText(frame, f"Computer: {emoji_map[computer_move]} ({computer_move})", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                cv2.putText(frame, f"Result: {result} {result_emoji[result]}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
                cv2.putText(frame, f"Score - You: {player_score}  Computer: {computer_score}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    else:
        game_ready = False
        result = ""
        computer_move = ""

    cv2.imshow("Rock Paper Scissors", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
