import streamlit as st
import cv2
import time
import numpy as np
import random
import mediapipe as mp
from predictor import predict_from_frame  # your trained model

# Game settings
GAME_TIME = 300  # 5 minutes
ALPHABET = [chr(i) for i in range(65, 91)]  # A-Z

# Fruit image paths
fruit_images = {
    "A": cv2.imread("fruits/apple.jpg"),
    "B": cv2.imread("fruits/banana.jpg"),
    "C": cv2.imread("fruits/cherry.jpg"),
    "D": cv2.imread("fruits/date.jpg"),
    "E": cv2.imread("fruits/elderberry.jpg"),
    "F": cv2.imread("fruits/fig.jpg"),
    "G": cv2.imread("fruits/grape.jpg"),
    "H": cv2.imread("fruits/honeydew.jpg"),
    "I": cv2.imread("fruits/papaya.jpg"),
    "J": cv2.imread("fruits/jackfruit.jpg"),
    "K": cv2.imread("fruits/kiwi.jpg"),
    "L": cv2.imread("fruits/lemon.jpg"),
    "M": cv2.imread("fruits/mango.jpg"),
    "N": cv2.imread("fruits/nectarine.jpg"),
    "O": cv2.imread("fruits/orange.jpg"),
    "P": cv2.imread("fruits/pineapple.jpg"),
    "Q": cv2.imread("fruits/quince.jpg"),
    "R": cv2.imread("fruits/raspberry.jpg"),
    "S": cv2.imread("fruits/strawberry.jpg"),
    "T": cv2.imread("fruits/tangerine.jpg"),
    "U": cv2.imread("fruits/ugli_fruit.jpg"),
    "V": cv2.imread("fruits/Pomegranate.jpg"),
    "W": cv2.imread("fruits/watermelon.jpg"),
    "X": cv2.imread("fruits/xigua.jpg"),
    "Y": cv2.imread("fruits/yellow_passionfruit.jpg"),
    "Z": cv2.imread("fruits/zucchini.jpg")
}

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(static_image_mode=False,
                                 max_num_hands=1,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

# Fruit drawing function
def draw_fruit(canvas, fruit_y, fruit_label):
    fruit_image = fruit_images.get(fruit_label)
    if fruit_image is not None:
        height, width = fruit_image.shape[:2]
        scale_factor = 80 / height
        resized_fruit = cv2.resize(fruit_image, (int(width * scale_factor), int(height * scale_factor)))

        center_x = canvas.shape[1] // 2
        center_y = fruit_y

        start_x = center_x - resized_fruit.shape[1] // 2
        start_y = center_y - resized_fruit.shape[0] // 2

        if start_y + resized_fruit.shape[0] <= canvas.shape[0] and start_x + resized_fruit.shape[1] <= canvas.shape[1]:
            canvas[start_y:start_y + resized_fruit.shape[0], start_x:start_x + resized_fruit.shape[1]] = resized_fruit
    return canvas

# UI
st.set_page_config(layout="wide")
st.title("🍓 Sign Language Fruit Game")
start_game = st.button("🎮 Start Game")

FRAME_WINDOW = st.empty()
gesture_col = st.sidebar.empty()
score_display = st.sidebar.empty()
timer_display = st.sidebar.empty()

if start_game:
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    score = 0

    fruit_y = 50
    fruit_speed = 5
    letter_label = random.choice(ALPHABET)
    fruit_label = random.choice(list(fruit_images.keys()))

    while time.time() - start_time < GAME_TIME:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to open camera.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

        hand_roi = frame
        label, confidence = predict_from_frame(hand_roi)

        fruit_y += fruit_speed
        if fruit_y > frame.shape[0]:
            fruit_y = 50
            letter_label = random.choice(ALPHABET)
            fruit_label = random.choice(list(fruit_images.keys()))

        # Show prompt in light green
        cv2.putText(frame, f"Sign this letter: {letter_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (144, 255, 144), 3)

        canvas = np.zeros_like(frame)
        canvas = draw_fruit(canvas, fruit_y, fruit_label)

        if label == letter_label:
            score += 1
            fruit_y = 50
            letter_label = random.choice(ALPHABET)

        combined = cv2.addWeighted(frame, 0.6, canvas, 0.4, 0)

        time_left = GAME_TIME - int(time.time() - start_time)
        score_display.markdown(f"### 🏆 Score: {score}")
        timer_display.markdown(f"### ⏳ Time Left: {time_left} sec")

        # Load gesture image for the current letter
        gesture_img = cv2.imread(f"gestures/{letter_label}.jpeg")
        if gesture_img is not None:
            gesture_img = cv2.cvtColor(gesture_img, cv2.COLOR_BGR2RGB)
            gesture_img = cv2.resize(gesture_img, (300, 300))
            gesture_col.image(gesture_img, caption=f"Sign: {letter_label}", use_column_width=False)

        FRAME_WINDOW.image(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))

    cap.release()
    st.success(f"⏰ Time's up! Final Score: {score}")
