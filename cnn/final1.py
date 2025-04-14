import cv2
import mediapipe as mp
import os
import numpy as np


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

DATA_DIR = "hand_dataset"
os.makedirs(DATA_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
current_label = None
images_saved = 0
MAX_IMAGES = 100  # Save 100 images per letter
IMG_SIZE = (128, 128)  # Resize to 128x128

def get_hand_bbox(hand_landmarks, frame_shape):
    h, w, _ = frame_shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0

    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        x_min = min(x, x_min)
        y_min = min(y, y_min)
        x_max = max(x, x_max)
        y_max = max(y, y_max)

    padding = 20
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)

    return x_min, y_min, x_max, y_max

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    print("Press any key (A-Z, 0-9) to start capturing 100 images. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        hand_bbox = None

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
               # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_bbox = get_hand_bbox(hand_landmarks, frame.shape)

                # Draw bounding box
                x_min, y_min, x_max, y_max = hand_bbox
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        cv2.putText(frame, f"Current Label: {current_label} ({images_saved}/{MAX_IMAGES})", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.imshow("Hand Tracking", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('.'):
            break
        elif images_saved >= MAX_IMAGES:
            print(f"Finished capturing 100 images for {current_label}. Press a new key.")
            current_label = None
            images_saved = 0

        elif not current_label and (48 <= key <= 57 or 65 <= key <= 90 or 97 <= key <= 122):
            current_label = chr(key).upper()
            print(f"Started capturing images for: {current_label}")

        if current_label and hand_bbox and images_saved < MAX_IMAGES:
            label_dir = os.path.join(DATA_DIR, current_label)
            os.makedirs(label_dir, exist_ok=True)

            x_min, y_min, x_max, y_max = hand_bbox
            hand_region = frame[y_min:y_max, x_min:x_max]

            if hand_region.size > 0:
                hand_region = cv2.resize(hand_region, IMG_SIZE)  # Resize to 128x128

                img_count = len(os.listdir(label_dir))
                img_path = os.path.join(label_dir, f"{img_count}.jpg")
                cv2.imwrite(img_path, hand_region)
                images_saved += 1
                print(f"Saved {img_path}")

cap.release()
cv2.destroyAllWindows()
