import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils

def compute_orientation(wrist, base, index_tip):
    vector_wrist_to_base = np.array(base) - np.array(wrist)
    vector_base_to_tip = np.array(index_tip) - np.array(base)
    vector_wrist_to_base /= np.linalg.norm(vector_wrist_to_base)
    vector_base_to_tip /= np.linalg.norm(vector_base_to_tip)
    yaw = math.atan2(vector_wrist_to_base[1], vector_wrist_to_base[0])
    pitch = math.asin(vector_wrist_to_base[2])
    roll = math.atan2(vector_base_to_tip[1], vector_base_to_tip[0])
    return [yaw, pitch, roll]

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

cap = cv2.VideoCapture(0)
print("Press a key (e.g., a, b, c) to capture motion. Press ESC to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # Mirror view

    # Convert and process for landmark detection
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Draw landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Motion Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    if key != 255:
        label = chr(key)
        print(f"Recording for class: {label}")
        for countdown in [3, 2, 1]:
            start_time = time.time()
            while time.time() - start_time < 1:
                ret, frame = cap.read()
                frame = cv2.flip(frame, 1)
                if not ret:
                    break
                cv2.putText(frame, f"Recording in {countdown}...", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
                cv2.imshow("Hand Motion Capture", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        sequence = []
        last_time = time.time()

        while len(sequence) < 20:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            if not ret:
                break

            now = time.time()
            if now - last_time < 0.1:
                continue
            last_time = now

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            hands_data = []

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                for hand_landmarks in results.multi_hand_landmarks:
                    keypoints = []
                    coords = []
                    for lm in hand_landmarks.landmark:
                        coords.append([lm.x, lm.y, lm.z])
                        keypoints.extend([lm.x, lm.y, lm.z])
                    wrist = coords[0]
                    base = coords[5]
                    index_tip = coords[8]
                    orientation = compute_orientation(wrist, base, index_tip)
                    angle = calculate_angle(coords[5], coords[6], coords[7])
                    distance = euclidean_distance(coords[4], coords[8])
                    keypoints.extend(orientation)
                    keypoints.append(angle)
                    keypoints.append(distance)
                    hands_data.append(keypoints)

                if len(hands_data) == 1:
                    hands_data.append([0] * len(hands_data[0]))

                combined = hands_data[0] + hands_data[1]
                sequence.append(combined)

            cv2.imshow("Hand Motion Capture", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        if len(sequence) == 20:
            folder = os.path.join("data", label)
            os.makedirs(folder, exist_ok=True)
            filename = f"{int(time.time())}.npy"
            np.save(os.path.join(folder, filename), np.array(sequence))
            print(f"Saved: {os.path.join(folder, filename)}")

cap.release()
cv2.destroyAllWindows()
hands.close()
