import cv2
import os
import numpy as np
import mediapipe as mp
import time
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

data_dir = "./media"
os.makedirs(data_dir, exist_ok=True)

frames_to_capture = 300 # Number of frames per key
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera not opening.")
    exit()

# Function to compute orientation (Yaw, Pitch, Roll)
def compute_orientation(wrist, base, index_tip):
    vector_wrist_to_base = np.array([base[0] - wrist[0], base[1] - wrist[1], base[2] - wrist[2]])
    vector_base_to_tip = np.array([index_tip[0] - base[0], index_tip[1] - base[1], index_tip[2] - base[2]])

    vector_wrist_to_base /= np.linalg.norm(vector_wrist_to_base)
    vector_base_to_tip /= np.linalg.norm(vector_base_to_tip)

    yaw = math.atan2(vector_wrist_to_base[1], vector_wrist_to_base[0])  # Left/Right rotation
    pitch = math.asin(vector_wrist_to_base[2])  # Up/Down tilt
    roll = math.atan2(vector_base_to_tip[1], vector_base_to_tip[0])  # Finger tilt

    return [yaw, pitch, roll]

# Function to compute Euclidean distance
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

# Function to compute angles between three landmarks
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Frame capture failed")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key != 255:  # If a key is pressed
        key_char = chr(key).strip()
        if key_char.lower() == '.':
            break

        folder_path = os.path.join(data_dir, key_char)
        os.makedirs(folder_path, exist_ok=True)
        data_buffer = []

        # Countdown without freezing
        start_time = time.time()
        while time.time() - start_time < 3:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            countdown_text = f"Recording starts in {3 - int(time.time() - start_time)}..."
            cv2.putText(frame, countdown_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Hand Data Collection", frame)
            cv2.waitKey(1)

        print("Recording started!")

        frame_count = 0
        while frame_count < frames_to_capture:
            if frame_count == 1500:
                pause_start = time.time()
                while time.time() - pause_start < 5:
                    ret, frame = cap.read()
                    if not ret:
                        break

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            frame_data = []

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

                # Compute orientation
                wrist, base, index_tip = landmarks[0], landmarks[9], landmarks[8]
                orientation = compute_orientation(wrist, base, index_tip)

                # Finger Bends (Curvature Angles)
                finger_angles = []
                for joints in [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]:  
                    finger_angles.append(calculate_angle(
                        landmarks[joints[0]], landmarks[joints[1]], landmarks[joints[2]]
                    ))

                # Finger Gaps (Spacing Between Fingertips)
                finger_gaps = []
                for pair in [(8, 12), (12, 16), (16, 20)]:
                    finger_gaps.append(euclidean_distance(landmarks[pair[0]], landmarks[pair[1]]))

                # Contact Detection (Are Fingers Touching?)
                contact_distances = []
                for pair in [(8, 12), (8, 4), (12, 4), (16, 4)]:
                    contact_distances.append(euclidean_distance(landmarks[pair[0]], landmarks[pair[1]]))

                # Flatten landmark coordinates
                flattened_landmarks = [coord for landmark in landmarks for coord in landmark]

                # Append extracted features
                frame_data.extend(flattened_landmarks)  # 63 features
                frame_data.extend(orientation)  # 3 features
                frame_data.extend(finger_angles)  # 4 features (one per finger)
                frame_data.extend(finger_gaps)  # 3 features (spacing)
                frame_data.extend(contact_distances)  # 4 features (contact)

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            else:
                continue  # Skip frames with no hands

            data_buffer.append(frame_data)
            frame_count += 1

            # Display current status
            cv2.putText(frame, "Recording...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Hand Data Collection", frame)
            cv2.waitKey(1)

        np.save(os.path.join(folder_path, f"{time.time()}.npy"), np.array(data_buffer))
        print(f"Saved {frames_to_capture} frames for {key_char}")

cap.release()
cv2.destroyAllWindows()
