import cv2
import mediapipe as mp
import numpy as np
import time
import math
import joblib

clf = joblib.load("rf_model.pkl")
encoder = joblib.load("label_encoder.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
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
sequence = []
prediction_text = "N/A"
prediction_made = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    now = time.time()
    hands_data = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

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

        # Only make prediction once if not already made
        if not prediction_made and len(sequence) >= 20:
            input_seq = np.array(sequence[-20:]).flatten().reshape(1, -1)
            pred = clf.predict(input_seq)
            label = encoder.inverse_transform(pred)[0]

            if label == "how_are_you":
                prediction_text = "How are you?"
            elif label == "im_good":
                prediction_text = "I'm good"
            elif label == "nice_to_meet_u":
                prediction_text = "Nice to meet u"
            elif label == "Im_learning_sign_language":
                prediction_text = "I'm learning sign language"
            else:
                prediction_text = label

            prediction_made = True

    else:
        sequence = []
        prediction_text = "N/A"
        prediction_made = False

    cv2.putText(frame, f"Prediction: {prediction_text}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)
    
    cv2.imshow("Live Prediction", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
