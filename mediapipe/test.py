import cv2
import os
import numpy as np
import mediapipe as mp
import math
from sklearn.ensemble import RandomForestClassifier
import joblib

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

data_dir = "./"  # Folder containing saved .npy files
os.makedirs(data_dir, exist_ok=True)

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

# Step 1: Load Training Data
X, y = [], []

print("Loading training data...")
for label in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, label)
    
    if os.path.isdir(folder_path):  
        for file in os.listdir(folder_path):
            if file.endswith(".npy"):
                data = np.load(os.path.join(folder_path, file))
                X.extend(data)
                y.extend([label] * len(data))

# Step 2: Train the Model
clf = None


model_path = "sign_language_model.pkl"

# Step 1: Load or Train the Model
if os.path.exists(model_path):
    print("Loading saved model...")
    clf = joblib.load(model_path)
    print("Model loaded successfully!")
else:
    if len(X) > 0:
        X = np.array(X)
        y = np.array(y)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        joblib.dump(clf, model_path)
        print("Model trained successfully!")
    else:
        print("No training data found. Please collect data first.")
        exit()

# Step 3: Real-Time Testing
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Camera not opening.")
    exit()

print("Real-time testing started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Frame capture failed")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    landmark_data = None  

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  
        # Extract landmark features
        landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]

        # Compute orientation
        wrist, base, index_tip = landmarks[0], landmarks[9], landmarks[8]
        orientation = compute_orientation(wrist, base, index_tip)

        # Finger Bends (Angles)
        finger_angles = []
        for joints in [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]:  
            finger_angles.append(calculate_angle(landmarks[joints[0]], landmarks[joints[1]], landmarks[joints[2]]))

        # Finger Gaps (Spacing)
        finger_gaps = []
        for pair in [(8, 12), (12, 16), (16, 20)]:
            finger_gaps.append(euclidean_distance(landmarks[pair[0]], landmarks[pair[1]]))

        # Contact Detection
        contact_distances = []
        for pair in [(8, 12), (8, 4), (12, 4), (16, 4)]:
            contact_distances.append(euclidean_distance(landmarks[pair[0]], landmarks[pair[1]]))

        # Flatten and Combine Features
        landmark_data = [coord for landmark in landmarks for coord in landmark]
        landmark_data.extend(orientation)
        landmark_data.extend(finger_angles)
        landmark_data.extend(finger_gaps)
        landmark_data.extend(contact_distances)  # Now total 77 features

        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Prediction if model is trained and hand is detected
    if clf and landmark_data:
        prediction = clf.predict([landmark_data])[0]
        cv2.putText(frame, f"Prediction: {prediction}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Sign Language Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
