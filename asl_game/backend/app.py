from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import math

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# Function to compute orientation (Yaw, Pitch, Roll)
def compute_orientation(wrist, base, index_tip):
    vector_wrist_to_base = np.array([
        base['x'] - wrist['x'],
        base['y'] - wrist['y'],
        base['z'] - wrist['z']
    ])
    
    vector_base_to_tip = np.array([
        index_tip['x'] - base['x'],
        index_tip['y'] - base['y'],
        index_tip['z'] - base['z']
    ])

    vector_wrist_to_base /= np.linalg.norm(vector_wrist_to_base)
    vector_base_to_tip /= np.linalg.norm(vector_base_to_tip)

    yaw = math.atan2(vector_wrist_to_base[1], vector_wrist_to_base[0])    # Left/Right rotation
    pitch = math.asin(vector_wrist_to_base[2])                            # Up/Down tilt
    roll = math.atan2(vector_base_to_tip[1], vector_base_to_tip[0])      # Finger tilt

    return [yaw, pitch, roll]


# Function to compute Euclidean distance
def euclidean_distance(point1, point2):
    p1 = np.array([point1['x'], point1['y'], point1['z']])
    p2 = np.array([point2['x'], point2['y'], point2['z']])
    return np.linalg.norm(p1 - p2)


# Function to compute angles between three landmarks
def calculate_angle(a, b, c):
    a = np.array([a['x'], a['y'], a['z']])
    b = np.array([b['x'], b['y'], b['z']])
    c = np.array([c['x'], c['y'], c['z']])
    
    ba = a - b
    bc = c - b

    # Normalize vectors
    ba /= np.linalg.norm(ba)
    bc /= np.linalg.norm(bc)

    # Compute the angle using arccos of dot product
    cosine_angle = np.dot(ba, bc)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Safe clip for numerical stability
    return math.degrees(angle)

# Route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    landmarks = data.get('landmarks')  # Extract landmarks from the request

    # Ensure landmarks have 21 points
    if len(landmarks) != 21:
        return jsonify({"error": "Invalid number of landmarks"}), 400

    # Extracting the features (Landmark positions, orientation, angles, gaps, and distances)
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

    # Flatten and combine features (77 total)
    feature_vector = []
    for landmark in landmarks:
        feature_vector.extend([landmark['x'], landmark['y'], landmark['z']])

    feature_vector.extend(orientation)
    feature_vector.extend(finger_angles)
    feature_vector.extend(finger_gaps)
    feature_vector.extend(contact_distances)


    # Load the trained model (assuming it's saved as a .pkl file)
    model_path = "sign_language_model2.pkl"
    model = joblib.load(model_path)
    
    # Use the model to predict the sign
    prediction = model.predict([feature_vector])[0]
    print(prediction)
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)
