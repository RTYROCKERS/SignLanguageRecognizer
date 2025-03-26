import cv2
import numpy as np
import mediapipe as mp
# Load trained weights and biases
W1 = np.load("W1.npy")
W3 = np.load("W3.npy")
B1 = np.load("B1.npy")
B3 = np.load("B3.npy")
#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Define activation functions
def softmax(Z):
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    
    # Apply the Softmax function
    exp_Z = np.exp(Z_shifted)
    sum_exp_Z = np.sum(exp_Z, axis=0, keepdims=True)
    A=exp_Z / sum_exp_Z
    A=np.clip(A,1e-10,1.0)
    return A

def clip_gradients(gradients, clip_value=5.0):
    """Clip gradients to prevent explosion."""
    return [np.clip(grad, -clip_value, clip_value) for grad in gradients]

def ReLU(Z):
    return np.maximum(Z,0)
# Forward propagation
def forward_propagation(W1, W3, B1, B3, X):
    Z1 = W1.dot(X) + B1
    A1 = ReLU(Z1)
    Z3 = W3.dot(A1) + B3
    A3 = softmax(Z3)
    return Z1, Z3, A1, A3

# OpenCV video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB (MediaPipe requires RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    hand_region = np.zeros((784,1))  # Default value

    if results.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get label (Left or Right Hand)
            handedness = results.multi_handedness[hand_index].classification[0].label

            if handedness == "Left":  # Only process the right hand
                # Get bounding box coordinates
                h, w, _ = frame.shape
                x_min, y_min = w, h
                x_max, y_max = 0, 0

                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)

                # Expand bounding box slightly
                margin = 20
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(w, x_max + margin)
                y_max = min(h, y_max + margin)

                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Extract hand region
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hand_region = gray[y_min:y_max, x_min:x_max]

                # Preprocess for MNIST model
                hand_region = cv2.resize(hand_region, (28, 28)) / 255.0
                hand_region = hand_region.flatten().reshape(-1, 1)
    # Perform forward propagation
    _, _, _, A3 = forward_propagation(W1, W3, B1, B3, hand_region)
    
    # Get predicted index
    predicted_index = np.argmax(A3, axis=0)[0]

    # Map index to alphabet
    alphabet = chr(65 + predicted_index)  # Assuming A-Z mapping

    # Display prediction
    cv2.putText(frame, f"Prediction: {alphabet}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Hand Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
