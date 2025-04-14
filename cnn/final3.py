import torch
import torch.nn as nn
import cv2
import mediapipe as mp
import numpy as np
from torchvision import transforms
import os
from PIL import Image
import torchvision.transforms.functional as F

class AdaptiveEqualization:
    def __call__(self, img):
        return F.equalize(img)  # Apply histogram equalization


# Load label names
DATA_DIR = "hand_dataset"
LABELS = sorted(os.listdir(DATA_DIR))

# Load the trained model
IMG_SIZE = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HandCNN(nn.Module):
    def __init__(self, num_classes):
        super(HandCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 1 channel (grayscale)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * (IMG_SIZE//4) * (IMG_SIZE//4), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = HandCNN(len(LABELS)).to(device)
model.load_state_dict(torch.load("hand_model.pth", map_location=device))
model.eval()

# Define image transformations


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    AdaptiveEqualization(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize for single-channel images
])

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

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

    padding = 20  # Add padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)

    return x_min, y_min, x_max, y_max

# Start webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        hand_bbox = None
        predicted_label = "No Hand Detected"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                #mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_bbox = get_hand_bbox(hand_landmarks, frame.shape)

                # Extract hand region
                x_min, y_min, x_max, y_max = hand_bbox
                hand_region = frame[y_min:y_max, x_min:x_max]

                if hand_region.size > 0:
                    # Resize and normalize the image
                    hand_region = cv2.resize(hand_region, (IMG_SIZE, IMG_SIZE))
                    hand_region = cv2.cvtColor(hand_region, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                    hand_region = Image.fromarray(hand_region)  # Convert to PIL image
                    hand_region = transform(hand_region).unsqueeze(0).to(device)

                    # Predict using CNN model
                    with torch.no_grad():
                        outputs = model(hand_region)
                        _, predicted_idx = torch.max(outputs, 1)
                        predicted_label = LABELS[predicted_idx.item()]

                # Draw bounding box and label
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, predicted_label, (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
