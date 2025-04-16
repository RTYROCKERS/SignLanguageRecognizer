import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained CNN model
MODEL = load_model("asl_model.h5")  # Make sure the path is correct

# Full list of characters (A-Z)
CLASSES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

def preprocess_frame(frame):
    """
    Preprocess the frame before passing it to the model.
    """
    # Resize to model's input size
    resized = cv2.resize(frame, (64, 64))  # Resize frame to match model input size
    normalized = resized / 255.0  # Normalize pixel values to range [0, 1]
    return normalized

def predict_from_frame(frame):
    """
    Predict the character from the input frame.
    """
    processed = preprocess_frame(frame)
    prediction = MODEL.predict(np.array([processed]))  # Predict the class

    # Safety checks for prediction output
    if prediction is None or len(prediction) == 0 or len(prediction[0]) == 0:
        return "?", 0.0

    class_idx = np.argmax(prediction[0])  # Get the index of the highest predicted class

    if class_idx >= len(CLASSES):  # Check if the index is within the bounds of the class list
        return "?", 0.0

    return CLASSES[class_idx], float(prediction[0][class_idx])  # Return the predicted class and the confidence score
