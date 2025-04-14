import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

X = []
y = []

data_dir = "data"

for label in os.listdir(data_dir):
    folder = os.path.join(data_dir, label)
    for file in os.listdir(folder):
        if file.endswith(".npy"):
            path = os.path.join(folder, file)
            sequence = np.load(path)
            X.append(sequence.flatten())

            # Group label logic
            if label in ["b", "c", "d", "e", "f", "g"]:
                y.append("how_are_you")
            elif label in ["h", "i", "j", "k", "l", "m"]:
                y.append("im_good")
            elif label in ["n", "o", "p", "q", "r", "s"]:
                y.append("nice_to_meet_u")
            elif label in ["t", "u", "v", "w", "x", "y", "z"]:
                y.append("Im_learning_sign_language")
            else:
                y.append(label)  # keep other labels as is (optional fallback)

X = np.array(X)
y = np.array(y)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and label encoder
joblib.dump(clf, "rf_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")
