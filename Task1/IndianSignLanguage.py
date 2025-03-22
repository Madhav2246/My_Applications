import cv2
import numpy as np
import pickle
import os
import mediapipe as mp
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import zipfile

# MediaPipe Hand Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils

# File Paths
MODEL_FILE = "isl_mlp_model.h5"
ENCODER_FILE = "isl_label_encoder.pkl"
ZIP_FILE_PATH = r"E:\ACM\My_Applications\Task1\archive (1).zip"
EXTRACT_DIR = "Indian-Sign-LanguageISL"
DATASET_DIR = os.path.join(EXTRACT_DIR, "Indian")

# Function to extract the zip file
def extract_zip_file(zip_path, extract_dir):
    if not os.path.exists(extract_dir):
        print(f"Extracting {zip_path} to {extract_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction complete.")
    else:
        print(f"Directory {extract_dir} already exists. Skipping extraction.")

# Function to extract landmarks from up to two hands
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if not results.multi_hand_landmarks:
        return None
    feature_vector = np.zeros(84)  # 84 features (42 per hand)
    for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
        start_idx = i * 42
        for j, lm in enumerate(hand_landmarks.landmark):
            feature_vector[start_idx + j * 2] = lm.x
            feature_vector[start_idx + j * 2 + 1] = lm.y
    return feature_vector

# Function to train the neural network
def train_isl_dataset(dataset_dir):
    X_train = []
    y_train = []
    
    print(f"Training from: {dataset_dir}")
    for gesture_name in os.listdir(dataset_dir):
        gesture_path = os.path.join(dataset_dir, gesture_name)
        if not os.path.isdir(gesture_path):
            continue
        image_files = [f for f in os.listdir(gesture_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Processing '{gesture_name}' with {len(image_files)} images")
        
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(gesture_path, image_file)
            image = cv2.imread(image_path)
            if image is None:
                continue
            landmarks = extract_landmarks(image)
            if landmarks is not None:
                X_train.append(landmarks)
                y_train.append(gesture_name)
            if i % 100 == 0:
                print(f"  Processed {i}/{len(image_files)} samples for '{gesture_name}'")

    if not X_train:
        print("No valid samples found.")
        return None, None

    # Encode labels (A-Z → 0-25)
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    X_train = np.array(X_train)
    
    # Define Neural Network (MLP)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(84,)),  # 84 features
        tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer 1
        tf.keras.layers.Dense(64, activation='relu'),   # Hidden layer 2
        tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')  # Output: 26 classes
    ])
    
    # Compile and train
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, validation_split=0.2)
    
    # Save model and encoder
    model.save(MODEL_FILE)
    with open(ENCODER_FILE, "wb") as f:
        pickle.dump(label_encoder, f)
    
    print("Training complete. Model and encoder saved.")
    return model, label_encoder

# Function for real-time recognition
def real_time_isl_recognition(model, label_encoder):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        
        landmarks = extract_landmarks(frame)
        if landmarks is not None:
            landmarks = landmarks.reshape(1, -1)  # Reshape for model input
            probs = model.predict(landmarks, verbose=0)[0]
            confidence = np.max(probs)
            gesture_idx = np.argmax(probs)
            gesture = label_encoder.inverse_transform([gesture_idx])[0]
            
            text = f"{gesture} ({confidence:.2%})" if confidence > 0.5 else f"Uncertain ({gesture}, {confidence:.2%})"
            color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255)
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            cv2.putText(frame, "No Hands", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("ISL Real-Time Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    extract_zip_file(ZIP_FILE_PATH, EXTRACT_DIR)
    
    if not os.path.exists(MODEL_FILE) or not os.path.exists(ENCODER_FILE):
        model, label_encoder = train_isl_dataset(DATASET_DIR)
    else:
        model = tf.keras.models.load_model(MODEL_FILE)
        with open(ENCODER_FILE, "rb") as f:
            label_encoder = pickle.load(f)
        print("Loaded existing model and encoder.")

    if model and label_encoder:
        print("Starting real-time ISL recognition. Press 'q' to quit.")
        real_time_isl_recognition(model, label_encoder)
    else:
        print("No model trained or loaded.")
