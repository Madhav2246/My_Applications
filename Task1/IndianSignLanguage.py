import cv2
import numpy as np
import pickle
import os
import mediapipe as mp
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import zipfile

# MediaPipe Hand Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)  # Changed to 2 hands
mp_drawing = mp.solutions.drawing_utils

# File Paths
MODEL_FILE = "isl_svm_model_two_hands.pkl"  # New file to distinguish from single-hand model
SCALER_FILE = "isl_scaler_two_hands.pkl"
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

# Function to extract hand landmarks as features (supports two hands)
def extract_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if not results.multi_hand_landmarks:
        return None
    
    # Fixed-size feature vector: 42 features per hand (21 landmarks × (x, y)), 84 total for two hands
    feature_vector = np.zeros(84)  # Default to zeros if fewer than 2 hands
    
    # Process up to two hands
    for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
        start_idx = i * 42
        for j, lm in enumerate(hand_landmarks.landmark):
            feature_vector[start_idx + j * 2] = lm.x
            feature_vector[start_idx + j * 2 + 1] = lm.y
    
    return feature_vector

# Function to preprocess and train with the ISL dataset
def train_isl_dataset(dataset_dir):
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory '{dataset_dir}' not found.")
        return None, None

    X_train = []
    y_train = []
    
    print(f"Starting training from directory: {dataset_dir}")
    for gesture_name in os.listdir(dataset_dir):
        gesture_path = os.path.join(dataset_dir, gesture_name)
        if not os.path.isdir(gesture_path):
            print(f"Skipping {gesture_path} (not a directory)")
            continue
        
        image_files = [f for f in os.listdir(gesture_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_images = len(image_files)
        print(f"Processing '{gesture_name}' with {total_images} images")
        
        samples_processed = 0
        for i, image_file in enumerate(image_files):  # Use all images
            image_path = os.path.join(gesture_path, image_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load {image_path}")
                continue
            
            landmarks = extract_landmarks(image)
            if landmarks is not None:
                X_train.append(landmarks)
                y_train.append(gesture_name)
                samples_processed += 1
                if samples_processed % 100 == 0:
                    print(f"  Processed {samples_processed}/{total_images} samples for '{gesture_name}'")
            else:
                print(f"No hands detected in {image_path}")

        print(f"Trained '{gesture_name}' with {samples_processed} samples")

    if not X_train:
        print("No valid samples found for training.")
        return None, None

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train SVM classifier
    clf = svm.SVC(kernel='rbf', C=10, gamma='scale', probability=True, cache_size=1000)
    clf.fit(X_train_scaled, y_train)
    
    # Save the model and scaler
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(clf, f)
    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)
    
    print("Training complete. Model and scaler saved.")
    return clf, scaler

# Function for real-time gesture recognition
def real_time_isl_recognition(clf, scaler):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        
        landmarks = extract_landmarks(frame)
        if landmarks is not None:
            landmarks_scaled = scaler.transform([landmarks])
            gesture = clf.predict(landmarks_scaled)[0]
            confidence = clf.predict_proba(landmarks_scaled)[0].max()
            
            print(f"Predicted: {gesture}, Confidence: {confidence:.2%}")
            
            if confidence > 0.4:
                text = f"{gesture} ({confidence:.2%})"
                color = (0, 255, 0)
            else:
                text = f"Uncertain ({gesture}, {confidence:.2%})"
                color = (0, 165, 255)
            
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Draw landmarks for all detected hands
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            cv2.putText(frame, "No Hands", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print("No hands detected in frame")

        cv2.imshow("ISL Real-Time Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    # Extract the zip file
    extract_zip_file(ZIP_FILE_PATH, EXTRACT_DIR)
    
    # Force retraining
    if os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)
    if os.path.exists(SCALER_FILE):
        os.remove(SCALER_FILE)
    
    # Train the model with all samples
    clf, scaler = train_isl_dataset(DATASET_DIR)
    
    # Test the model
    if clf and scaler:
        print("Starting real-time ISL recognition. Press 'q' to quit.")
        real_time_isl_recognition(clf, scaler)
    else:
        print("No model trained. Check the dataset and try again.")