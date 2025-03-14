import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances
import os
import mediapipe as mp
import time

# MediaPipe Hand Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.75)
mp_drawing = mp.solutions.drawing_utils

# File and Directory Paths
GESTURE_FILE = "gesture_encodings.pkl"
IMAGE_DIR = "gesture_images"

# Ensure directory exists
os.makedirs(IMAGE_DIR, exist_ok=True)

# Load or initialize gesture encodings
try:
    with open(GESTURE_FILE, "rb") as f:
        gesture_encodings = pickle.load(f)
except FileNotFoundError:
    gesture_encodings = {}

# Function to detect a hand in an image
def detect_hand(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        return True, results.multi_hand_landmarks
    return False, None

# Function to preprocess and encode a hand gesture
def gesture_to_vector(image, fixed_length=30):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    points = np.squeeze(approx)
    if len(points.shape) != 2 or points.shape[0] < 2:
        return None
    x = np.linspace(0, len(points) - 1, fixed_length)
    interpolated_x = np.interp(x, np.arange(len(points)), points[:, 0])
    interpolated_y = np.interp(x, np.arange(len(points)), points[:, 1])
    z = np.arange(fixed_length)  # Add a Z-dimension for 3D visualization
    vector = np.stack([interpolated_x, interpolated_y, z]).flatten()
    return normalize([vector])[0]

# Function to create a 3D plot of gesture samples
def create_3d_gesture(samples, gesture_name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for vector in samples:
        x, y, z = vector[::3], vector[1::3], vector[2::3]
        ax.plot(x, y, z, label=f"Sample {len(x)}")
    ax.set_title(f"Gesture: {gesture_name}")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    image_path = os.path.join(IMAGE_DIR, f"{gesture_name}.png")
    plt.legend()
    plt.savefig(image_path)
    plt.close()
    return image_path

# Function to recognize a gesture
def recognize_gesture(vector, encodings, threshold=0.3):
    min_distance = float("inf")
    recognized_name = None
    vector = np.array(vector).reshape(1, -1)

    for name, vectors in encodings.items():
        vectors = np.array(vectors)
        if vectors.ndim != 2 or vectors.shape[1] != vector.shape[1]:
            continue
        distances = euclidean_distances(vector, vectors).min()
        if distances < min_distance and distances < threshold:
            min_distance = distances
            recognized_name = name

    confidence = 1 - (min_distance / threshold) if recognized_name else 0
    return recognized_name, min_distance, confidence

# Function to capture a hand gesture with timeout
def capture_hand_gesture():
    print("Place your hand in the blue box. Non-hand objects will be rejected.")
    print("Press 'q' to capture, or 'c' to cancel capturing.")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        box_start = (w // 3, h // 3)
        box_end = (2 * w // 3, h // 3 * 2)
        
        roi = frame[box_start[1]:box_end[1], box_start[0]:box_end[0]]
        hand_detected, landmarks = detect_hand(roi)

        # Draw bounding box
        if hand_detected:
            cv2.rectangle(frame, box_start, box_end, (0, 255, 0), 2)  # Green for valid hand
        else:
            cv2.rectangle(frame, box_start, box_end, (0, 0, 255), 2)  # Red for non-hand

        cv2.putText(frame, "Press 'q' to capture or 'c' to cancel", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Gesture Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') and hand_detected:
            cap.release()
            cv2.destroyAllWindows()
            return roi  # Return ROI if hand is detected
        elif key == ord('q'):
            print("Non-hand object detected. Please retry.")
        elif key == ord('c'):
            print("Gesture capture canceled.")
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()
    return None

# Function to validate and add gestures
def validate_and_add_gesture(gesture_name, new_samples):
    new_samples = np.array(new_samples)
    if gesture_name in gesture_encodings:
        existing_samples = np.array(gesture_encodings.get(gesture_name, []))
        if existing_samples.ndim == 2 and existing_samples.shape[1] == new_samples.shape[1]:
            gesture_encodings[gesture_name] = np.vstack([existing_samples, new_samples])
        else:
            gesture_encodings[gesture_name] = new_samples  # Replace invalid data
    else:
        gesture_encodings[gesture_name] = new_samples

def update_gesture(gesture_name, new_samples):
    new_samples = np.array(new_samples)

    if gesture_name in gesture_encodings:
        existing_samples = np.array(gesture_encodings[gesture_name])

        # Ensure existing samples have valid shape
        if existing_samples.ndim != 2:
            print(f"Gesture '{gesture_name}' has invalid encoding data. Replacing with new samples.")
            gesture_encodings[gesture_name] = new_samples
            return

        # Check for dimension mismatch
        if existing_samples.shape[1] != new_samples.shape[1]:
            print(f"Dimension mismatch for gesture '{gesture_name}': Existing samples ({existing_samples.shape[1]}) vs new samples ({new_samples.shape[1]}). Cannot update.")
            return

        # Append new samples to existing ones
        gesture_encodings[gesture_name] = np.vstack([existing_samples, new_samples])
        print(f"Gesture '{gesture_name}' successfully updated with {len(new_samples)} new samples.")
    else:
        print(f"Gesture '{gesture_name}' not found. Adding it as a new gesture.")
        gesture_encodings[gesture_name] = new_samples

# Function for real-time gesture recognition
def real_time_gesture_recognition():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        box_start = (w // 3, h // 3)
        box_end = (2 * w // 3, h // 3 * 2)
        
        roi = frame[box_start[1]:box_end[1], box_start[0]:box_end[0]]
        hand_detected, landmarks = detect_hand(roi)

        if hand_detected:
            vector = gesture_to_vector(roi)
            if vector is not None:
                name, _, confidence = recognize_gesture(vector, gesture_encodings)
                # Display gesture name at top of box
                text = f"{name} ({confidence:.2%})" if name else "Unknown"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = box_start[0] + (box_end[0] - box_start[0] - text_size[0]) // 2  # Center text
                text_y = box_start[1] - 10  # Position above box
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(frame, box_start, box_end, (0, 255, 0), 2)  # Green for valid hand
        else:
            cv2.putText(frame, "No Hand", (box_start[0] + 10, box_start[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.rectangle(frame, box_start, box_end, (0, 0, 255), 2)  # Red for non-hand

        cv2.imshow("Real-Time Gesture Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main Menu Loop
while True:
    print("\nOptions:")
    print("1. Add a new gesture")
    print("2. Search for a gesture by name")
    print("3. Update an existing gesture")
    print("4. Delete a gesture")
    print("5. List all gestures")
    print("6. Real-time gesture recognition")
    print("7. Exit")

    choice = input("Enter your choice (1/2/3/4/5/6/7): ")

    if choice == "1":
        gesture_name = input("Enter a name for this gesture: ")
        samples = []
        for i in range(5):
            roi = capture_hand_gesture()
            if roi is not None:
                vector = gesture_to_vector(roi)
                if vector is not None:
                    samples.append(vector)
                else:
                    print("Invalid gesture. Please retry.")
        if samples:
            validate_and_add_gesture(gesture_name, samples)
            image_path = create_3d_gesture(samples, gesture_name)
            with open(GESTURE_FILE, "wb") as f:
                pickle.dump(gesture_encodings, f)
            print(f"Gesture '{gesture_name}' saved successfully!")

    elif choice == "2":
        search_name = input("Enter the name of the gesture to search: ")
        if search_name in gesture_encodings:
            print(f"Gesture '{search_name}' found.")
        else:
            print(f"Gesture '{search_name}' not found.")

    elif choice == "3":
        update_name = input("Enter the name of the gesture to update: ")
        if update_name in gesture_encodings:
            print(f"Updating gesture: {update_name}. Capture 5 samples for the gesture.")
            samples = []
            for i in range(5):
                print(f"Capturing sample {i + 1} of 5. Press 'c' to cancel capturing.")
                roi = capture_hand_gesture()
                if roi is None:
                    print("Gesture capture canceled. No changes were made.")
                    break
                vector = gesture_to_vector(roi)
                if vector is not None:
                    samples.append(vector)
                else:
                    print(f"Sample {i + 1} was invalid. Skipping.")
            
            if len(samples) == 5:
                update_gesture(update_name, samples)
                with open(GESTURE_FILE, "wb") as f:
                    pickle.dump(gesture_encodings, f)
                print(f"Gesture '{update_name}' updated successfully!")
            elif samples:
                print(f"Partial samples captured ({len(samples)}). Update aborted.")
            else:
                print("No valid samples captured. Update aborted.")
        else:
            print(f"Gesture '{update_name}' not found.")

    elif choice == "4":
        delete_name = input("Enter the name of the gesture to delete: ")
        if delete_name in gesture_encodings:
            del gesture_encodings[delete_name]
            with open(GESTURE_FILE, "wb") as f:
                pickle.dump(gesture_encodings, f)
            print(f"Gesture '{delete_name}' deleted successfully.")
        else:
            print(f"Gesture '{delete_name}' not found.")

    elif choice == "5":
        print("Saved gestures:", list(gesture_encodings.keys()))

    elif choice == "6":
        real_time_gesture_recognition()

    elif choice == "7":
        with open(GESTURE_FILE, "wb") as f:
            pickle.dump(gesture_encodings, f)
        print("Exiting...")
        break

    else:
        print("Invalid choice. Please try again.")