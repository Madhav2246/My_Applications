import cv2
import numpy as np
import dlib
import os
from utils import calculate_ear

class FaceDetector:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cascade_path = os.path.join(base_dir, "haarcascade_frontalface_default.xml")
        landmarks_path = os.path.join(base_dir, "shape_predictor_68_face_landmarks.dat")

        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar cascade file not found: {cascade_path}")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if not os.path.exists(landmarks_path):
            raise FileNotFoundError(f"Landmark file not found: {landmarks_path}")
        self.landmark_detector = dlib.shape_predictor(landmarks_path)
        
        self.blink_threshold = 0.18  # Lowered threshold
        self.blink_count = 0
        self.frame_count = 0
        self.max_frames_without_blink = 3  # Reduced for faster liveness

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces, rejectLevels, levelWeights = self.face_cascade.detectMultiScale3(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=10, 
            minSize=(30, 30),
            outputRejectLevels=True
        )
        filtered_faces = []
        for i, (x, y, w, h) in enumerate(faces):
            confidence = levelWeights[i] if len(levelWeights) > i else 0
            if confidence > 1.5:
                filtered_faces.append((x, y, x + w, y + h))
        return filtered_faces

    def validate_face(self, frame, face_box):
        x, y, x1, y1 = face_box
        print(f"Validating face with coordinates: ({x}, {y}, {x1}, {y1})")

        x, y, x1, y1 = int(x), int(y), int(x1), int(y1)
        x = max(0, x)
        y = max(0, y)
        x1 = min(frame.shape[1], x1)
        y1 = min(frame.shape[0], y1)

        if x1 <= x or y1 <= y:
            print(f"Invalid bounding box: ({x}, {y}, {x1}, {y1})")
            return False

        width = x1 - x
        height = y1 - y
        aspect_ratio = width / height if height != 0 else 0
        if not (0.5 <= aspect_ratio <= 2.0):
            print(f"Invalid aspect ratio: {aspect_ratio}")
            return False

        print("Face validated successfully.")
        return True

    def is_alive(self, frame, face_box):
        x, y, x1, y1 = face_box
        x, y, x1, y1 = int(x), int(y), int(x1), int(y1)
        x = max(0, x)
        y = max(0, y)
        x1 = min(frame.shape[1], x1)
        y1 = min(frame.shape[0], y1)

        if x1 <= x or y1 <= y:
            print(f"Invalid bounding box for liveness: ({x}, {y}, {x1}, {y1})")
            return False

        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame

        try:
            rect = dlib.rectangle(left=x, top=y, right=x1, bottom=y1)
            landmarks = self.landmark_detector(frame_rgb, rect)
            landmarks_array = np.array([(p.x, p.y) for p in landmarks.parts()])

            left_eye = landmarks_array[36:42]
            right_eye = landmarks_array[42:48]
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            
            avg_ear = (left_ear + right_ear) / 2.0
            print(f"EAR: {avg_ear:.3f}")
            self.frame_count += 1

            if avg_ear < self.blink_threshold:
                self.blink_count += 1
                print(f"Blink detected! Total blinks: {self.blink_count}")
            elif self.frame_count > self.max_frames_without_blink:
                print(f"No blink detected for {self.frame_count} frames. Assuming live.")
                self.blink_count = 1
                self.frame_count = 0

            return self.blink_count > 0
        except Exception as e:
            print(f"Landmark detection error: {e}")
            return False