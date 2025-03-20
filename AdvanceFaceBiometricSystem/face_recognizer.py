import face_recognition
import numpy as np
import cv2
import json
import os
import dlib

class FaceRecognizer:
    def __init__(self, storage_path="database.json", max_encodings_per_person=5):
        self.storage_path = storage_path
        self.max_encodings_per_person = max_encodings_per_person
        self.embeddings = {}
        self.shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        print("Preloading face_recognition model...")
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        face_recognition.face_encodings(dummy_image)
        print("Face_recognition model preloaded.")
        self.load_encodings()

    def load_encodings(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.embeddings = {name.lower(): [np.array(encoding) for encoding in encodings] 
                                     for name, encodings in data.items()}
                print(f"Loaded encodings for {len(self.embeddings)} users from {self.storage_path}")
            except Exception as e:
                print(f"Error loading encodings: {e}")
                self.embeddings = {}
        else:
            print(f"No existing encodings found at {self.storage_path}. Starting fresh.")

    def save_encodings(self):
        try:
            data = {name: [encoding.tolist() for encoding in encodings] 
                    for name, encodings in self.embeddings.items()}
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Saved encodings for {len(self.embeddings)} users to {self.storage_path}")
        except Exception as e:
            print(f"Error saving encodings: {e}")

    def reset_database(self):
        self.embeddings = {}
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)
        print("Database reset successfully.")

    def align_face(self, image):
        if image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        face_locations = face_recognition.face_locations(image_rgb)
        if not face_locations:
            print("Align_face: No face locations found.")
            return image_rgb

        top, right, bottom, left = face_locations[0]
        face = dlib.rectangle(left=left, top=top, right=right, bottom=bottom)
        landmarks = self.shape_predictor(image_rgb, face)
        left_eye = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)], axis=0)
        right_eye = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)], axis=0)

        dX = right_eye[0] - left_eye[0]
        dY = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dY, dX))

        eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        aligned_image = cv2.warpAffine(image_rgb, M, (image_rgb.shape[1], image_rgb.shape[0]))
        return aligned_image

    def enroll(self, get_next_frame, user_id):
        user_id = user_id.lower()
        if user_id in self.embeddings and len(self.embeddings[user_id]) >= self.max_encodings_per_person:
            print(f"User {user_id} already enrolled with maximum encodings.")
            return True

        encodings_list = []
        angles = ["front", "left", "right", "up", "down"]
        for i, angle in enumerate(angles):
            print(f"Please show your face from the {angle} angle (Angle {i+1}/{len(angles)}). Press any key to capture.")
            frame = get_next_frame()
            if frame is None:
                print("Failed to capture frame for enrollment.")
                return False

            if frame.shape[2] == 3:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = frame

            image_rgb = cv2.resize(image_rgb, (0, 0), fx=0.5, fy=0.5)
            aligned_image = self.align_face(image_rgb)

            # Save debug image
            cv2.imwrite(f"debug_enroll_{user_id}_angle_{angle}.jpg", cv2.cvtColor(aligned_image, cv2.COLOR_RGB2BGR))
            print(f"Enroll: Saved debug image as debug_enroll_{user_id}_angle_{angle}.jpg")

            encodings = face_recognition.face_encodings(aligned_image)
            if not encodings:
                print(f"Failed to enroll {user_id} for {angle} angle: No face encoding found.")
                print(f"Image shape after alignment: {aligned_image.shape}")
                return False

            encodings_list.append(encodings[0])
            print(f"Enroll: Generated encoding for {user_id} ({angle} angle): {encodings[0][:5]}...")

        if user_id not in self.embeddings:
            self.embeddings[user_id] = []
        
        self.embeddings[user_id].extend(encodings_list)
        if len(self.embeddings[user_id]) > self.max_encodings_per_person:
            self.embeddings[user_id] = self.embeddings[user_id][-self.max_encodings_per_person:]
        
        print(f"Enrolled {user_id} successfully! Total encodings for {user_id}: {len(self.embeddings[user_id])}")
        self.save_encodings()
        return True

    def recognize(self, image):
        print(f"Recognize: Original image shape: {image.shape}")

        if image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        image_rgb = cv2.resize(image_rgb, (0, 0), fx=0.5, fy=0.5)
        aligned_image = self.align_face(image_rgb)

        encodings = face_recognition.face_encodings(aligned_image)
        if not encodings:
            print("Recognize: No face encoding found.")
            return "Unknown"

        encoding = encodings[0]
        print(f"Recognize: Generated encoding: {encoding[:5]}...")

        best_match = "Unknown"
        best_distance = float('inf')
        for name, stored_encodings in self.embeddings.items():
            distances = face_recognition.face_distance(stored_encodings, encoding)
            for idx, distance in enumerate(distances):
                distance = float(distance)
                print(f"Distance to {name} (encoding {idx}): {distance:.3f}")
                if distance < best_distance:
                    best_distance = distance
                    best_match = name if distance < 0.7 else "Unknown"

        if best_match != "Unknown":
            print(f"Recognized {best_match} with best distance {best_distance:.3f}")
        else:
            print("Recognize: No match found.")
        return best_match