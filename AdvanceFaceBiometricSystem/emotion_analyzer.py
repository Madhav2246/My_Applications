import cv2
import numpy as np
from deepface import DeepFace

class EmotionAnalyzer:
    def __init__(self):
        self.emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        print("Emotion analyzer initialized with DeepFace.")

    def detect_emotion(self, face_image):
        try:
            if face_image.size == 0:
                print("Emotion detection error: Empty face image.")
                return "Unknown"

            face_image = cv2.resize(face_image, (0, 0), fx=0.25, fy=0.25)

            result = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False, silent=True)
            if not result or len(result) == 0:
                print("Emotion detection error: No face detected by DeepFace.")
                return "Unknown"

            emotion_dict = result[0]['emotion']
            dominant_emotion = max(emotion_dict, key=emotion_dict.get)
            confidence = emotion_dict[dominant_emotion] / 100.0

            emotion = dominant_emotion.capitalize()
            if emotion not in self.emotions:
                print(f"Emotion detection error: Unrecognized emotion {emotion}.")
                return "Unknown"

            if confidence < 0.4:
                print(f"Emotion detection: Low confidence ({confidence:.3f}) for {emotion}.")
                return "Unknown"

            print(f"Detected emotion: {emotion} (Confidence: {confidence:.3f})")
            return emotion
        except Exception as e:
            print(f"Emotion detection error: {e}")
            return "Unknown"