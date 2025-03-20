import cv2
import time
import threading
import queue
import numpy as np
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from emotion_analyzer import EmotionAnalyzer
from deepface import DeepFace

def recognition_worker(recognizer, face_roi, result_queue):
    try:
        user_id = recognizer.recognize(face_roi)
        result_queue.put(user_id)
    except Exception as e:
        print(f"Recognition worker error: {e}")
        result_queue.put("Unknown")

def emotion_worker(emotion_analyzer, face_roi, result_queue):
    try:
        emotion = emotion_analyzer.detect_emotion(face_roi)
        result_queue.put(emotion)
    except Exception as e:
        print(f"Emotion worker error: {e}")
        result_queue.put("Unknown")

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

def main():
    try:
        print("Preloading models...")
        detector = FaceDetector()
        recognizer = FaceRecognizer()
        emotion_analyzer = EmotionAnalyzer()
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        DeepFace.analyze(dummy_image, actions=['emotion'], enforce_detection=False, silent=True)
        print("Models preloaded.")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS, 30)
        print(f"Webcam resolution set to: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"Webcam FPS set to: {cap.get(cv2.CAP_PROP_FPS)}")

        cv2.namedWindow("Face Biometric System", cv2.WINDOW_AUTOSIZE)
        print("Camera window created.")

        frame_count = 0
        detection_interval = 5
        recognition_interval = 15  # Increased for performance
        emotion_update_interval = 10  # Increased for performance
        last_faces = []

        face_data = {}
        next_face_id = 0

        def get_next_frame_for_enrollment():
            ret, frame = cap.read()
            if not ret:
                return None
            cv2.imshow("Face Biometric System", frame)
            cv2.waitKey(0)  # Wait for user to press any key
            return frame

        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break

            if frame_count % detection_interval == 0:
                faces = detector.detect_faces(frame)
            else:
                faces = last_faces

            print(f"Detected {len(faces)} faces")

            new_face_data = {}
            for face in faces:
                x, y, x1, y1 = face
                best_iou = 0
                best_face_id = None

                for face_id, data in face_data.items():
                    iou = calculate_iou(face, data["box"])
                    if iou > best_iou and iou > 0.5:
                        best_iou = iou
                        best_face_id = face_id

                if best_face_id is not None:
                    new_face_data[best_face_id] = face_data[best_face_id]
                    new_face_data[best_face_id]["box"] = (x, y, x1, y1)
                else:
                    new_face_data[next_face_id] = {
                        "box": (x, y, x1, y1),
                        "user_id": "Unknown",
                        "emotion": "Unknown",
                        "recognition_queue": queue.Queue(),
                        "emotion_queue": queue.Queue(),
                        "last_seen": frame_count,
                        "recognized": False,
                        "last_recognized_frame": -recognition_interval
                    }
                    next_face_id += 1

            face_data = {k: v for k, v in new_face_data.items() if frame_count - v["last_seen"] < 30}
            last_faces = [data["box"] for data in face_data.values()]

            for face_id, data in face_data.items():
                x, y, x1, y1 = data["box"]
                data["last_seen"] = frame_count

                if not detector.validate_face(frame, (x, y, x1, y1)):
                    print(f"Face {face_id}: Not a valid face, skipping.")
                    continue

                padding = 50
                x_roi = max(0, x - padding)
                y_roi = max(0, y - padding)
                x1_roi = min(frame.shape[1], x1 + padding)
                y1_roi = min(frame.shape[0], y1 + padding)

                face_roi = frame[y_roi:y1_roi, x_roi:x1_roi]
                min_size = 50
                if face_roi.size == 0 or face_roi.shape[0] < min_size or face_roi.shape[1] < min_size:
                    print(f"Face {face_id}: Face ROI too small or empty ({face_roi.shape}), skipping.")
                    continue

                is_alive = detector.is_alive(frame, (x, y, x1, y1))
                print(f"Face {face_id} at ({x}, {y}, {x1}, {y1}) - Alive: {is_alive}")

                if is_alive:
                    # Perform recognition if not recently recognized
                    if (frame_count - data["last_recognized_frame"] >= recognition_interval):
                        threading.Thread(target=recognition_worker, args=(recognizer, face_roi, data["recognition_queue"]), daemon=True).start()
                        data["last_recognized_frame"] = frame_count

                    try:
                        new_user_id = data["recognition_queue"].get_nowait()
                        if new_user_id != "Unknown":
                            data["user_id"] = new_user_id
                            data["recognized"] = True
                    except queue.Empty:
                        pass

                    # Automatic enrollment for unknown faces
                    if data["user_id"] == "Unknown" and not data["recognized"]:
                        cv2.putText(frame, f"Please enter name for new face (Face {face_id})", 
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv2.imshow("Face Biometric System", frame)
                        print(f"Prompting for registration of Face {face_id}")
                        name = input(f"Enter name for Face {face_id}: ")
                        if name.strip():
                            success = recognizer.enroll(get_next_frame_for_enrollment, name)
                            if success:
                                data["user_id"] = name.lower()
                                data["recognized"] = True
                                data["last_recognized_frame"] = frame_count
                            else:
                                print(f"Enrollment failed for {name}. Keeping as Unknown.")
                                data["user_id"] = "Unknown"
                        else:
                            print("No name provided. Keeping as Unknown.")
                            data["user_id"] = "Unknown"

                    # Update emotion
                    if frame_count % emotion_update_interval == 0:
                        threading.Thread(target=emotion_worker, args=(emotion_analyzer, face_roi, data["emotion_queue"]), daemon=True).start()

                    try:
                        data["emotion"] = data["emotion_queue"].get_nowait()
                    except queue.Empty:
                        pass
                else:
                    continue

                # Display name and emotion
                label = f"{data['user_id']} - {data['emotion']}"
                cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 1)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("Face Biometric System", frame)
            print("Frame displayed.")
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            frame_count += 1
            print(f"Frame processing time: {time.time() - start_time:.3f} seconds")

    except Exception as e:
        print(f"Main loop error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()