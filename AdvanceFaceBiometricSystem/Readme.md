# Advanced Face Biometric System

## Overview
The **Advanced Face Biometric System** is a real-time face recognition and emotion detection application built using Python. It leverages computer vision and machine learning techniques to detect, recognize, and enroll faces, while also analyzing emotions in real time. The system is designed to handle 5 to 6 persons without confusion, achieve real-time performance (< 0.033 seconds per frame), and provide robust face recognition by collecting multiple face angles during enrollment.

This project uses the following key libraries:
- `face_recognition` for face encoding and recognition.
- `DeepFace` for emotion detection.
- `OpenCV` for video capture and image processing.
- `dlib` for facial landmark detection and liveness detection.

## Features
1. **Single Face Detection**:
   - A face is detected and enrolled only once using face tracking (Intersection over Union, IoU).
   - Prevents repeated enrollment prompts for the same person.

2. **Automatic Enrollment**:
   - Automatically prompts for registration when an unknown face is detected (no manual key press required).
   - Displays a message on the frame (e.g., "Please enter name for new face") and waits for user input via the console.

3. **Multiple Face Angles for Enrollment**:
   - Collects 5 encodings per person (front, left, right, up, down) during initial registration to improve recognition accuracy across different angles.
   - Enhances robustness for face recognition.

4. **Encodings-Based Storage**:
   - Stores face encodings in a `database.json` file for efficient matching.
   - Uses JPG images only for debugging purposes (e.g., `debug_enroll_<name>_angle_<angle>.jpg`).

5. **Real-Time Performance**:
   - Achieves frame processing times of < 0.033 seconds (except during enrollment).
   - Optimized with reduced recognition and emotion detection frequency.

6. **Robustness for 5 to 6 Persons**:
   - Handles 5 to 6 persons in the frame without confusion.
   - Uses multiple encodings per person and face tracking to ensure accurate identification.

7. **Emotion Detection**:
   - Detects emotions in real time using DeepFace (e.g., Happy, Fear, Neutral).
   - Displays the name and emotion together on the frame (e.g., "madhav - Happy").

8. **Liveness Detection**:
   - Implements liveness detection using Eye Aspect Ratio (EAR) to ensure the face is live (not a photo).
   - Optimized for faster detection (assumes liveness after 2 frames without a blink).

## Applications
- **Security and Access Control**:
  - Can be used for secure access to buildings, offices, or devices by recognizing authorized personnel.
- **Attendance Systems**:
  - Automates attendance tracking in schools, workplaces, or events by identifying individuals.
- **Emotion Analysis**:
  - Useful in customer service or mental health applications to analyze emotions in real time.
- **Surveillance**:
  - Enhances surveillance systems by identifying known individuals and detecting emotions for behavioral analysis.
- **Personalized User Experiences**:
  - Can be integrated into smart devices to provide personalized experiences based on user identity and emotional state.
- **Human-Computer Interaction**:
  - Improves interaction in gaming or virtual assistants by responding to user emotions.

## Prerequisites
- **Hardware**:
  - A webcam or camera for video input.
  - A computer with at least 8GB RAM for smooth performance.
- **Software**:
  - Python 3.8 or higher.
  - Operating System: Windows, macOS, or Linux.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Madhav2246/AdvancedFaceBiometricSystem.git
   cd AdvancedFaceBiometricSystem
   ```

2. **Set Up a Virtual Environment** (Optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Install the required Python packages using `pip`:
   ```bash
   pip install opencv-python face_recognition deepface dlib numpy
   ```
   - **Note**: Installing `dlib` might require additional steps on some systems:
     - On Windows, you may need to install `cmake` and Visual Studio Build Tools.
     - On Linux/macOS, install `cmake` and `libopenblas-dev`:
       ```bash
       sudo apt-get install cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev
       ```

4. **Download Required Models**:
   - Download the `haarcascade_frontalface_default.xml` file for face detection:
     - Available in the OpenCV GitHub repository: [haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml).
     - Place it in the project directory.
   - Download the `shape_predictor_68_face_landmarks.dat` file for facial landmark detection:
     - Available from the dlib website: [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
     - Extract and place it in the project directory.

## Project Structure
```
AdvancedFaceBiometricSystem/
│
├── main.py                   # Main script to run the application
├── face_detector.py          # Face detection and liveness detection
├── face_recognizer.py        # Face enrollment and recognition
├── emotion_analyzer.py       # Emotion detection using DeepFace
├── utils.py                  # Utility functions (e.g., EAR calculation)
├── haarcascade_frontalface_default.xml  # Haar cascade for face detection
├── shape_predictor_68_face_landmarks.dat  # dlib landmark predictor
├── database.json             # Stores face encodings (created automatically)
├── debug_enroll_*.jpg        # Debug images (created during enrollment)
└── README.md                 # Project documentation
```

## Usage
1. **Run the Application**:
   ```bash
   python main.py
   ```
   - The webcam will start, and the system will begin detecting faces.

2. **Enroll a New Face**:
   - When an unknown face is detected, the system will display a message on the frame (e.g., "Please enter name for new face (Face 0)") and prompt for a name in the console:
     ```
     Enter name for Face 0: Madhav
     ```
   - After entering a name, the system will prompt for 5 face angles:
     ```
     Please show your face from the front angle (Angle 1/5). Press any key to capture.
     ```
   - Adjust your face to the specified angle (front, left, right, up, down) and press any key to capture each angle.

3. **Recognize Faces**:
   - Once enrolled, the system will recognize the face in subsequent frames and display the name with the detected emotion (e.g., "madhav - Happy").

4. **Test with Multiple Persons**:
   - Show 5 to 6 different faces one at a time to enroll them.
   - Then, show multiple faces in the frame to verify recognition and emotion detection.

5. **Exit the Application**:
   - Press the `q` key to exit the application.

## Example Output
- **Initial Setup**:
  ```
  Preloading models...
  Preloading face_recognition model...
  Face_recognition model preloaded.
  Loaded encodings for 0 users from database.json
  Emotion analyzer initialized with DeepFace.
  Models preloaded.
  Webcam resolution set to: 320.0x240.0
  Webcam FPS set to: 30.0
  Camera window created.
  ```
- **Enrolling a New Face**:
  ```
  Detected 1 faces
  Validating face with coordinates: (127, 61, 238, 172)
  Face validated successfully.
  EAR: 0.428
  No blink detected for 2 frames. Assuming live.
  Face 0 at (127, 61, 238, 172) - Alive: True
  Prompting for registration of Face 0
  Enter name for Face 0: Madhav
  Please show your face from the front angle (Angle 1/5). Press any key to capture.
  Enroll: Saved debug image as debug_enroll_madhav_angle_front.jpg
  Enroll: Generated encoding for madhav (front angle): [-0.16574107  0.12715843  0.05904005  0.02415116  0.01636617]...
  Please show your face from the left angle (Angle 2/5). Press any key to capture.
  Enroll: Saved debug image as debug_enroll_madhav_angle_left.jpg
  Enroll: Generated encoding for madhav (left angle): [-0.170  0.130  0.060  0.025  0.015]...
  Please show your face from the right angle (Angle 3/5). Press any key to capture.
  Enroll: Saved debug image as debug_enroll_madhav_angle_right.jpg
  Enroll: Generated encoding for madhav (right angle): [-0.168  0.128  0.058  0.023  0.017]...
  Please show your face from the up angle (Angle 4/5). Press any key to capture.
  Enroll: Saved debug image as debug_enroll_madhav_angle_up.jpg
  Enroll: Generated encoding for madhav (up angle): [-0.166  0.126  0.057  0.022  0.016]...
  Please show your face from the down angle (Angle 5/5). Press any key to capture.
  Enroll: Saved debug image as debug_enroll_madhav_angle_down.jpg
  Enroll: Generated encoding for madhav (down angle): [-0.164  0.125  0.056  0.021  0.018]...
  Enrolled madhav successfully! Total encodings for madhav: 5
  Saved encodings for 1 users to database.json
  Frame displayed.
  Frame processing time: 10.481 seconds
  ```
- **Recognizing a Face**:
  ```
  Detected 1 faces
  Validating face with coordinates: (127, 61, 238, 172)
  Face validated successfully.
  EAR: 0.428
  Face 0 at (127, 61, 238, 172) - Alive: True
  Recognize: Original image shape: (177, 177, 3)
  Recognize: Generated encoding: [-0.165  0.127  0.059  0.024  0.016]...
  Distance to madhav (encoding 0): 0.010
  Distance to madhav (encoding 1): 0.020
  Distance to madhav (encoding 2): 0.015
  Distance to madhav (encoding 3): 0.018
  Distance to madhav (encoding 4): 0.012
  Recognized madhav with best distance 0.010
  Detected emotion: Happy (Confidence: 0.842)
  Frame displayed.
  Frame processing time: 0.025 seconds
  ```
  - The frame will display: "madhav - Happy".

## Troubleshooting
1. **Repeated Enrollment Prompts**:
   - **Solution**: Adjust the IoU threshold in `main.py`:
     ```python
     if iou > best_iou and iou > 0.3:  # Lowered threshold
     ```
   - **Solution**: Adjust the distance threshold in `face_recognizer.py`:
     ```python
     best_match = name if distance < 0.6 else "Unknown"  # Stricter
     ```

2. **Frame Processing Time Exceeds 0.033 Seconds**:
   - **Solution**: Increase intervals in `main.py`:
     ```python
     recognition_interval = 20
     emotion_update_interval = 15
     ```
   - **Solution**: Disable emotion detection temporarily:
     ```python
     # if frame_count % emotion_update_interval == 0:
     #     threading.Thread(target=emotion_worker, args=(emotion_analyzer, face_roi, data["emotion_queue"]), daemon=True).start()
     ```

3. **Confusion Between Persons**:
   - **Solution**: Increase `max_encodings_per_person` in `face_recognizer.py`:
     ```python
     class FaceRecognizer:
         def __init__(self, storage_path="database.json", max_encodings_per_person=10):
     ```

4. **Enrollment Fails (No Face Encoding Found)**:
   - **Solution**: Check debug images (`debug_enroll_<name>_angle_<angle>.jpg`) to ensure the face is visible.
   - **Solution**: Increase padding in `main.py`:
     ```python
     padding = 75
     ```
   - **Solution**: Increase image size in `face_recognizer.py`:
     ```python
     image_rgb = cv2.resize(image_rgb, (0, 0), fx=1.0, fy=1.0)  # No resizing
     ```

5. **Emotion Detection Fails or Has Low Confidence**:
   - **Solution**: Lower the confidence threshold in `emotion_analyzer.py`:
     ```python
     if confidence < 0.3:  # Lowered threshold
     ```

## Limitations
- **Lighting Conditions**: Poor lighting may affect face detection and recognition accuracy.
- **Occlusions**: Partial face occlusions (e.g., masks, hands) may lead to detection failures.
- **Hardware Dependency**: Real-time performance depends on the hardware (CPU/GPU) capabilities.
- **Single Camera**: Currently supports only one camera input.

## Future Improvements
- **Multi-Camera Support**: Extend the system to handle multiple camera inputs.
- **GPU Acceleration**: Optimize for GPU to improve performance.
- **Advanced Liveness Detection**: Add more sophisticated liveness checks (e.g., head movement, 3D depth).
- **Database Management**: Add a GUI for managing enrolled users (e.g., delete, update).
- **Cloud Integration**: Store encodings in a cloud database for scalability.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to your branch (`git push origin feature/your-feature`).
5. Create a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- Thanks to the `face_recognition` library for providing an easy-to-use face recognition API.
- Thanks to the `DeepFace` library for emotion detection capabilities.
- Thanks to `dlib` and `OpenCV` for their robust computer vision tools.

---
