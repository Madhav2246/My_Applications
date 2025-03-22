import cv2
import numpy as np

def adjust_lighting(image):
    """Adjust brightness and contrast while preserving color."""
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    y_eq = cv2.equalizeHist(y)
    yuv_eq = cv2.merge((y_eq, u, v))
    return cv2.cvtColor(yuv_eq, cv2.COLOR_YUV2BGR)

def calculate_ear(eye_points):
    """Calculate Eye Aspect Ratio for liveness detection."""
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear if C != 0 else 1.0