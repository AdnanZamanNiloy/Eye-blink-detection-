import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import imutils

# Load dlib's face detector and shape predictor model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'F:\Desktop\lv 1\Eye-Blink-Detector\Model\shape_predictor_68_face_landmarks.dat')

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])  # Vertical distance
    B = dist.euclidean(eye[2], eye[4])  # Vertical distance
    C = dist.euclidean(eye[0], eye[3])  # Horizontal distance
    ear = (A + B) / (2.0 * C)
    return ear

# Thresholds and Counters
EAR_THRESHOLD = 0.2  # Eye aspect ratio threshold for blink detection
CONSEC_FRAMES = 3    # Number of consecutive frames for a blink
blink_counter = 0     # Counter for consecutive frames
total_blinks = 0      # Total blinks detected

# Start Video Capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=640)  # Resize for efficiency
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    faces = detector(gray)  # Detect faces
    for face in faces:
        shape = predictor(gray, face)  # Get landmarks
        shape = np.array([[p.x, p.y] for p in shape.parts()])  # Convert to NumPy array

        # Extract Eye Coordinates
        left_eye = shape[42:48]  # Left eye landmarks (dlib index)
        right_eye = shape[36:42]  # Right eye landmarks

        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0  # Average EAR

        # Check for blink
        if avg_EAR < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= CONSEC_FRAMES:
                total_blinks += 1
                print("Blink detected! Total blinks:", total_blinks)
            blink_counter = 0  # Reset counter

        # Draw Eyes
        for (x, y) in np.concatenate((left_eye, right_eye), axis=0):
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Display Output
    cv2.putText(frame, f"Blinks: {total_blinks}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Liveness Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
