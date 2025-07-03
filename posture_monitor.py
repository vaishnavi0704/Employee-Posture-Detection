import cv2
import mediapipe as mp
import numpy as np
from plyer import notification
import time
import pandas as pd
from datetime import datetime

# Initialize Mediapipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Capture from webcam
cap = cv2.VideoCapture(0)

# Variables
bad_posture_counter = 0
alert_threshold = 30  # Number of consecutive bad frames before alert
alert_interval = 60   # Seconds between alerts
last_alert_time = time.time()

# CSV Log File
log_file = 'posture_log.csv'

# Create CSV with headers if not exist
try:
    df = pd.read_csv(log_file)
except FileNotFoundError:
    df = pd.DataFrame(columns=['Timestamp', 'Posture Status'])
    df.to_csv(log_file, index=False)

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # Mirror effect

        # Convert frame to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        posture_status = "Good Posture"

        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates for important points
            left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            # ----------- 1. Forward Head (Tech Neck) Detection ------------
            left_dist = np.linalg.norm(np.array(left_ear) - np.array(left_shoulder))
            right_dist = np.linalg.norm(np.array(right_ear) - np.array(right_shoulder))
            avg_ear_shoulder_dist = (left_dist + right_dist) / 2

            tech_neck_threshold = 0.1

            if avg_ear_shoulder_dist < tech_neck_threshold:
                posture_status = "Bad Posture: Head Forward (Tech Neck)"

            # ----------- 2. Shoulder Slouch/Side Lean Detection ------------
            shoulder_diff_y = abs(left_shoulder[1] - right_shoulder[1])

            slouch_threshold = 0.05

            if shoulder_diff_y > slouch_threshold:
                posture_status = "Bad Posture: Slouching Sideways"

            # ----------- 3. Spine Angle (Hunchback) Detection ------------
            spine_angle = calculate_angle(left_hip, left_shoulder, left_ear)

            spine_angle_threshold = 150

            if spine_angle < spine_angle_threshold:
                posture_status = "Bad Posture: Bent Spine (Hunchback)"

            # ------------------ Display posture status --------------------
            if "Bad Posture" in posture_status:
                bad_posture_counter += 1
                cv2.putText(image, posture_status, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                bad_posture_counter = 0
                cv2.putText(image, posture_status, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            # ------------- Send Notification if Bad Posture Persists -------------
            if bad_posture_counter > alert_threshold and (time.time() - last_alert_time) > alert_interval:
                notification.notify(
                    title='Posture Alert!',
                    message=posture_status + " — Please Correct Your Posture!",
                    timeout=5
                )
                last_alert_time = time.time()

            # ------------------- Log to CSV -------------------
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            new_row = {'Timestamp': timestamp, 'Posture Status': posture_status}
            df = pd.DataFrame([new_row])
            df.to_csv(log_file, mode='a', header=False, index=False)

        except Exception as e:
            print("Detection Error:", e)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Show the output
        cv2.imshow('Office Posture Monitor', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
