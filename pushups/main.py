from ultralytics import YOLO
import cv2
import time
from ultralytics.utils.plotting import Annotator
import numpy as np
import math

def get_angle(a, b, c):
    cb = math.atan2(c[1] - b[1], c[0] - b[0])
    ab = math.atan2(a[1] - b[1], a[0] - b[0])
    angle = np.rad2deg(cb - ab)
    angle = angle + 360 if angle < 0 else angle
    return 360 - angle if angle > 180 else angle

def push_ups_account(annotator, keypoints, push_ups, is_down, last_face_time, current_time):
    nose_seen = (keypoints[0][0] > 0 and keypoints[0][1] > 0)
    eyes_seen = (keypoints[1][0] > 0 and keypoints[1][1] > 0 and keypoints[2][0] > 0 and keypoints[2][1] > 0)
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_knee = keypoints[13]
    right_knee = keypoints[14]
    left_ankle = keypoints[15]
    right_ankle = keypoints[16]

    if nose_seen and eyes_seen:
        last_face_time = current_time
        if (left_wrist[1] > left_elbow[1] > left_shoulder[1]) or (right_wrist[1] > right_elbow[1] > right_shoulder[1]):
            left_angle = get_angle(left_shoulder, left_elbow, left_wrist)
            # cv2.putText(annotated, f"Hands Up({left_angle:.1f})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 1)

            if (left_ankle[1] > left_knee[1] > left_hip[1]) or (right_ankle[1] > right_knee[1] > right_hip[1]):
                if left_angle <= 90 and left_angle >= 60 and is_down == False:
                    is_down = True
                    push_ups += 0.5
                if left_angle >= 160 and is_down == True:
                    is_down = False
                    push_ups += 0.5

            cv2.putText(annotator, f"Hands Up({push_ups})", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 3)

    return push_ups, is_down, last_face_time

model = YOLO("yolo26n-pose.pt")
model.to("mps")
camera = cv2.VideoCapture(0)

push_ups = 0
is_down = False
last_face_time = time.time()

while camera.isOpened():
    ret, frame = camera.read()
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break

    t = time.perf_counter()
    results = model(frame, verbose=False)
    # print(f"FPS {1 / (time.perf_counter() - t):.1f}")
    current_time = time.time() 

    if (current_time - last_face_time) > 10:
        push_ups = 0
        is_down = False

    if not results:
        continue
    else:
        result = results[0]
        keypoints = result.keypoints.xy.tolist()

        annotated = frame.copy()

        if keypoints:
            annotator = Annotator(frame)
            annotator.kpts(result.keypoints.data[0], result.orig_shape, 5, True)
            annotated = annotator.result()
    
            push_ups, is_down, last_face_time = push_ups_account(annotated, keypoints[0], push_ups, is_down, last_face_time, current_time)
        else:
            cv2.putText(annotated, f"Hands Up({push_ups})", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 3)
       
    cv2.imshow("Pose", annotated)