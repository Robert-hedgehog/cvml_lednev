from ultralytics import YOLO
import cv2
import time

model = YOLO("./runs/detect/figures/yolo/weights/best.pt")
model.to("mps")
camera = cv2.VideoCapture(0)

while camera.isOpened():
    ret, frame = camera.read()
    
    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break

    results = model(frame, verbose=False)
    result = results[0]
    for i in result.boxes:
        x1, y1, x2, y2 = map(int, i.xyxy[0])
        conf = float(i.conf[0])
        cls = int(i.cls[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"{model.names[cls]}: {conf:.2f}", (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Camera", frame)