import cv2
from ultralytics import YOLO
import time

# Initialize the YOLO model
model = YOLO("yolo11n_int8_openvino_model")

# Capture video
cap = cv2.VideoCapture("../4p-c2.mp4")

while True:
    ret, img = cap.read()
    if not ret:
        break

    # Calculate time before prediction
    prev_time = time.time()
    
    # Run prediction
    res = model.predict(img, verbose=False, classes=[0], device="cpu")[0]
    
    # Calculate FPS
    fps = 1 / (time.time() - prev_time)
    
    # Display FPS on frame
    cv2.putText(img, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Draw bounding boxes
    for box in res.boxes:
        xyxy = box.xyxy.cpu().numpy().tolist()[0]
        xyxy = [int(x) for x in xyxy]
        cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2] ,xyxy[3]), (0, 255, 89), 2, cv2.LINE_AA)
        
    # Show frame with detections
    cv2.imshow('Detected People', img)
    
    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
