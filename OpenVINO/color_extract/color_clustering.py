import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import cv2
import numpy as np
from ultralytics import YOLO
from clusterer import Clusterer  # Ensure this is correctly installed or defined

class YOLODetectorWithCluster:
    def __init__(self, model_path, video_path=None, stream=None, n_clusters=5):
        self.video_path = video_path
        self.stream = stream
        self.model = YOLO(model_path)
        self.prev_frame_time = 0
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.background_subtr_method = cv2.bgsegm.createBackgroundSubtractorGSOC(
            noiseRemovalThresholdFacFG=0.2,
            blinkingSupressionMultiplier=0.4,
            blinkingSupressionDecay=0.3
        )
        self.clusterer = Clusterer(n_clusters, 3, 100)  # RGB, 100 iterations

    def capture_video(self):
        if self.stream:
            return cv2.VideoCapture(f"rtsp://admin:MYMDIZ@{self.stream}:554/ch01/0")
        return cv2.VideoCapture(self.video_path if self.video_path else 0)

    def process_frame(self, frame):
        frame = cv2.resize(frame, (640, 360))
        foreground_mask = self.background_subtr_method.apply(frame, learningRate=0.5)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, self.kernel)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, self.kernel)

        detected_frame = frame.copy()
        results = self.model.predict(frame, verbose=False, classes=[0], device="cpu")[0]
        bbox_mask = np.zeros_like(foreground_mask, dtype=np.uint8)

        person_boxes = []
        for result in results:
            for box in result.boxes:
                if box.cls == 0:  # Ensure only "person" is processed
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_boxes.append((x1, y1, x2, y2))
                    cv2.rectangle(bbox_mask, (x1, y1), (x2, y2), 255, -1)
                    cv2.rectangle(detected_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Perform clustering on the foreground mask
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        centroids, labels = self.clusterer.forward(rgb_frame, mask=foreground_mask)

        if centroids is not None and len(person_boxes) > 0:
            # Convert centroids to BGR colors
            unique_colors = [tuple(map(int, centroid[::-1])) for centroid in centroids]
            color_strip_height = 20  # Adjust strip height for visibility

            for i, (x1, y1, x2, y2) in enumerate(person_boxes):
                strip_width = x2 - x1
                num_colors = len(unique_colors)
                segment_width = max(1, strip_width // num_colors)  # Width for each color segment

                # Check position to decide where to draw the color strip
                if y1 - color_strip_height < 0:
                    y_start = y2
                    y_end = y2 + color_strip_height
                else:
                    y_start = y1 - color_strip_height
                    y_end = y1

                # Draw multi-color strip
                for j, color in enumerate(unique_colors):
                    x_start = x1 + j * segment_width
                    x_end = min(x1 + (j + 1) * segment_width, x2)  # Ensure it doesn't overflow the box
                    cv2.rectangle(detected_frame, (x_start, y_start), (x_end, y_end), color, -1)

        return frame, foreground_mask, detected_frame

    def calculate_fps(self):
        current_frame_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (current_frame_time - self.prev_frame_time) if self.prev_frame_time != 0 else 0
        self.prev_frame_time = current_frame_time
        return fps

    def run(self):
        captured_video = self.capture_video()
        if not captured_video.isOpened():
            print("Unable to access the webcam or video file.")
            return

        while True:
            retval, frame = captured_video.read()
            if not retval:
                print("Failed to grab frame.")
                if self.stream:
                    captured_video = self.capture_video()
                    continue
                else:
                    break

            frame, foreground_mask, detected_frame = self.process_frame(frame)
            fps = self.calculate_fps()
            cv2.putText(detected_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Initial Frames", frame)
            cv2.imshow("Foreground Masks", foreground_mask)
            cv2.imshow("YOLO Detection with Overlays", detected_frame)

            if cv2.waitKey(10) == 27:  # Press 'Esc' to exit
                break

        captured_video.release()
        cv2.destroyAllWindows()

