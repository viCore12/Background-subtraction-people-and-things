import cv2
import numpy as np
from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path, video_path=None, stream=None):
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
        for result in results:
            for box in result.boxes:
                if box.cls == 0:  # Ensure only "person" is processed
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(bbox_mask, (x1, y1), (x2, y2), 255, -1)
                    cv2.rectangle(detected_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        red_area = cv2.bitwise_and(foreground_mask, bbox_mask)
        blue_area = cv2.bitwise_and(foreground_mask, cv2.bitwise_not(bbox_mask))

        red_overlay = np.full_like(frame, (0, 0, 255), dtype=np.uint8)
        blue_overlay = np.full_like(frame, (255, 0, 0), dtype=np.uint8)

        red_mask = cv2.bitwise_and(red_overlay, red_overlay, mask=red_area)
        blue_mask = cv2.bitwise_and(blue_overlay, blue_overlay, mask=blue_area)

        alpha = 0.3
        detected_frame = cv2.addWeighted(detected_frame, 1 - alpha, red_mask, alpha, 0)
        detected_frame = cv2.addWeighted(detected_frame, 1 - alpha, blue_mask, alpha, 0)

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
