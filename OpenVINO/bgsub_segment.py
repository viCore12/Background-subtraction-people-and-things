import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

class YoloSegmentation:
    def __init__(self, model_path, class_names_path, kernel_size=(3, 3)):
        self.model = YOLO(model_path)  # Load YOLO segmentation model
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)  # Morphological kernel
        self.prev_frame_time = 0  # For FPS calculation
        with open(class_names_path, "r") as file:
            self.names = {index: line.strip() for index, line in enumerate(file)}  # Class names dictionary
        self.background_subtractor = cv2.bgsegm.createBackgroundSubtractorGSOC(
            noiseRemovalThresholdFacFG=0.2,
            blinkingSupressionMultiplier=0.4,
            blinkingSupressionDecay=0.3
        )

    def process_frame(self, frame):
        original_frame = frame.copy()
        foreground_mask = self.background_subtractor.apply(frame, learningRate=0.5)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, self.kernel)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, self.kernel)

        results = self.model.predict(frame, verbose=False, classes=[0], device="cpu")[0]
        annotator = Annotator(frame, line_width=2)
        bbox_mask = np.zeros_like(foreground_mask, dtype=np.uint8)

        if results.masks is not None:
            clss = results.boxes.cls.cpu().tolist()
            boxes = results.boxes.xyxy.cpu().tolist()
            masks = results.masks.xy
            for box, mask, cls in zip(boxes, masks, clss):
                color = colors(int(cls), True)
                txt_color = annotator.get_txt_color(color)
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(bbox_mask, (x1, y1), (x2, y2), 255, -1)
                annotator.seg_bbox(mask=mask, mask_color=color, label=self.names[int(cls)], txt_color=txt_color)

            green_area = cv2.bitwise_and(foreground_mask, cv2.bitwise_not(bbox_mask))
            green_overlay = np.full_like(frame, (0, 255, 0), dtype=np.uint8)
            green_mask = cv2.bitwise_and(green_overlay, green_overlay, mask=green_area)
            alpha = 0.3
            frame = cv2.addWeighted(frame, 1 - alpha, green_mask, alpha, 0)

        return original_frame, foreground_mask, frame

    def run(self, video_path=None, stream=None):
        if stream:
            capture = cv2.VideoCapture(f"rtsp://admin:MYMDIZ@{stream}:554/ch01/0")
        else:
            capture = cv2.VideoCapture(video_path if video_path else 0)

        if not capture.isOpened():
            print("Unable to access the webcam or video file.")
            return

        while True:
            ret, frame = capture.read()
            if not ret:
                print("Failed to grab frame.")
                if stream:
                    capture = cv2.VideoCapture(f"rtsp://admin:MYMDIZ@{stream}:554/ch01/0")
                    continue
                else:
                    break

            frame = cv2.resize(frame, (640, 360))
            original_frame, foreground_mask, processed_frame = self.process_frame(frame)

            current_frame_time = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (current_frame_time - self.prev_frame_time) if self.prev_frame_time != 0 else 0
            self.prev_frame_time = current_frame_time
            cv2.putText(processed_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Original Frame", original_frame)
            cv2.imshow("Foreground Mask", foreground_mask)
            cv2.imshow("Processed Frame", processed_frame)

            if cv2.waitKey(10) == 27:
                break

        capture.release()
        cv2.destroyAllWindows()


