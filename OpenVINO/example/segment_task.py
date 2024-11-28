import cv2
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Initialize the YOLO model
model = YOLO("yolo11n-seg_int8_openvino_model")  # segmentation model

# Read the class names into a dictionary
with open("coco.names", "r") as file:
    names = {index: line.strip() for index, line in enumerate(file)}

# Open the video capture and define video properties
cap = cv2.VideoCapture("../4p-c2.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Initialize video writer
out = cv2.VideoWriter("instance-segmentation.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

# Main processing loop
while True:
    start_time = time.time()
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = model.predict(im0, verbose=False, classes=[0], device="cpu")[0]
    annotator = Annotator(im0, line_width=2)
    # Check if masks are detected
    if results.masks is not None:
        clss = results.boxes.cls.cpu().tolist()
        boxes = results.boxes.xyxy.cpu().tolist()  # Get bounding boxes
        masks = results.masks.xy
        for box, mask, cls in zip(boxes, masks, clss):
            color = colors(int(cls), True)
            txt_color = annotator.get_txt_color(color)
            
            # Draw bounding box
            annotator.box_label(box, color=color)
            
            # Draw segmentation mask
            annotator.seg_bbox(mask=mask, mask_color=color, label=names[int(cls)], txt_color=txt_color)

    # Calculate and display FPS
    fps_display = 1 / (time.time() - start_time)
    cv2.putText(im0, f"FPS: {fps_display:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Write the frame with FPS display to the output video
    out.write(im0)
    cv2.imshow("instance-segmentation", im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
out.release()
cap.release()
cv2.destroyAllWindows()
