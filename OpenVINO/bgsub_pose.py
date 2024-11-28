import argparse
import cv2
import numpy as np
from ultralytics import YOLO

def extract_body_colors(image, keypoints, detected_frame):
    """
    Extract average color for body parts based on keypoints and visualize ROIs.
    :param image: Original image (BGR format).
    :param keypoints: List of keypoints, each with [x, y, score].
    :param detected_frame: Frame to visualize the ROIs.
    :return: Dictionary of body parts with their average colors.
    """
    # Define body parts based on keypoint indexes (example with COCO format)
    body_parts = {
        #"head": [0],  # Nose
        "left_arm": [5, 7],  # Shoulder, wrist
        "right_arm": [6, 8],
        "left_thigh": [11, 13],  # thigh
        "left_leg": [13, 15], # Hip, ankle
        "right_thigh": [12, 14],  # thigh
        "right_leg": [14, 16],
        "torso": [5, 6, 11, 12]  # Shoulders and hips
    }

    body_colors = {}

    for part, indexes in body_parts.items():
        roi_points = []
        for idx in indexes:
            if idx < len(keypoints) and keypoints[idx][2] > 0.5:  # Check confidence score
                roi_points.append((int(keypoints[idx][0]), int(keypoints[idx][1])))

        # If at least two points exist, create an ROI
        if len(roi_points) >= 2:
            x_coords, y_coords = zip(*roi_points)
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Expand ROI slightly for better context
            x_min = max(x_min - 5, 0)
            x_max = min(x_max + 5, image.shape[1])
            y_min = max(y_min - 5, 0)
            y_max = min(y_max + 5, image.shape[0])

            # Extract ROI from image
            roi = image[y_min:y_max, x_min:x_max]

            # Visualize the ROI on the detected_frame
            cv2.rectangle(detected_frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)
            cv2.putText(detected_frame, part, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            #print(roi)
            #cv2.imwrite("roi_file.png", roi)
            # # Compute average color in ROI (BGR)
            if roi.size > 0:
                #avg_color = roi.mean(axis=(0, 1))  # Average over height and width
                # Tạo mặt nạ để lọc bỏ pixel có giá trị (255, 255, 255)
                mask = ~(roi == [255, 255, 255]).all(axis=-1)

                # Lấy các giá trị không phải (255, 255, 255)
                filtered_roi = roi[mask]

                # Tính giá trị trung bình trên các pixel hợp lệ
                if len(filtered_roi) > 0:
                    avg_color = filtered_roi.mean(axis=0)
                else:
                    avg_color = [0, 0, 0] 
                body_colors[part] = tuple(map(int, avg_color))  # Convert to int
            
    

    return body_colors

def draw_body_colors_on_frame(frame, bbox, body_colors, person_id):
    """
    Draw color bars for each body part on the frame.
    :param frame: The image frame to draw on.
    :param bbox: Bounding box (x1, y1, x2, y2) of the person.
    :param body_colors: Dictionary of body parts and their average colors.
    :param person_id: Index of the person.
    """
    x1, y1, x2, y2 = bbox
    # Xác định vị trí y_offset để vẽ dải màu trên hoặc dưới bounding box
    y_offset = y1 - 20 if y1 - 20 > 0 else y2 + 10  # Place above or below the bounding box
    x_offset = x1

    for i, (part, color) in enumerate(body_colors.items()):
        # Chuyển đổi màu từ BGR sang RGB cho OpenCV
        b, g, r = color
        color_rgb = (r, g, b)

        # Vẽ một hình chữ nhật nhỏ cho mỗi bộ phận cơ thể
        cv2.rectangle(frame, (x_offset, y_offset + i * 15), (x_offset + 50, y_offset + (i + 1) * 15), color_rgb, -1)

        # Thêm nhãn cho bộ phận cơ thể
        cv2.putText(frame, part, (x_offset + 55, y_offset + (i + 1) * 15 - 2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # Label the person ID
    cv2.putText(frame, f"Person {person_id}", (x1, y1 - 10 if y1 - 10 > 0 else y2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

def get_opencv_yolo_result(video_path=None, stream=None):
    if stream:
        captured_video = cv2.VideoCapture(f"rtsp://admin:MYMDIZ@{stream}:554/ch01/0")
    else:
        captured_video = cv2.VideoCapture(video_path if video_path else 0)
    if not captured_video.isOpened():
        print("Unable to access the webcam or video file.")
        exit(0)

    # Background subtraction method
    background_subtr_method = cv2.bgsegm.createBackgroundSubtractorGSOC(
        noiseRemovalThresholdFacFG=0.2,
        blinkingSupressionMultiplier=0.4,
        blinkingSupressionDecay=0.3
    )

    # Load YOLO model
    model = YOLO("yolo11n-pose_int8_openvino_model") 
    prev_frame_time = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # Kernel for morphological operations

    while True:
        retval, frame = captured_video.read()
        if not retval:
            print("Failed to grab frame.")
            if stream:
                captured_video = cv2.VideoCapture(f"rtsp://admin:MYMDIZ@{stream}:554/ch01/0")
                continue
            else:
                break

        # Resize frame for consistency
        frame = cv2.resize(frame, (640, 360))
        detected_frame = frame.copy()  # Copy the original frame for overlay

        # Background subtraction and noise reduction
        foreground_mask = background_subtr_method.apply(frame, learningRate=0.5)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)

        # YOLO detection
        results = model.predict(frame, verbose=False, classes=[0], device="cpu")[0]
        bbox_mask = np.zeros_like(foreground_mask, dtype=np.uint8)

        for i, box in enumerate(results.boxes):
            if box.cls == 0:  # Ensure only "person" class is processed
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box on the bbox_mask
                cv2.rectangle(bbox_mask, (x1, y1), (x2, y2), 255, -1)

                # Draw bounding box on the detected_frame
                cv2.rectangle(detected_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw key points if they exist
                if results.keypoints is not None and len(results.keypoints.data) > i:
                    keypoints = results.keypoints.data[i]  # Get key points for this detection
                    if keypoints is not None:
                        foreground_only = cv2.bitwise_and(frame, frame, mask=foreground_mask)
                        colors = extract_body_colors(foreground_only, keypoints, detected_frame)
                        print(f"Colors for person {i}: {colors}")
                        draw_body_colors_on_frame(detected_frame, map(int, box.xyxy[0]), colors, i)
                    for kp in keypoints:        
                        kp_x, kp_y, kp_score = kp  # x, y coordinates and confidence score
                        if kp_score > 0.5:  # Draw only if confidence is above 0.5
                            cv2.circle(detected_frame, (int(kp_x), int(kp_y)), 3, (0, 0, 255), -1)

        # Isolate pixels in the mask inside bounding boxes
        red_area = cv2.bitwise_and(foreground_mask, bbox_mask)  # Pixels in mask & bounding boxes
        blue_area = cv2.bitwise_and(foreground_mask, cv2.bitwise_not(bbox_mask))  # Pixels in mask but outside bounding boxes

        # Create colored overlays
        red_overlay = np.full_like(frame, (0, 0, 255), dtype=np.uint8)  # Red fill
        blue_overlay = np.full_like(frame, (255, 0, 0), dtype=np.uint8)  # Blue fill

        # Apply masks to overlays
        red_mask = cv2.bitwise_and(red_overlay, red_overlay, mask=red_area)
        blue_mask = cv2.bitwise_and(blue_overlay, blue_overlay, mask=blue_area)

        # Alpha blending for transparency
        alpha = 0.3
        detected_frame = cv2.addWeighted(detected_frame, 1 - alpha, red_mask, alpha, 0)
        detected_frame = cv2.addWeighted(detected_frame, 1 - alpha, blue_mask, alpha, 0)

        # FPS calculation
        current_frame_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (current_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
        prev_frame_time = current_frame_time
        cv2.putText(detected_frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the results
        cv2.imshow("Initial Frames", frame)
        cv2.imshow("Foreground Masks", foreground_mask)
        cv2.imshow("YOLO Detection with Overlays", detected_frame)

        if cv2.waitKey(10) == 27:  # Press 'Esc' to exit
            break

    captured_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default=None, help="Path to the video file.")
    parser.add_argument('--stream', type=str, default=None, help="Host to the camera.")
    args = parser.parse_args()
    get_opencv_yolo_result(args.video_path, stream=args.stream)
