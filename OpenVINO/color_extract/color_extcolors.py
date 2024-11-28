import cv2
import time
import numpy as np
import extcolors
from PIL import Image
from collections import Counter
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

def extract_colors_from_mask(image, mask, top_n=5):
    # Áp dụng mask lên ảnh: Chỉ lấy những pixel trong vùng mask
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Lưu ảnh tạm thời để sử dụng với extcolors (extcolors hoạt động với định dạng PIL.Image)
    masked_image_pil = Image.fromarray(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    
    # Sử dụng extcolors để trích xuất màu sắc
    colors, pixel_count = extcolors.extract_from_image(masked_image_pil)
    
    # Tính tỷ lệ
    color_ratios = [(color, count / pixel_count) for color, count in colors[:top_n]]
    return color_ratios

def display_color_strip(frame, top_colors):
    # Tăng chiều cao dải màu
    strip_height = 50  # Điều chỉnh chiều cao
    strip_width = frame.shape[1]  # Chiều rộng dải màu bằng chiều rộng frame
    color_strip = np.zeros((strip_height, strip_width, 3), dtype=np.uint8)

    # Lặp qua các màu và vẽ chúng trên dải màu
    start_x = 0
    for color, ratio in top_colors:
        color_length = int(ratio * strip_width)
        color_bgr = (int(color[2]), int(color[1]), int(color[0]))
        end_x = min(start_x + color_length, strip_width)
        color_strip[:, start_x:end_x] = color_bgr

        # Hiển thị tỷ lệ phần trăm ở giữa phần màu
        text = f"{ratio*100:.1f}%"
        text_x = start_x + color_length // 2 - len(text) * 3
        text_y = strip_height // 2 + 5
        cv2.putText(color_strip, text, (max(text_x, 0), text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Thêm khoảng cách
        start_x = end_x + 1

    # Nối dải màu với frame
    combined_frame = np.vstack((frame, color_strip))

    return combined_frame

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    model = YOLO("yolo11n-seg_int8_openvino_model")  
    with open("coco.names", "r") as file:
        names = {index: line.strip() for index, line in enumerate(file)}
    
    prev_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
        prev_time = current_time

        combined_frame = frame.copy()
        frame_with_boxes = frame.copy()
        results = model.predict(frame, verbose=False, classes=[0], device="cpu")[0]
        annotator = Annotator(frame_with_boxes, line_width=2)
        
        if results.masks is not None:
            clss = results.boxes.cls.cpu().tolist()
            boxes = results.boxes.xyxy.cpu().tolist()  
            masks = results.masks.xy
            for box, mask, cls in zip(boxes, masks, clss):
                color = colors(int(cls), True)
                txt_color = annotator.get_txt_color(color)
                x1, y1, x2, y2 = map(int, box)
                mask_image = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask_image, [np.array(mask, dtype=np.int32)], 255)
                top_colors = extract_colors_from_mask(frame, mask_image, top_n=5)
                combined_frame = display_color_strip(combined_frame, top_colors)
                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
                annotator.seg_bbox(mask=mask, mask_color=color, label=names[int(cls)], txt_color=txt_color)

        # Add FPS text to combined_frame
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(combined_frame, fps_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Video with Segmented Object Color", combined_frame)
        cv2.imshow("Segmentation", frame_with_boxes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Chạy video với dải màu
video_path = '../4p-c2.mp4'
process_video(video_path)
