import cv2
import time
import numpy as np
from collections import Counter
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

def extract_colors_from_mask(image, mask, top_n=5):
    # Chuyển ảnh từ BGR sang RGB (OpenCV mặc định là BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Áp dụng mask lên ảnh: Chỉ lấy những pixel trong vùng mask
    masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    # masked_image_bgr = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("masked_image.png", masked_image_bgr)
    # Lấy các pixel đã được trích xuất từ vùng mask
    pixels = masked_image.reshape(-1, 3)
    
    # Lọc bỏ các pixel trắng (không thuộc vùng mask, ví dụ như vùng nền)
    pixels = pixels[~np.all(pixels == [0, 0, 0], axis=1)]
    
    # Giảm độ phân giải màu (mỗi kênh giảm xuống 16 mức)
    pixels = np.round(pixels / 16) * 16  
    
    # Chuyển đổi thành tuple để đếm
    pixel_counts = Counter(map(tuple, pixels))

    # Tính tỷ lệ
    total_pixels = sum(pixel_counts.values())
    color_ratios = {color: count / total_pixels for color, count in pixel_counts.items()}

    # Lấy ra top_n màu phổ biến nhất
    top_colors = sorted(color_ratios.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top_colors

def display_color_strip(frame, top_colors):
    # Tạo dải màu tương ứng với các màu phổ biến
    strip_height = 50  # Chiều cao dải màu
    strip_width = frame.shape[1]  # Chiều rộng dải màu bằng chiều rộng của frame
    color_strip = np.zeros((strip_height, strip_width, 3), dtype=np.uint8)
    # Lặp qua các màu và vẽ chúng trên dải màu
    total_ratio = sum([ratio for _, ratio in top_colors])
    start_x = 0
    for color, ratio in top_colors:
        # Tính toán chiều dài của mỗi phần màu dựa trên tỷ lệ
        color_length = int(ratio * strip_width)
        
        # Chuyển RGB sang BGR (OpenCV sử dụng BGR)
        color_bgr = (int(color[2]), int(color[1]), int(color[0]))
        
        # Vẽ màu lên dải màu
        color_strip[:, start_x:start_x + color_length] = color_bgr
        start_x += color_length
    
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
