import argparse
from bgsub_detect import YOLODetector
from bgsub_segment import YoloSegmentation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Detection and Segmentation")
    parser.add_argument('--task', type=str, required=True, choices=['detect', 'segment'], 
                        help="Specify the task: 'detect' or 'segment'.")
    parser.add_argument('--video_path', type=str, default=None, help="Path to the video file.")
    parser.add_argument('--stream', type=str, default=None, help="Host to the camera.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the YOLO model file.")
    parser.add_argument('--class_names_path', type=str, help="Path to the class names file (required for 'segment').")
    args = parser.parse_args()

    if args.task == "detect":
        # Run detection task
        detector = YOLODetector(
            model_path=args.model_path,
            video_path=args.video_path,
            stream=args.stream
        )
        detector.run()
    elif args.task == "segment":
        # Check if class_names_path is provided
        if not args.class_names_path:
            raise ValueError("You must provide --class_names_path for the 'segment' task.")
        # Run segmentation task
        segmentation = YoloSegmentation(
            model_path=args.model_path,
            class_names_path=args.class_names_path
        )
        segmentation.run(video_path=args.video_path, stream=args.stream)

#Running example
# Detect task
#python main.py --task detect --model_path "yolo11n-detect_int8_openvino_model" --video_path "../test_input/4p-c2.mp4"
#python main.py --task detect --model_path "yolo11n-detect_int8_openvino_model" --stream "192.168.1.100"

# Stream task
#python main.py --task segment --model_path "yolo11n-seg_int8_openvino_model" --class_names_path "coco.names" --video_path "../test_input/4p-c2.mp4"
#python main.py --task segment --model_path "yolo11n-seg_int8_openvino_model" --class_names_path "coco.names" --stream "192.168.1.100"

