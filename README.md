- pip install -r requirements.txt
- cd OpenVINO

## Running example

### Detect task
Video
```rb
python main.py --task detect --model_path "yolo11n-detect_int8_openvino_model" --video_path "../4p-c2.mp4"
```
Stream
```rb
python main.py --task detect --model_path "yolo11n-detect_int8_openvino_model" --stream "192.168.1.100"
```
### Segment task
Video
```rb
- python main.py --task segment --model_path "yolo11n-seg_int8_openvino_model" --class_names_path "coco.names" --video_path "../4p-c2.mp4"
```
Stream
```rb
- python main.py --task segment --model_path "yolo11n-seg_int8_openvino_model" --class_names_path "coco.names" --stream "192.168.1.100"
```
