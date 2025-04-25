# YOLO Divot Detection

This project uses YOLOv11 for detecting and segmenting golf divots in videos. It includes multiple trained models with different configurations for comparison.

## Project Structure

- `yolo11n_1600_40ep/`: YOLOv11-nano model trained on 1600 images for 40 epochs
- `yolo11n_1600_100ep/`: YOLOv11-nano model trained on 1600 images for 100 epochs
- `yolo11n_373_noaug_50ep/`: YOLOv11-nano model trained on 373 images without augmentation for 50 epochs
- `yolo11s_1600_100ep/`: YOLOv11-small model trained on 1600 images for 100 epochs

## Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLO
- Supervision

## Usage

1. Place your input videos in the `videos/` directory
2. Run the video processing script:
   ```bash
   python yolo11n_1600_40ep/divot_from_video.py
   ```
3. Processed videos will be saved in `processed_videos/` subdirectories within each model's folder

## Model Details

Each model folder contains:
- `.pt` file: The trained YOLO model
- `divot_from_video.py`: Script for processing videos
- `divot_detection.py`: Script for real-time detection 