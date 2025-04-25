import pyrealsense2 as rs
import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO

def initialize_camera():
    # Create a pipeline
    pipeline = rs.pipeline()
    
    # Create a config object
    config = rs.config()
    
    # Configure the pipeline to stream
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # Start streaming
    pipeline.start(config)
    return pipeline

def main():
    # Initialize YOLO model
    #model_path = r"C:\Users\teomeo\Desktop\aMU_MSc\desertation\Yolo11seg\yolo11_noaug\373n_seg_noaug.pt"
    model_path = r"C:\Users\teomeo\Desktop\aMU_MSc\desertation\Yolo11seg\yolo11_noaug\373s_seg_noaug_50ep.onnx"
    model = YOLO(model_path)
    
    # Initialize annotators
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.CLASS, opacity=0.5)
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT, text_padding=3)
    
    try:
        # Initialize camera
        pipeline = initialize_camera()
        
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
            
            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            
            # Run YOLO detection
            results = model(color_image)
            
            # Convert results to supervision Detections
            detections = sv.Detections.from_ultralytics(results[0])
            
            # Annotate the image with bounding boxes and labels
            annotated_image = color_image.copy()
            
            # Add segmentation masks
            annotated_image = mask_annotator.annotate(
                scene=annotated_image, 
                detections=detections
            )
            
            # Add bounding boxes
            annotated_image = box_annotator.annotate(
                scene=annotated_image, 
                detections=detections
            )
            
            # Add labels
            annotated_image = label_annotator.annotate(
                scene=annotated_image, 
                detections=detections
            )
            
            # Show the annotated stream
            cv2.imshow('RealSense Divot Detection', annotated_image)
            
            # Break loop with 'q' - using a more reliable key detection
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("Exiting program...")
                break
                
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()