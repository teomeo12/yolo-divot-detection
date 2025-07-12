import pyrealsense2 as rs
import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO
import os
from datetime import datetime

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

def create_output_directory():
    """Create the output directory if it doesn't exist"""
    output_dir = "divot_detection_vids"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def get_video_filename(output_dir):
    """Generate a unique filename with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(output_dir, f"divot_detection_{timestamp}.mp4")

def main():
    # Initialize YOLO model
    model_path = r"C:\Users\teomeo\Desktop\aMU_MSc\desertation\Yolo11seg\yolo11n_1600_40ep\1600n-aug-40ep.pt"
    #model_path = r"C:\Users\teomeo\Desktop\aMU_MSc\desertation\Yolo11seg\yolo11n_1600_40ep\1600n-aug-40ep.onnx"
    model = YOLO(model_path)
    
    # Initialize annotators
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.CLASS, opacity=0.5)
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT, text_padding=3)
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Recording variables
    is_recording = False
    video_writer = None
    video_filename = None
    
    try:
        # Initialize camera
        pipeline = initialize_camera()
        
        print("Controls:")
        print("  SPACEBAR - Start/Stop recording")
        print("  Q - Quit")
        print("  Recording will be saved to 'divot_detection_vids' folder")
        
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
            
            # Add recording indicator
            if is_recording:
                cv2.circle(annotated_image, (20, 20), 10, (0, 0, 255), -1)  # Red circle
                cv2.putText(annotated_image, "REC", (40, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 0, 255), 2)
            
            # Record frame if recording is active
            if is_recording and video_writer is not None:
                video_writer.write(annotated_image)
            
            # Show the annotated stream
            cv2.imshow('RealSense Divot Detection', annotated_image)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Spacebar to toggle recording
                if not is_recording:
                    # Start recording
                    video_filename = get_video_filename(output_dir)
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_filename, fourcc, 30.0, 
                                                 (annotated_image.shape[1], annotated_image.shape[0]))
                    is_recording = True
                    print(f"Started recording: {video_filename}")
                else:
                    # Stop recording
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    is_recording = False
                    print(f"Stopped recording: {video_filename}")
                    
            elif key == ord('q') or key == ord('Q'):
                print("Exiting program...")
                break
                
    finally:
        # Clean up recording if still active
        if video_writer is not None:
            video_writer.release()
            print(f"Final recording saved: {video_filename}")
        
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 