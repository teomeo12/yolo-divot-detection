import numpy as np
import cv2
import supervision as sv
import os
from ultralytics import YOLO

def process_video(video_path, model_path, resize_factor=0.5, process_every_n_frames=2):
    # Initialize YOLO model
    model = YOLO(model_path)
    
    # Initialize annotators
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.CLASS, opacity=0.5)
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT, text_padding=3)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate new dimensions
    new_width = int(original_width * resize_factor)
    new_height = int(original_height * resize_factor)
    
    print(f"Resizing video from {original_width}x{original_height} to {new_width}x{new_height}")
    print(f"Processing every {process_every_n_frames} frames")
    
    # Create output video with the resized dimensions
    output_path = os.path.splitext(video_path)[0] + "_processed.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps/process_every_n_frames, (new_width, new_height))
    
    try:
        frame_count = 0
        while cap.isOpened():
            # Read a frame
            ret, frame = cap.read()
            
            if not ret:
                # End of video
                break
            
            frame_count += 1
            
            # Skip frames according to process_every_n_frames
            if (frame_count - 1) % process_every_n_frames != 0:
                continue
            
            # Resize frame
            resized_frame = cv2.resize(frame, (new_width, new_height))
            
            # Run YOLO detection
            results = model(resized_frame)
            
            # Convert results to supervision Detections
            detections = sv.Detections.from_ultralytics(results[0])
            
            # Annotate the image
            annotated_frame = resized_frame.copy()
            
            # Add segmentation masks
            annotated_frame = mask_annotator.annotate(
                scene=annotated_frame, 
                detections=detections
            )
            
            # Add bounding boxes
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, 
                detections=detections
            )
            
            # Add labels
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, 
                detections=detections
            )
            
            # Write the frame to output video
            out.write(annotated_frame)
            
            # Show the annotated frame
            cv2.imshow('Divot Detection (Video)', annotated_frame)
            
            # Break loop with 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("Exiting program...")
                break
                
    finally:
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Processed video saved to: {output_path}")

def main():
    # Path to YOLO model
    model_path = r"C:\Users\teomeo\Desktop\aMU_MSc\desertation\Yolo11seg\yolo11n_1600_40ep\1600n-aug-40ep.pt"
    #model_path = r"C:\Users\teomeo\Desktop\aMU_MSc\desertation\Yolo11seg\yolo11n_1600_40ep\1600n-aug-40ep.onnx"
    
    # Folder containing videos
    video_folder = r"C:\Users\teomeo\Desktop\aMU_MSc\desertation\GolfDivotsImages\VideoGolf"
    
    # List all video files in the folder
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if not video_files:
        print(f"No video files found in {video_folder}")
        return
    
    # Print available videos
    print("Available videos:")
    for i, video in enumerate(video_files):
        print(f"{i+1}. {video}")
    
    # Process the first video automatically
    video_path = os.path.join(video_folder, video_files[0])
    print(f"Processing video: {video_files[0]}")
    
    # Process with reduced size (50%) and only every 2nd frame
    process_video(video_path, model_path, resize_factor=0.5, process_every_n_frames=2)

if __name__ == "__main__":
    main() 