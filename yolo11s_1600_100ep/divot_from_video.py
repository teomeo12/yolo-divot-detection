import numpy as np
import cv2
import supervision as sv
import os
from ultralytics import YOLO
import torch

def process_video(video_path, model_path, output_path, resize_factor=0.5, process_every_n_frames=2):
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
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create output video with the resized dimensions
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
    # Check for CUDA availability
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA not available. Running on CPU.")

    # Base workspace directory
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Dictionary mapping project folders to their model files
    project_models = {
        
        'yolo11s_1600_100ep': '1600s_aug_100ep.pt',
        
        # 'yolo11s_1600_100ep': 'model_name.pt'  # Uncomment and add correct model name when available
    }
    
    # Videos directory
    videos_dir = os.path.join(workspace_dir, 'videos')
    
    # Process videos with each available model
    for project_folder, model_file in project_models.items():
        print(f"\nProcessing with model from {project_folder}")
        
        # Create processed_videos subfolder
        processed_dir = os.path.join(workspace_dir, project_folder, 'processed_videos')
        os.makedirs(processed_dir, exist_ok=True)
        
        # Get model path
        model_path = os.path.join(workspace_dir, project_folder, model_file)
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}")
            continue
            
        # Process each video
        for video_file in sorted(os.listdir(videos_dir)):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                print(f"\nProcessing video: {video_file}")
                
                video_path = os.path.join(videos_dir, video_file)
                output_path = os.path.join(processed_dir, f"{os.path.splitext(video_file)[0]}_processed.mp4")
                
                try:
                    process_video(video_path, model_path, output_path, resize_factor=0.5, process_every_n_frames=2)
                except Exception as e:
                    print(f"Error processing {video_file}: {str(e)}")

if __name__ == "__main__":
    main() 