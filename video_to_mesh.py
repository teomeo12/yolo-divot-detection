import os
import time
import argparse
import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO
import torch
import sys
# Add repository to path and import
sys.path.insert(0, './Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if torch.backends.mps.is_available() else 
                     "cpu")
print(f"Using device: {device}")

def initialize_depth_model(model_path='checkpoints/depth_anything_v2_vits.pth'):
    """Initialize the Depth Anything V2 model with a specific checkpoint file"""
    print(f"Loading depth model from: {model_path}")
    
    # Configure for vits model
    model_config = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    
    # Create model and load weights
    depth_model = DepthAnythingV2(**model_config)
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        print("Please specify the correct path to the model file using the --depth-model argument")
        return None
        
    depth_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    depth_model = depth_model.to(device).eval()
    return depth_model

def get_mask_from_detection(detections, class_id=None, index=0):
    """Extract a mask from detections by class_id or index"""
    if len(detections) == 0 or detections.mask is None:
        return None, None
    
    if class_id is not None:
        # Look for the specified class_id
        for i, det_class_id in enumerate(detections.class_id):
            if det_class_id == class_id and detections.mask[i] is not None:
                return detections.mask[i], i
        # If we didn't find the class_id, return None
        return None, None
    else:
        # Return the mask at the specified index, if it exists
        if index < len(detections.mask) and detections.mask[index] is not None:
            return detections.mask[index], index
        return None, None

def get_divot_depth_map(image, mask, depth_model, padding=20):
    """Get depth map for the divot region only"""
    # Get bounds of the mask with padding
    mask_indices = np.where(mask)
    if len(mask_indices[0]) == 0:
        # Empty mask
        return None, None
    
    y_min, y_max = np.min(mask_indices[0]), np.max(mask_indices[0])
    x_min, x_max = np.min(mask_indices[1]), np.max(mask_indices[1])
    
    # Add padding
    height, width = mask.shape
    y_min = max(0, y_min - padding)
    y_max = min(height - 1, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(width - 1, x_max + padding)
    
    # Extract ROI
    roi = image[y_min:y_max+1, x_min:x_max+1]
    
    # Process depth for ROI
    if roi.size == 0:
        return None, None
    
    # Process depth map for ROI using Depth Anything V2
    roi_depth = depth_model.infer_image(roi)
    
    # Create a full depth map with zeros except at the ROI
    full_depth = np.zeros_like(mask, dtype=np.float32)
    full_depth[y_min:y_max+1, x_min:x_max+1] = roi_depth
    
    # Apply mask to get only the divot part
    masked_depth = full_depth * mask
    
    return masked_depth, full_depth

def calculate_area_and_volume(mask, depth_map, focal_length=525.0, depth_scale=1000.0):
    """Calculate area and volume of a divot from mask and depth map"""
    if mask is None or depth_map is None:
        return None, None, None
    
    # Get pixel indices where mask is True
    pixel_indices = np.where(mask)
    if len(pixel_indices[0]) == 0:
        return 0, 0, None
    
    # Get depth values at mask positions
    depth_values = depth_map[pixel_indices]
    
    # Convert depth map values to actual depth in millimeters
    # The depth_scale value depends on the model and normalization
    depth_mm = depth_values * depth_scale
    
    # Get image dimensions
    h, w = mask.shape
    
    # Create meshgrid of coordinates
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    
    # Get coordinates for masked pixels
    x_coords = x_grid[pixel_indices]
    y_coords = y_grid[pixel_indices]
    
    # Calculate real-world coordinates using pinhole camera model
    # The focal_length value is camera-specific
    # For depth_map values that are already in mm
    
    # Initialize point cloud array (N x 3)
    point_cloud = np.zeros((len(x_coords), 3), dtype=np.float32)
    
    # Calculate center of the image
    cx, cy = w / 2, h / 2
    
    # Calculate real-world coordinates (in mm)
    point_cloud[:, 0] = (x_coords - cx) * depth_mm / focal_length  # X
    point_cloud[:, 1] = (y_coords - cy) * depth_mm / focal_length  # Y
    point_cloud[:, 2] = depth_mm  # Z
    
    # Calculate pixel area in real world:
    # Each pixel represents a real area that depends on depth
    # Area per pixel = (Z / focal_length)² * pixel_area_in_sensor
    # We approximate pixel_area_in_sensor as 1 unit squared
    
    # Calculate real world area in mm²
    pixel_real_areas = np.square(depth_mm / focal_length)
    total_area_mm2 = np.sum(pixel_real_areas)
    
    # Calculate average depth to determine baseline for volume
    min_depth = np.min(depth_mm)
    
    # Calculate volume by summing up the individual volume elements
    # Each element has a volume of: real_area * (max_depth - depth)
    volume_mm3 = np.sum((min_depth - depth_mm) * pixel_real_areas)
    
    # Filter out negative values (should not happen with proper calibration)
    volume_mm3 = max(0, volume_mm3)
    
    return total_area_mm2, volume_mm3, point_cloud

def process_video(video_path, yolo_model_path=None, depth_model_path=None, focal_length=525.0, 
                 start_frame=0, show_visualization=True, process_interval=5):
    """Process a video file and detect divots with segmentation"""
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Open the video file
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Initialize models
    if yolo_model_path is None:
        yolo_model_path = r"C:\Users\teomeo\Desktop\aMU_MSc\desertation\Yolo11seg\yolo11n_1600_40ep\1600n-aug-40ep.pt"
    
    print(f"Loading YOLO model: {yolo_model_path}")
    yolo_model = YOLO(yolo_model_path)
    depth_model = initialize_depth_model(depth_model_path)
    
    # Skip processing if depth model couldn't be loaded
    if depth_model is None:
        print("Error: Failed to initialize depth model. Exiting.")
        return
    
    # Get class names from YOLO model
    class_names = yolo_model.names
    # Get the class_id for 'divot' if it exists, otherwise use None
    divot_class_id = None
    fixed_divot_class_id = None
    for id, name in class_names.items():
        if name.lower() == 'divot':
            divot_class_id = id
        elif name.lower() == 'fixed_divot' or name.lower() == 'fixed_divots':
            fixed_divot_class_id = id
    
    print(f"Detected classes in model: {class_names}")
    print(f"Will focus on divots with class ID: {divot_class_id}")
    print(f"Will ignore fixed divots with class ID: {fixed_divot_class_id}")
    
    # Initialize supervision annotators
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.CLASS, opacity=0.5)
    label_annotator = sv.LabelAnnotator(
        text_position=sv.Position.TOP_LEFT, 
        text_padding=3,
        text_scale=3.0,  # Increase text scale (default is usually 0.5)
        text_thickness=2  # Increase text thickness for better visibility
    )
    
    # Variables for frame processing
    frame_count = 0
    processing_time = 0
    
    # Variables for tracking detections
    divot_volumes = {}  # Dictionary to store volume for each divot by index
    divot_areas = {}    # Dictionary to store area for each divot by index
    detections = None
    
    # Set the starting frame if specified
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_count = start_frame
        print(f"Starting from frame {start_frame}")
    
    # Create visualization window
    if show_visualization:
        cv2.namedWindow('Divot Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Divot Detection', 800, 600)
    
    print("Processing video... Press 'q' to quit, 'p' to pause/play")
    paused = False
    
    while True:
        if not paused:
            # Read a frame
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                break
            
            frame_count += 1
            
            # Check if we should process this frame
            should_process = frame_count % process_interval == 0
            
            if should_process:
                start_time = time.time()
                
                # Run YOLO detection
                results = yolo_model(frame)
                
                # Convert to supervision Detections
                detections = sv.Detections.from_ultralytics(results[0])
                
                # Reset volumes and areas for this frame
                divot_volumes = {}
                divot_areas = {}
                
                if len(detections) > 0:
                    # Process each detection
                    for i, (class_id, mask) in enumerate(zip(detections.class_id, detections.mask)):
                        # Only process regular divots, not fixed divots
                        if class_id != divot_class_id or mask is None:
                            continue
                            
                        # Get depth map for this divot
                        masked_depth, _ = get_divot_depth_map(frame, mask, depth_model)
                        
                        if masked_depth is not None:
                            # Calculate area and volume
                            area_mm2, volume, _ = calculate_area_and_volume(mask, masked_depth, focal_length)
                            
                            # Store results
                            divot_areas[i] = area_mm2
                            divot_volumes[i] = volume
                
                processing_time = time.time() - start_time
                fps_processing = 1.0 / processing_time if processing_time > 0 else 0
                print(f"Frame {frame_count}/{total_frames} - Processing time: {processing_time:.3f}s ({fps_processing:.1f} FPS)")
                print(f"Found {len(divot_volumes)} divots with total volume: {sum(divot_volumes.values() or [0]):.2f} mm³")
            
            # Only create visualization if needed
            if show_visualization:
                # Create visualization image
                annotated_frame = frame.copy()
                
                # Always annotate with the latest available information
                if detections is not None and len(detections) > 0:
                    # Add segmentation mask
                    annotated_frame = mask_annotator.annotate(
                        scene=annotated_frame,
                        detections=detections
                    )
                    
                    # Add bounding boxes
                    annotated_frame = box_annotator.annotate(
                        scene=annotated_frame,
                        detections=detections
                    )
                    
                    # Create labels for detections with measurements
                    labels = []
                    for i, (class_id, confidence) in enumerate(zip(detections.class_id, detections.confidence)):
                        label = f"{class_names[class_id]} {confidence:.2f}"
                        
                        # Add measurement information for regular divots
                        if i in divot_volumes and i in divot_areas and divot_volumes[i] is not None:
                            label += f"\nVol: {divot_volumes[i]:.2f} mm³"
                            label += f"\nArea: {divot_areas[i]:.2f} mm²"
                        
                        labels.append(label)
                    
                    # Apply labels using the label annotator
                    annotated_frame = label_annotator.annotate(
                        scene=annotated_frame,
                        detections=detections,
                        labels=labels
                    )
                
                # Add frame number and processing info
                cv2.putText(
                    annotated_frame,
                    f"Frame: {frame_count}/{total_frames}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                
                # Show visualization
                cv2.imshow('Divot Detection', annotated_frame)
        
        # Handle keyboard input (wait for a short time to allow interaction)
        key = cv2.waitKey(1)
        if key == ord('q'):
            print("Exiting...")
            break
        elif key == ord('p'):
            # Toggle pause
            paused = not paused
            print("Paused" if paused else "Resumed")
            
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Video processing complete. Processed {frame_count} frames.")

def main():
    parser = argparse.ArgumentParser(description='Process a video to detect divots and calculate measurements')
    parser.add_argument('--video', default='videos/1.mp4', 
                        help='Path to the input video file (default: videos/1.mp4)')
    parser.add_argument('--model', default=r"C:\Users\teomeo\Desktop\aMU_MSc\desertation\Yolo11seg\yolo11n_1600_40ep\1600n-aug-40ep.pt", 
                        help='Path to the YOLO model')
    parser.add_argument('--depth-model', default='checkpoints/depth_anything_v2_vits.pth',
                        help='Path to the Depth Anything V2 model file')
    parser.add_argument('--focal-length', type=float, default=525.0, 
                        help='Focal length for depth calculations (default: 525.0)')
    parser.add_argument('--start-frame', type=int, default=0, 
                        help='Frame to start processing from (default: 0)')
    parser.add_argument('--interval', type=int, default=5, 
                        help='Process every Nth frame (default: 5)')
    parser.add_argument('--no-viz', action='store_true', 
                        help='Disable visualization window')
    
    args = parser.parse_args()
    
    process_video(
        args.video,
        args.model,
        args.depth_model,
        args.focal_length,
        args.start_frame,
        not args.no_viz,
        args.interval
    )

if __name__ == "__main__":
    main() 