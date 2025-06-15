import numpy as np
import cv2
import supervision as sv
import torch
import open3d as o3d
import argparse
import os
import sys
import time
import pyrealsense2 as rs
from pathlib import Path
from ultralytics import YOLO

# Add repository to path and import
sys.path.insert(0, './Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2

# Device selection for PyTorch
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def initialize_depth_model(encoder='vits'):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()
    return model

def get_divot_depth_map(color_image, mask, depth_model):
    # Get depth map for the entire image using Depth Anything V2
    depth = depth_model.infer_image(color_image)
    
    # Apply mask to depth map to isolate the divot area
    masked_depth = np.where(mask, depth, 0)
    
    return masked_depth, depth

def initialize_realsense():
    """Initialize the Intel RealSense camera pipeline"""
    # Create a pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    
    print(f"Using RealSense device: {device_product_line}")
    
    # Enable color stream
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    pipeline.start(config)
    
    return pipeline

def calculate_volume_from_depth(depth_map, mask, focal_length=525.0):
    # Get dimensions
    height, width = depth_map.shape
    
    # Create meshgrid of coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Apply mask
    masked_points = mask.copy()
    
    # Only consider points that are part of the divot
    depth_values = depth_map[masked_points]
    x_values = x[masked_points]
    y_values = y[masked_points]
    
    # Calculate 3D coordinates
    Z = depth_values
    X = (x_values - width/2) * Z / focal_length
    Y = (y_values - height/2) * Z / focal_length
    
    # Stack coordinates
    points = np.stack((X, Y, Z), axis=1)
    
    # Remove points with zero depth
    mask = points[:, 2] > 0
    points = points[mask]
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Calculate approximate volume using convex hull
    try:
        hull, _ = pcd.compute_convex_hull()
        volume = hull.get_volume()
        return pcd, volume
    except Exception as e:
        print(f"Error calculating volume: {e}")
        return pcd, None

def process_realsense_frame(yolo_model, depth_model, color_image, output_dir=None, focal_length=525.0, show_visualization=True):
    """Process a single frame from the RealSense camera"""
    # Get timestamp for file naming
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Processing image with shape: {color_image.shape}")
    
    # Initialize annotators
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.CLASS, opacity=0.5)
    label_annotator = sv.LabelAnnotator(
        text_position=sv.Position.TOP_LEFT, 
        text_padding=3,
        text_scale=3.0,  # Increase text scale (default is usually 0.5)
        text_thickness=2  # Increase text thickness for better visibility
    )
    
    # Run YOLO detection
    results = yolo_model(color_image)
    
    # Convert results to supervision Detections
    detections = sv.Detections.from_ultralytics(results[0])
    
    if len(detections) == 0:
        print("No divots detected in the image.")
        return None, None, None
    
    print(f"Detected {len(detections)} divot(s) in the image.")
    
    # Prepare visualization image
    annotated_image = color_image.copy()
    
    # Get class names from YOLO model
    class_names = yolo_model.names
    
    # Get the class_id for 'divot' if it exists
    divot_class_id = None
    fixed_divot_class_id = None
    for id, name in class_names.items():
        if name.lower() == 'divot':
            divot_class_id = id
        elif name.lower() == 'fixed_divot' or name.lower() == 'fixed_divots':
            fixed_divot_class_id = id
    
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
    
    # Create labels with measurements
    labels = []
    
    # Process each detected divot and create labels
    for i, detection_idx in enumerate(range(len(detections))):
        mask = detections.mask[detection_idx] if detections.mask is not None else None
        class_id = detections.class_id[detection_idx] if len(detections.class_id) > 0 else None
        confidence = detections.confidence[detection_idx] if len(detections.confidence) > 0 else None
        
        # Skip fixed divots or detections without masks
        if class_id == fixed_divot_class_id or mask is None:
            labels.append(f"{class_names[class_id]} {confidence:.2f}")
            continue
        
        # Create initial label with class name and confidence
        label = f"{class_names[class_id]} {confidence:.2f}"
        
        if class_id == divot_class_id:
            # Get depth map of the divot
            masked_depth, full_depth = get_divot_depth_map(color_image, mask, depth_model)
            
            # Calculate area in pixels
            area_pixels = np.sum(mask)
            
            # Calculate volume
            pcd, volume = calculate_volume_from_depth(masked_depth, mask, focal_length=focal_length)
            
            # Add measurements to label
            if volume is not None:
                label += f"\nVol: {volume:.2f} mm³"
                label += f"\nArea: {area_pixels:.2f} px²"
                print(f"Divot {i+1} volume: {volume:.2f} cubic millimeters")
            
            # Save point cloud if we have output directory
            if output_dir and pcd is not None and len(pcd.points) > 0:
                pcd_filename = os.path.join(output_dir, f"{timestamp}_divot{i+1}_point_cloud.ply")
                o3d.io.write_point_cloud(pcd_filename, pcd)
                print(f"Saved point cloud to {pcd_filename}")
        
        labels.append(label)
    
    # Add the labels to the image
    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=detections,
        labels=labels
    )
    
    # Save the annotated image if output directory is provided
    if output_dir:
        annotated_filename = os.path.join(output_dir, f"{timestamp}_annotated.jpg")
        cv2.imwrite(annotated_filename, annotated_image)
        print(f"Saved annotated image to {annotated_filename}")
    
    # Also save the color image
    if output_dir:
        color_filename = os.path.join(output_dir, f"{timestamp}_color.jpg")
        cv2.imwrite(color_filename, color_image)
        print(f"Saved color image to {color_filename}")
    
    return annotated_image, full_depth if 'full_depth' in locals() else None, pcd if 'pcd' in locals() else None

def main():
    parser = argparse.ArgumentParser(description='Capture from Intel RealSense camera and detect divots')
    parser.add_argument('--output', default='detected_divots_intel', help='Output directory for saved images and data')
    parser.add_argument('--yolo-model', default=r"yolo11n_1600_40ep/1600n-aug-40ep.pt", help='Path to YOLO model')
    parser.add_argument('--depth-model', default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'], help='Depth model to use')
    parser.add_argument('--focal-length', type=float, default=525.0, help='Focal length for depth calculations')
    
    args = parser.parse_args()
    
    # Initialize models
    print(f"Initializing YOLO model from {args.yolo_model}")
    yolo_model = YOLO(args.yolo_model)
    
    print(f"Initializing depth model ({args.depth_model})")
    depth_model = initialize_depth_model(args.depth_model)
    
    # Initialize RealSense camera
    try:
        pipeline = initialize_realsense()
    except Exception as e:
        print(f"Error initializing RealSense camera: {e}")
        print("Make sure the camera is connected and the pyrealsense2 package is installed correctly.")
        print("Install with: pip install pyrealsense2")
        return
    
    # Create visualization window
    cv2.namedWindow('RealSense Divot Detection', cv2.WINDOW_AUTOSIZE)
    
    print("\nCamera stream is active:")
    print("- Press SPACE to capture and process an image")
    print("- Press ESC to exit")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    try:
        while True:
            # Wait for a coherent pair of frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
            
            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            
            # Show the live feed
            cv2.imshow('RealSense Divot Detection', color_image)
            
            # Handle keyboard input
            key = cv2.waitKey(1)
            
            # ESC to exit
            if key == 27:
                print("Exiting...")
                break
            
            # SPACE to capture and process
            if key == 32:
                print("\nCapturing and processing image...")
                
                # Make a copy of the current frame
                captured_image = color_image.copy()
                
                # Process the image
                annotated_image, depth_map, _ = process_realsense_frame(
                    yolo_model=yolo_model,
                    depth_model=depth_model,
                    color_image=captured_image,
                    output_dir=args.output,
                    focal_length=args.focal_length
                )
                
                # Show the processed image
                if annotated_image is not None:
                    cv2.imshow('Processed Image', annotated_image)
                    
                    # If we have a depth map, visualize it
                    if depth_map is not None:
                        # Normalize depth for visualization
                        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
                        depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)
                        cv2.imshow('Depth Map', depth_colored)
                
                print("Processing complete. Ready for next capture.")
                
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 