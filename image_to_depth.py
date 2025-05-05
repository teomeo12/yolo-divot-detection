import numpy as np
import cv2
import supervision as sv
import torch
import open3d as o3d
import argparse
import os
import sys
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

def process_image(image_path, yolo_model_path, output_dir, focal_length=525.0, show_visualization=True):
    # Determine the output prefix based on the image filename
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_prefix = os.path.join(output_dir, image_name)
    
    print(f"Processing image: {image_path}")
    print(f"Using YOLO model: {yolo_model_path}")
    print(f"Output will be saved to: {output_dir}")
    
    # Load image
    color_image = cv2.imread(image_path)
    if color_image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Initialize models
    yolo_model = YOLO(yolo_model_path)
    depth_model = initialize_depth_model()
    
    print(f"Image shape: {color_image.shape}")
    print(f"Using device: {DEVICE}")
    
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
        return
    
    print(f"Detected {len(detections)} divot(s) in the image.")
    
    # Prepare visualization image
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
    
    # Process each detected divot
    for i, detection_idx in enumerate(range(len(detections))):
        mask = detections.mask[detection_idx] if detections.mask is not None else None
        class_id = detections.class_id[detection_idx] if len(detections.class_id) > 0 else None
        
        if mask is None:
            print(f"No segmentation mask for detection {i}")
            continue
        
        print(f"Processing divot {i+1} (class ID: {class_id})")
        
        # Get depth map of the divot
        masked_depth, full_depth = get_divot_depth_map(color_image, mask, depth_model)
        
        # Calculate area of the mask in pixels
        area_pixels = np.sum(mask)
        
        # Approximate real-world area (assuming square pixels)
        # Estimate pixel size based on depth and focal length
        if np.sum(masked_depth) > 0:
            avg_depth = np.sum(masked_depth) / np.sum(masked_depth > 0)
            # Calculate pixel size in mm at the average depth
            pixel_size_mm = avg_depth / focal_length
            # Calculate area in square mm
            area_mm2 = area_pixels * (pixel_size_mm ** 2)
        else:
            area_mm2 = 0
            
        print(f"Divot {i+1} area: {area_mm2:.2f} square millimeters")
        
        # Calculate volume
        pcd, volume = calculate_volume_from_depth(masked_depth, mask, focal_length=focal_length)
        
        # Prepare depth visualization
        depth_normalized = cv2.normalize(masked_depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_scaled = depth_normalized.astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_INFERNO)
        
        # Create a visualization with the bounding box coordinates
        xyxy = detections.xyxy[detection_idx]
        x1, y1, x2, y2 = map(int, xyxy)
        
        # Add volume information to the visualization
        if volume is not None:
            volume_text = f"Vol: {volume:.2f} mm^3"
            area_text = f"Area: {area_mm2:.2f} mm^2"
            print(f"Divot {i+1} volume: {volume:.2f} cubic millimeters")
            
            # Add text to the annotated image - just larger size
            # Calculate center of bounding box
            center_x = x1 + (x2 - x1) // 2
            center_y = y1 + (y2 - y1) // 2
            
            # Position text at the center of the bounding box
            cv2.putText(
                annotated_image, 
                volume_text, 
                (center_x - (len(volume_text) * 11) // 2, center_y - 20),  # Center text above the middle
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0,  # Increased text size
                (255, 255, 255),  # White color
                2
            )
            
            # Add area text below volume
            cv2.putText(
                annotated_image, 
                area_text, 
                (center_x - (len(area_text) * 11) // 2, center_y + 20),  # Center text below the middle
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.0,  # Same text size as volume
                (255, 255, 255),  # White color
                2
            )
        else:
            print(f"Could not calculate volume for divot {i+1}")
        
        # Save depth map and point cloud
        mask_depth_filename = f"{output_prefix}_divot{i+1}_depth_map.jpg"
        cv2.imwrite(mask_depth_filename, depth_colored)
        print(f"Saved masked depth map to {mask_depth_filename}")
        
        # Save point cloud to PLY format
        if pcd is not None and len(pcd.points) > 0:
            pcd_filename = f"{output_prefix}_divot{i+1}_point_cloud.ply"
            o3d.io.write_point_cloud(pcd_filename, pcd)
            print(f"Saved point cloud to {pcd_filename}")
    
    # Save annotated image
    annotated_filename = f"{output_prefix}_annotated.jpg"
    cv2.imwrite(annotated_filename, annotated_image)
    print(f"Saved annotated image to {annotated_filename}")
    
    # Show visualization if requested
    if show_visualization:
        # Resize windows for better viewing
        cv2.namedWindow('Detected Divots', cv2.WINDOW_NORMAL)
        cv2.imshow('Detected Divots', annotated_image)
        
        # Resize for comfort - adjust based on your screen
        cv2.resizeWindow('Detected Divots', 800, 600)
        
        if len(detections) > 0:
            cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)
            cv2.imshow('Depth Map', depth_colored)
            cv2.resizeWindow('Depth Map', 800, 600)
        
        print("Press any key to continue to the next image...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return annotated_image, depth_colored if len(detections) > 0 else None, pcd if len(detections) > 0 else None

def process_all_images(input_dir, output_dir, yolo_model_path, focal_length=525.0, show_visualization=True):
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files from the input directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(Path(input_dir).glob(f'*{ext}')))
        image_files.extend(list(Path(input_dir).glob(f'*{ext.upper()}')))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for image_path in image_files:
        process_image(
            str(image_path),
            yolo_model_path,
            output_dir,
            focal_length,
            show_visualization
        )
        print("-" * 50)

def main():
    model_path = r"C:\Users\teomeo\Desktop\aMU_MSc\desertation\Yolo11seg\yolo11n_1600_40ep\1600n-aug-40ep.pt"
    
    args = argparse.Namespace(
        input_dir='images',
        output_dir='detected_divots',
        model=model_path,
        focal_length=525.0,
        no_viz=False
    )
    
    process_all_images(
        args.input_dir,
        args.output_dir,
        args.model,
        args.focal_length,
        not args.no_viz
    )

if __name__ == "__main__":
    main() 