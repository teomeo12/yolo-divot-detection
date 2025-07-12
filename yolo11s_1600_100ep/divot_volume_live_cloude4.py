import pyrealsense2 as rs
import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO

# --- Tweakable Parameters ---
# Calibration factor for 1D distance, based on your previous tests.
CALIBRATION_FACTOR = 0.67
# The area calibration factor is the square of the linear one.
AREA_CALIBRATION_FACTOR = CALIBRATION_FACTOR ** 2

# Morphological Kernel size for cleaning up the mask
MORPH_KERNEL = np.ones((5, 5), np.uint8)

# How many pixels to expand the mask to find the surrounding "ground"
GROUND_RING_WIDTH = 15
# --- End of Parameters ---

def initialize_camera():
    """Initialize RealSense camera with aligned depth and color streams"""
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    print("Starting pipeline...")
    profile = pipeline.start(config)
    
    # Create an align object to map depth to color
    align = rs.align(rs.stream.color)
    
    # Get camera intrinsics for volume calculation
    intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    fx = intrinsics.fx  # Focal length in x
    fy = intrinsics.fy  # Focal length in y
    
    print(f"fx: {fx}, fy: {fy}")
    print("Pipeline started. Press 'q' to quit.")
    
    return pipeline, align, fx, fy

def convert_yolo_mask_to_contour(mask):
    """Convert YOLO segmentation mask to contour format"""
    if mask is None:
        return None
    
    # Convert boolean mask to uint8
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Clean mask using morphological operations
    mask_clean = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, MORPH_KERNEL)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, MORPH_KERNEL)
    
    # Find contours
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Return the largest contour (assuming it's the main divot)
        return max(contours, key=cv2.contourArea), mask_clean
    
    return None, mask_clean

def calculate_volume_and_depth(divot_contour, divot_mask, depth_image, fx, fy):
    """Calculate volume and depth measurements for the detected divot"""
    if divot_contour is None:
        return None
    
    # --- AUTOMATIC GROUND PLANE DETECTION ---
    # Create a mask for just the divot
    divot_mask_binary = np.zeros(depth_image.shape, dtype="uint8")
    cv2.drawContours(divot_mask_binary, [divot_contour], -1, 255, -1)
    
    # Dilate the divot mask to get a "ring" around it
    ring_kernel = np.ones((GROUND_RING_WIDTH, GROUND_RING_WIDTH), np.uint8)
    dilated_mask = cv2.dilate(divot_mask_binary, ring_kernel, iterations=1)
    # The ground ring is the dilated area minus the divot area
    ground_ring_mask = cv2.bitwise_and(dilated_mask, cv2.bitwise_not(divot_mask_binary))
    
    # Get depth values for the ground ring
    ground_depths = depth_image[ground_ring_mask == 255]
    ground_depths_mm = ground_depths[ground_depths != 0]  # Filter out invalid 0 values
    
    if ground_depths_mm.size == 0:
        return {
            'error': 'No ground plane detected',
            'ground_ring_mask': ground_ring_mask
        }
    
    # This is our reference plane!
    ground_level_mm = np.mean(ground_depths_mm)
    
    # --- VOLUME CALCULATION AND SINGLE-POINT DEPTH CHECK ---
    total_volume_cm3 = 0.0
    total_area_cm2 = 0.0
    divot_point_depth_mm = 0
    
    # --- Get a single point depth reference from inside the divot ---
    # Calculate the centroid of the contour to get a sample point
    M = cv2.moments(divot_contour)
    cx, cy = 0, 0
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Get the depth at the centroid
        centroid_depth_mm = depth_image[cy, cx]
        if centroid_depth_mm > 0:
            # Calculate depth relative to the ground plane
            divot_point_depth_mm = centroid_depth_mm - ground_level_mm
    
    # Get all pixel depths inside the divot mask
    divot_depths = depth_image[divot_mask_binary == 255]
    divot_depths_mm = divot_depths[divot_depths != 0]
    
    # Calculate depth difference relative to the ground
    depth_differences_mm = divot_depths_mm - ground_level_mm
    
    # Consider only points that are deeper than the ground
    hole_depths_mm = depth_differences_mm[depth_differences_mm > 0]
    
    if hole_depths_mm.size > 0:
        # Estimate the area of one pixel at the ground depth
        # This is an approximation but good enough for this purpose
        pixel_area_m2 = ((ground_level_mm / 1000) / fx) * ((ground_level_mm / 1000) / fy)
        pixel_area_cm2 = pixel_area_m2 * 10000
        
        # Apply calibration factor to the raw calculations
        total_volume_cm3 = (np.sum(hole_depths_mm / 10) * pixel_area_cm2) * AREA_CALIBRATION_FACTOR
        total_area_cm2 = (hole_depths_mm.size * pixel_area_cm2) * AREA_CALIBRATION_FACTOR
    
    return {
        'ground_level_mm': ground_level_mm,
        'total_area_cm2': total_area_cm2,
        'total_volume_cm3': total_volume_cm3,
        'divot_point_depth_mm': divot_point_depth_mm,
        'centroid': (cx, cy),
        'ground_ring_mask': ground_ring_mask
    }

def main():
    # Initialize YOLO model - Use absolute path to avoid directory issues
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "1600s_aug_100ep.pt")
    model = YOLO(model_path)
    
    # Initialize annotators
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.CLASS, opacity=0.3)
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT, text_padding=3)
    
    try:
        # Initialize camera
        pipeline, align, fx, fy = initialize_camera()
        
        while True:
            # Get aligned frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            display_image = color_image.copy()
            
            # Run YOLO detection
            results = model(color_image)
            
            # Convert results to supervision Detections
            detections = sv.Detections.from_ultralytics(results[0])
            
            # Get class names from the model
            class_names = results[0].names
            
            # Process detections - volume calculation only for 'divot' class
            if len(detections) > 0:
                # Find divot detections (class_id for 'divot')
                divot_class_id = None
                for class_id, class_name in class_names.items():
                    if class_name == 'divot':
                        divot_class_id = class_id
                        break
                
                if divot_class_id is not None:
                    # Filter to get only 'divot' class detections
                    divot_mask = detections.class_id == divot_class_id
                    divot_detections = detections[divot_mask]
                    
                    if len(divot_detections) > 0:
                        # Process each divot detection individually
                        for detection_idx in range(len(divot_detections)):
                            if divot_detections.mask is not None and len(divot_detections.mask) > detection_idx:
                                # Convert YOLO mask to contour
                                divot_contour, mask_clean = convert_yolo_mask_to_contour(divot_detections.mask[detection_idx])
                                
                                if divot_contour is not None:
                                    # Calculate volume and depth
                                    measurements = calculate_volume_and_depth(divot_contour, mask_clean, depth_image, fx, fy)
                                    
                                    if measurements and 'error' not in measurements:
                                        # --- VISUALIZATION FOR VOLUME CALCULATION ---
                                        # Draw the divot contour
                                        cv2.drawContours(display_image, [divot_contour], -1, (255, 0, 0), 2)  # Blue
                                        
                                        # Draw the ground ring for visual feedback
                                        display_image[measurements['ground_ring_mask'] == 255] = [0, 255, 0]  # Green
                                        
                                        # Draw a red dot at the centroid for the point depth measurement
                                        cx, cy = measurements['centroid']
                                        if cx > 0 and cy > 0:
                                            cv2.circle(display_image, (cx, cy), 5, (0, 0, 255), -1)
                                            
                                            # Get bounding box for text positioning
                                            bbox = divot_detections.xyxy[detection_idx]
                                            x1, y1, x2, y2 = map(int, bbox)
                                            
                                            # Position text just below the class label (which appears at top-left of bbox)
                                            text_x = x1
                                            text_y = y1 + 45  # Start below the class label
                                            
                                            # Display the results next to this specific divot
                                            cv2.putText(display_image, f"A: {measurements['total_area_cm2']:.1f}cm^2", 
                                                       (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                                            cv2.putText(display_image, f"V: {measurements['total_volume_cm3']:.2f}cm^3", 
                                                       (text_x, text_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                                            if measurements['divot_point_depth_mm'] > 0:
                                                cv2.putText(display_image, f"D: {measurements['divot_point_depth_mm']:.1f}mm", 
                                                           (text_x, text_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                                    else:
                                        # Display error message next to the divot
                                        bbox = divot_detections.xyxy[detection_idx]
                                        x1, y1, x2, y2 = map(int, bbox)
                                        text_x = x1
                                        text_y = y1 + 45
                                        
                                        error_msg = measurements.get('error', 'Unknown error') if measurements else 'No measurements'
                                        cv2.putText(display_image, error_msg, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Add YOLO annotations (optional, can be commented out for cleaner view)
            display_image = mask_annotator.annotate(scene=display_image, detections=detections)
            display_image = box_annotator.annotate(scene=display_image, detections=detections)
            display_image = label_annotator.annotate(scene=display_image, detections=detections)
            
            # Show the annotated stream
            cv2.imshow('YOLO Divot Detection with Volume Analysis', display_image)
            
            # Break loop with 'q'
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