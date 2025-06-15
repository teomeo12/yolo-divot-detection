import numpy as np
import cv2
import supervision as sv
import pyrealsense2 as rs
import argparse
import os
import time
from ultralytics import YOLO

# Standard measurements for calibration
CREDIT_CARD_WIDTH_MM = 85.6  # Standard credit card width in mm
CREDIT_CARD_HEIGHT_MM = 53.98  # Standard credit card height in mm

def initialize_realsense():
    """Initialize the Intel RealSense camera pipeline with color stream"""
    # Create a pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    
    print(f"Using RealSense device: {device_product_line}")
     
    # Enable color stream only
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    profile = pipeline.start(config)
    
    return pipeline

def calibrate_with_object(pipeline, object_type='credit_card'):
    """
    Calibration procedure using a credit card or measuring tape
    
    Returns:
        mm_per_pixel - Calculated millimeters per pixel
    """
    # Set up display window
    cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
    
    # Define object dimensions based on type
    if object_type == 'credit_card':
        object_width_mm = CREDIT_CARD_WIDTH_MM
        object_height_mm = CREDIT_CARD_HEIGHT_MM
        calibration_text = f"Place a credit card in the rectangle (width={object_width_mm}mm, height={object_height_mm}mm)"
    else:
        object_width_mm = 100  # Default to 10cm for measuring tape
        object_height_mm = 20   # Approximate height of measuring tape
        calibration_text = f"Place a 10cm section of measuring tape in the rectangle"
    
    # Calculate center rectangle for placement guide
    def draw_guide(image):
        h, w = image.shape[:2]
        
        # Calculate rectangle in center (approximately credit card sized)
        if object_type == 'credit_card':
            # Credit card has roughly 1.6:1 aspect ratio, adjust for screen
            rect_w = w // 3
            rect_h = int(rect_w / 1.6)
        else:
            # For measuring tape, make a longer rectangle
            rect_w = w // 2
            rect_h = h // 8
            
        x1 = (w - rect_w) // 2
        y1 = (h - rect_h) // 2
        x2 = x1 + rect_w
        y2 = y1 + rect_h
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add text instructions
        cv2.putText(image, calibration_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, "Press SPACE when object is aligned", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, "Press ESC to cancel", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return image, (x1, y1, x2, y2)
    
    # Main calibration loop
    print(f"\n===== CALIBRATION MODE =====")
    print(f"1. {calibration_text}")
    print(f"2. Press SPACE when the object is properly aligned in the green rectangle")
    print(f"3. Press ESC to cancel calibration\n")
    
    while True:
        # Get frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        
        if not color_frame:
            continue
        
        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        
        # Draw guide rectangle
        guide_image, rect_coords = draw_guide(color_image.copy())
        
        # Show image
        cv2.imshow('Calibration', guide_image)
        
        # Handle keyboard input
        key = cv2.waitKey(1)
        
        # ESC to cancel
        if key == 27:
            print("Calibration cancelled")
            cv2.destroyWindow('Calibration')
            return None
        
        # SPACE to capture
        if key == 32:
            # Get rectangle dimensions in pixels
            x1, y1, x2, y2 = rect_coords
            rect_width_px = x2 - x1
            rect_height_px = y2 - y1
            
            # Calculate mm per pixel
            width_mm_per_px = object_width_mm / rect_width_px
            height_mm_per_px = object_height_mm / rect_height_px
            
            # Use average of width and height
            mm_per_pixel = (width_mm_per_px + height_mm_per_px) / 2
            
            print(f"\nCalibration Results:")
            print(f"Rectangle size: {rect_width_px}x{rect_height_px} pixels")
            print(f"Object size: {object_width_mm}x{object_height_mm} mm")
            print(f"Width calibration: {width_mm_per_px:.4f} mm/pixel")
            print(f"Height calibration: {height_mm_per_px:.4f} mm/pixel")
            print(f"Average calibration: {mm_per_pixel:.4f} mm/pixel")
            
            # Save calibration image
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            cv2.imwrite(f"calibration_{timestamp}.jpg", guide_image)
            print(f"Saved calibration image to calibration_{timestamp}.jpg")
            
            # Save calibration value to file for future use
            with open("camera_calibration.txt", "w") as f:
                f.write(f"{mm_per_pixel:.6f}")
            print(f"Saved calibration value to camera_calibration.txt")
            
            cv2.destroyWindow('Calibration')
            return mm_per_pixel

def calculate_area(detections, mm_per_pixel=1.0):
    """
    Calculate area of detections in both pixels and approximate square millimeters
    
    Parameters:
        detections - supervision Detections object
        mm_per_pixel - Calibration factor (mm per pixel)
        
    Returns:
        areas_px - List of areas in pixels
        areas_mm2 - List of approximate areas in square millimeters
    """
    areas_px = []
    areas_mm2 = []
    
    for i in range(len(detections)):
        # Get mask if available
        mask = detections.mask[i] if detections.mask is not None else None
        
        if mask is not None:
            # Calculate area in pixels
            area_px = float(np.sum(mask))
            
            # Calculate approximate real-world area using calibration factor
            area_mm2 = area_px * (mm_per_pixel ** 2)
            
            areas_px.append(area_px)
            areas_mm2.append(area_mm2)
        else:
            # If no mask is available, calculate from bounding box
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            width_px = x2 - x1
            height_px = y2 - y1
            area_px = width_px * height_px
            
            # Calculate approximate real-world area
            area_mm2 = area_px * (mm_per_pixel ** 2)
            
            areas_px.append(area_px)
            areas_mm2.append(area_mm2)
    
    return areas_px, areas_mm2

def process_frame(color_image, yolo_model, mm_per_pixel=1.0, output_dir=None):
    """Process a single frame for area measurement"""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize annotators
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.CLASS, opacity=0.5)
    label_annotator = sv.LabelAnnotator(
        text_position=sv.Position.TOP_LEFT, 
        text_padding=3,
        text_scale=1.0,
        text_thickness=2
    )
    
    # Run YOLO detection
    results = yolo_model(color_image)
    
    # Convert results to supervision Detections
    detections = sv.Detections.from_ultralytics(results[0])
    
    if len(detections) == 0:
        print("No divots detected in the image.")
        return None
    
    print(f"Detected {len(detections)} divot(s) in the image.")
    
    # Calculate areas
    areas_px, areas_mm2 = calculate_area(detections, mm_per_pixel)
    
    # Prepare visualization image
    annotated_image = color_image.copy()
    
    # Get class names from YOLO model
    class_names = yolo_model.names
    
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
    
    # Process each detected divot
    for i, detection_idx in enumerate(range(len(detections))):
        class_id = detections.class_id[detection_idx] if len(detections.class_id) > 0 else None
        confidence = detections.confidence[detection_idx] if len(detections.confidence) > 0 else None
        
        # Create initial label with class name and confidence
        label = f"{class_names[class_id]} {confidence:.2f}"
        
        # Add area measurements
        label += f"\nArea: {areas_px[i]:.0f} px²"
        
        # If calibrated, add real-world area
        if mm_per_pixel != 1.0:
            label += f"\nArea: {areas_mm2[i]:.1f} mm²"
        
        # Calculate dimensions of bounding box
        xyxy = detections.xyxy[detection_idx]
        x1, y1, x2, y2 = map(int, xyxy)
        width_px = x2 - x1
        height_px = y2 - y1
        
        # Add dimensions information
        label += f"\nSize: {width_px}×{height_px} px"
        
        if mm_per_pixel != 1.0:
            width_mm = width_px * mm_per_pixel
            height_mm = height_px * mm_per_pixel
            label += f"\nSize: {width_mm:.1f}×{height_mm:.1f} mm"
        
        # Print to console
        print(f"Divot {i+1}: {width_px}×{height_px} px = {areas_px[i]:.0f} px²")
        if mm_per_pixel != 1.0:
            print(f"   Dimensions: {width_mm:.1f}×{height_mm:.1f} mm = {areas_mm2[i]:.1f} mm²")
        
        labels.append(label)
    
    # Add calibration information on the image
    if mm_per_pixel != 1.0:
        cv2.putText(
            annotated_image, 
            f"Calibration: {mm_per_pixel:.4f} mm/px", 
            (10, annotated_image.shape[0] - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (255, 255, 255), 
            1
        )
    
    # Add the labels to the image
    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=detections,
        labels=labels
    )
    
    # Save the annotated image if output directory is provided
    if output_dir:
        annotated_filename = os.path.join(output_dir, f"{timestamp}_measured.jpg")
        cv2.imwrite(annotated_filename, annotated_image)
        print(f"Saved annotated image to {annotated_filename}")
        
        # Also save the original image
        color_filename = os.path.join(output_dir, f"{timestamp}_color.jpg")
        cv2.imwrite(color_filename, color_image)
    
    return annotated_image

def main():
    parser = argparse.ArgumentParser(description='Measure divot areas using Intel RealSense camera')
    parser.add_argument('--output', default='divot_measurements', help='Output directory for saved images')
    parser.add_argument('--yolo-model', default=r"yolo11n_1600_40ep/1600n-aug-40ep.pt", help='Path to YOLO model')
    parser.add_argument('--calibration', type=float, default=None, 
                        help='Manual calibration factor (mm per pixel).')
    parser.add_argument('--calibrate', action='store_true', 
                        help='Enter calibration mode using credit card')
    parser.add_argument('--tape', action='store_true',
                        help='Use measuring tape for calibration (10cm section)')
    
    args = parser.parse_args()
    
    # Initialize YOLO model
    print(f"Initializing YOLO model from {args.yolo_model}")
    yolo_model = YOLO(args.yolo_model)
    
    # Initialize RealSense camera
    try:
        pipeline = initialize_realsense()
    except Exception as e:
        print(f"Error initializing RealSense camera: {e}")
        print("Make sure the camera is connected and the pyrealsense2 package is installed correctly.")
        print("Install with: pip install pyrealsense2")
        return
    
    # Calibration process
    mm_per_pixel = None
    
    # First check if a saved calibration file exists
    if os.path.exists("camera_calibration.txt") and not args.calibrate and args.calibration is None:
        try:
            with open("camera_calibration.txt", "r") as f:
                mm_per_pixel = float(f.read().strip())
            print(f"Loaded saved calibration: {mm_per_pixel:.4f} mm/pixel")
        except:
            print("Error loading saved calibration file.")
    
    # If calibration mode is requested
    if args.calibrate:
        object_type = 'measuring_tape' if args.tape else 'credit_card'
        mm_per_pixel = calibrate_with_object(pipeline, object_type)
        if mm_per_pixel is None:
            print("Using default 1.0 mm/pixel (uncalibrated)")
            mm_per_pixel = 1.0
    
    # If manual calibration is provided
    if args.calibration is not None:
        mm_per_pixel = args.calibration
        print(f"Using manual calibration: {mm_per_pixel:.4f} mm/pixel")
    
    # If no calibration is set, use default
    if mm_per_pixel is None:
        mm_per_pixel = 1.0
        print("No calibration provided. Measurements will be in pixels only.")
        print("Use --calibrate to calibrate with a credit card")
        print("Use --calibrate --tape to calibrate with a measuring tape")
        print("Use --calibration VALUE to set calibration manually")
    
    # Create visualization window
    cv2.namedWindow('RealSense Divot Measurement', cv2.WINDOW_AUTOSIZE)
    
    print("\nCamera stream is active:")
    print("- Press SPACE to capture and measure an image")
    print("- Press ESC to exit")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    try:
        while True:
            # Wait for a coherent frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
            
            # Convert image to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            
            # Add calibration info to live view
            display_image = color_image.copy()
            if mm_per_pixel != 1.0:
                cv2.putText(
                    display_image, 
                    f"Calibration: {mm_per_pixel:.4f} mm/px", 
                    (10, display_image.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
            
            # Show the live feed
            cv2.imshow('RealSense Divot Measurement', display_image)
            
            # Handle keyboard input
            key = cv2.waitKey(1)
            
            # ESC to exit
            if key == 27:
                print("Exiting...")
                break
            
            # SPACE to capture and process
            if key == 32:
                print("\nCapturing and measuring divots...")
                
                # Process the frame
                annotated_image = process_frame(
                    color_image,
                    yolo_model,
                    mm_per_pixel=mm_per_pixel,
                    output_dir=args.output
                )
                
                # Show the processed image
                if annotated_image is not None:
                    cv2.imshow('Measured Divots', annotated_image)
                
                print("Measurement complete. Ready for next capture.")
                
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 