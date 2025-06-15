import numpy as np
import cv2
import pyrealsense2 as rs
import argparse
import os
import time

# Known object dimensions for reference
CREDIT_CARD_WIDTH_MM = 85.6  # Standard credit card width in mm
CREDIT_CARD_HEIGHT_MM = 53.98  # Standard credit card height in mm

def initialize_realsense():
    """Initialize the Intel RealSense camera pipeline with color and depth streams"""
    # Create a pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    
    print(f"Using RealSense device: {device_product_line}")
     
    # Enable streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    profile = pipeline.start(config)
    
    # Get depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth Scale is: {depth_scale}")
    
    # Create alignment object
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    return pipeline, align, depth_scale

def detect_objects(color_image, min_area=5000, max_area=100000):
    """
    Detect potential objects in the image using contour detection
    
    Parameters:
        color_image - RGB image
        min_area - Minimum contour area to consider
        max_area - Maximum contour area to consider
        
    Returns:
        rectangles - List of rectangle coordinates (x1, y1, x2, y2)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    
    # Apply slight blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Alternative: Use Canny edge detection
    edges = cv2.Canny(blurred, 30, 100)
    
    # Find contours in both threshold and edge images
    contours_thresh, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_edges, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Combine both sets of contours
    all_contours = contours_thresh + contours_edges
    
    rectangles = []
    for contour in all_contours:
        # Calculate contour area
        area = cv2.contourArea(contour)
        
        # Filter by area
        if min_area < area < max_area:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Add to list if not already present
            rect = (x, y, x+w, y+h)
            if rect not in rectangles:
                rectangles.append(rect)
    
    # Merge overlapping rectangles
    merged_rectangles = []
    for rect in rectangles:
        should_merge = False
        for i, merged_rect in enumerate(merged_rectangles):
            # Check overlap
            if rectangles_overlap(rect, merged_rect):
                # Merge by taking the outer bounds
                merged_rectangles[i] = (
                    min(rect[0], merged_rect[0]),
                    min(rect[1], merged_rect[1]),
                    max(rect[2], merged_rect[2]),
                    max(rect[3], merged_rect[3])
                )
                should_merge = True
                break
        
        if not should_merge:
            merged_rectangles.append(rect)
    
    # Filter by aspect ratio (eliminate very thin or wide rectangles)
    filtered_rectangles = []
    for x1, y1, x2, y2 in merged_rectangles:
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 0
        
        # Keep only rectangles with reasonable aspect ratios (0.2 to 5)
        if 0.2 < aspect_ratio < 5:
            filtered_rectangles.append((x1, y1, x2, y2))
    
    return filtered_rectangles

def rectangles_overlap(rect1, rect2):
    """Check if two rectangles overlap"""
    x1_1, y1_1, x2_1, y2_1 = rect1
    x1_2, y1_2, x2_2, y2_2 = rect2
    
    # Check if one rectangle is to the left of the other
    if x2_1 < x1_2 or x2_2 < x1_1:
        return False
    
    # Check if one rectangle is above the other
    if y2_1 < y1_2 or y2_2 < y1_1:
        return False
    
    return True

def measure_object(depth_frame, rect, depth_scale, intrinsics):
    """
    Measure the real-world dimensions of an object
    
    Parameters:
        depth_frame - RealSense depth frame
        rect - Rectangle coordinates (x1, y1, x2, y2)
        depth_scale - Depth scale from RealSense sensor
        intrinsics - Camera intrinsics
        
    Returns:
        width_mm, height_mm, depth_mm - Dimensions in millimeters
    """
    x1, y1, x2, y2 = rect
    
    # Get the center point
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    # Get corners of the rectangle
    corners = [
        (x1, y1),  # top-left
        (x2, y1),  # top-right
        (x1, y2),  # bottom-left
        (x2, y2),  # bottom-right
        (center_x, center_y)  # center
    ]
    
    # Get depth at the corners and center
    corner_depths = []
    for x, y in corners:
        # Get depth value at this pixel (in meters)
        depth_value = depth_frame.get_distance(x, y)
        if depth_value == 0:  # Invalid depth
            # Try to get average depth from surrounding pixels
            depths = []
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < depth_frame.width and 0 <= new_y < depth_frame.height:
                        d = depth_frame.get_distance(new_x, new_y)
                        if d > 0:
                            depths.append(d)
            depth_value = np.mean(depths) if depths else 0
        corner_depths.append(depth_value)
    
    # Filter out any zero depth values and average the rest
    valid_depths = [d for d in corner_depths if d > 0]
    if not valid_depths:
        print("Warning: Could not get valid depth for this object")
        return None, None, None
    
    avg_depth = np.median(valid_depths)  # Use median for more robustness
    
    # Deproject corners to 3D points using camera intrinsics
    corner_3d_points = []
    for i, (x, y) in enumerate(corners[:4]):  # Use only the four corners
        point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], avg_depth)
        corner_3d_points.append(point)
    
    # Calculate dimensions in 3D space
    # Width: distance between top-left and top-right
    width_mm = np.sqrt(
        (corner_3d_points[1][0] - corner_3d_points[0][0])**2 + 
        (corner_3d_points[1][1] - corner_3d_points[0][1])**2 + 
        (corner_3d_points[1][2] - corner_3d_points[0][2])**2
    ) * 1000  # Convert meters to mm
    
    # Height: distance between top-left and bottom-left
    height_mm = np.sqrt(
        (corner_3d_points[2][0] - corner_3d_points[0][0])**2 + 
        (corner_3d_points[2][1] - corner_3d_points[0][1])**2 + 
        (corner_3d_points[2][2] - corner_3d_points[0][2])**2
    ) * 1000  # Convert meters to mm
    
    # Depth is just the average depth
    depth_mm = avg_depth * 1000  # Convert meters to mm
    
    return width_mm, height_mm, depth_mm

def analyze_object(color_image, depth_frame, rect, depth_scale, intrinsics):
    """Analyze object details"""
    x1, y1, x2, y2 = rect
    width_px = x2 - x1
    height_px = y2 - y1
    
    # Get real-world dimensions
    width_mm, height_mm, depth_mm = measure_object(depth_frame, rect, depth_scale, intrinsics)
    
    # Check if this might be a credit card based on dimensions and aspect ratio
    credit_card_match = False
    match_percentage = 0
    if width_mm is not None and height_mm is not None:
        # Calculate aspect ratios
        actual_aspect = width_mm / height_mm if height_mm > 0 else 0
        card_aspect = CREDIT_CARD_WIDTH_MM / CREDIT_CARD_HEIGHT_MM
        
        # Check how close dimensions and aspect ratio are to a credit card
        width_diff = abs(width_mm - CREDIT_CARD_WIDTH_MM) / CREDIT_CARD_WIDTH_MM
        height_diff = abs(height_mm - CREDIT_CARD_HEIGHT_MM) / CREDIT_CARD_HEIGHT_MM
        aspect_diff = abs(actual_aspect - card_aspect) / card_aspect
        
        # Combined difference score (lower is better)
        total_diff = (width_diff + height_diff + aspect_diff) / 3
        
        # Calculate match percentage
        match_percentage = max(0, min(100, 100 * (1 - total_diff)))
        
        # Consider it a match if total difference is small enough
        if total_diff < 0.3:  # 30% tolerance
            credit_card_match = True
    
    results = {
        "rect": rect,
        "width_px": width_px,
        "height_px": height_px,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "depth_mm": depth_mm,
        "credit_card_match": credit_card_match,
        "match_percentage": match_percentage
    }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Measure objects using Intel RealSense camera')
    parser.add_argument('--output', default='measured_objects', help='Output directory for saved images')
    parser.add_argument('--min-area', type=int, default=5000, help='Minimum contour area to consider')
    parser.add_argument('--max-area', type=int, default=100000, help='Maximum contour area to consider')
    
    args = parser.parse_args()
    
    # Initialize RealSense camera
    try:
        pipeline, align, depth_scale = initialize_realsense()
    except Exception as e:
        print(f"Error initializing RealSense camera: {e}")
        print("Make sure the camera is connected and the pyrealsense2 package is installed correctly.")
        print("Install with: pip install pyrealsense2")
        return
    
    # Create visualization window
    cv2.namedWindow('Object Measurement', cv2.WINDOW_AUTOSIZE)
    
    # Create output directory
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Print instructions
    print("\n=== Object Measurement Tool ===")
    print("Place objects in front of the camera to measure them.")
    print("- Press SPACE to capture the current frame with measurements")
    print("- Press 'r' to reset the detection parameters")
    print("- Press ESC to exit")
    
    min_area = args.min_area
    max_area = args.max_area
    
    try:
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            
            # Align depth frame to color frame
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # Get camera intrinsics for 3D calculations
            color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Detect objects
            display_image = color_image.copy()
            rectangles = detect_objects(color_image, min_area, max_area)
            
            # Analyze and measure each detected object
            for rect in rectangles:
                results = analyze_object(color_image, depth_frame, rect, depth_scale, color_intrinsics)
                
                x1, y1, x2, y2 = rect
                width_mm = results["width_mm"]
                height_mm = results["height_mm"]
                depth_mm = results["depth_mm"]
                credit_card_match = results["credit_card_match"]
                match_percentage = results["match_percentage"]
                
                # Draw rectangle
                color = (0, 255, 0) if credit_card_match else (0, 165, 255)
                cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
                
                # Add measurements as text
                if width_mm is not None and height_mm is not None:
                    cv2.putText(
                        display_image, 
                        f"W: {width_mm:.1f}mm", 
                        (x1, y1 - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        color, 
                        2
                    )
                    cv2.putText(
                        display_image, 
                        f"H: {height_mm:.1f}mm", 
                        (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        color, 
                        2
                    )
                    
                    # Add credit card match percentage if high enough
                    if match_percentage > 50:
                        cv2.putText(
                            display_image, 
                            f"Card match: {match_percentage:.1f}%", 
                            (x1, y2 + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            color, 
                            2
                        )
            
            # Show information about current settings
            cv2.putText(
                display_image, 
                f"Min area: {min_area}, Max area: {max_area}", 
                (10, display_image.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1
            )
            
            # Show the result
            cv2.imshow('Object Measurement', display_image)
            
            # Handle keyboard input
            key = cv2.waitKey(1)
            
            # ESC to exit
            if key == 27:
                print("Exiting...")
                break
                
            # SPACE to capture
            elif key == 32:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                output_file = os.path.join(args.output, f"measurement_{timestamp}.jpg")
                cv2.imwrite(output_file, display_image)
                print(f"Saved measurement image to {output_file}")
                
                # Also save depth colormap for reference
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), 
                    cv2.COLORMAP_JET
                )
                depth_file = os.path.join(args.output, f"depth_{timestamp}.jpg")
                cv2.imwrite(depth_file, depth_colormap)
                
                # Save object details
                if rectangles:
                    with open(os.path.join(args.output, f"measurements_{timestamp}.txt"), "w") as f:
                        f.write("Object Measurements:\n")
                        for i, rect in enumerate(rectangles):
                            results = analyze_object(color_image, depth_frame, rect, depth_scale, color_intrinsics)
                            f.write(f"Object {i+1}:\n")
                            f.write(f"  Dimensions (pixels): {results['width_px']}x{results['height_px']}\n")
                            if results["width_mm"] is not None:
                                f.write(f"  Dimensions (mm): {results['width_mm']:.1f}x{results['height_mm']:.1f}\n")
                                f.write(f"  Depth (mm): {results['depth_mm']:.1f}\n")
                                if results["match_percentage"] > 50:
                                    f.write(f"  Credit card match: {results['match_percentage']:.1f}%\n")
                            f.write("\n")
            
            # 'r' to reset detection parameters
            elif key == ord('r'):
                # Prompt for new min_area
                try:
                    min_area_str = input("Enter new minimum area (current: {}): ".format(min_area))
                    if min_area_str:
                        min_area = int(min_area_str)
                    
                    max_area_str = input("Enter new maximum area (current: {}): ".format(max_area))
                    if max_area_str:
                        max_area = int(max_area_str)
                        
                    print(f"Updated detection parameters: min_area={min_area}, max_area={max_area}")
                except:
                    print("Invalid input. Using previous values.")
                
    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 