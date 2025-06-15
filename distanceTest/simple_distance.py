import numpy as np
import cv2
import pyrealsense2 as rs

def setup_camera():
    """Setup RealSense camera - THIS IS WHERE SENSORS ARE ACTIVATED"""
    pipeline = rs.pipeline()
    config = rs.config()
    
    # ACTIVATE DIFFERENT SENSORS HERE:
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Stereo depth sensor
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB camera
    # config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)  # Left IR camera
    # config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)  # Right IR camera
    
    # Start the camera
    profile = pipeline.start(config)
    
    # Get depth sensor properties
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    # Manually set key options for high accuracy
    depth_sensor.set_option(rs.option.exposure, 200)  # Adjust exposure
    depth_sensor.set_option(rs.option.gain, 16)       # Adjust gain
    depth_sensor.set_option(rs.option.laser_power, 150)  # Adjust laser power
    
    print(f"Camera started! Depth scale: {depth_scale}")
    return pipeline, depth_scale

def measure_distance_at_point(depth_frame, x, y):
    """THIS IS THE CORE DISTANCE MEASUREMENT FUNCTION"""
    # Ensure the frame is a valid depth frame
    if not isinstance(depth_frame, rs.depth_frame):
        print("Error: Not a valid depth frame")
        return None

    # Get depth value at pixel (x, y) in meters
    distance_meters = depth_frame.get_distance(x, y)
    
    # Convert to millimeters
    distance_mm = distance_meters * 1000
    
    print(f"Distance at pixel ({x}, {y}): {distance_mm:.1f}mm")
    return distance_mm

# Create filters globally (before your loop)
decimation = rs.decimation_filter()
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
hole_filling = rs.hole_filling_filter()

def main():
    """Simple main function to test distance measurement"""
    print("=== SIMPLE DISTANCE MEASUREMENT ===")
    print("1. Setting up camera...")
    
    try:
        pipeline, depth_scale = setup_camera()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure RealSense camera is connected!")
        return
    
    print("2. Camera ready! Real-time distance measurement at center crosshair")
    print("Point the center crosshair at different objects to test accuracy")
    print("Press ESC to exit")
    
    # Create two windows - one for RGB+distance, one for depth map
    cv2.namedWindow('Distance Measurement', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Depth Map', cv2.WINDOW_AUTOSIZE)
    
    # Position windows side by side
    cv2.moveWindow('Distance Measurement', 50, 50)
    cv2.moveWindow('Depth Map', 700, 50)
    
    try:
        while True:
            # Get frames from camera
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # Ensure the frame is a valid depth frame
            if not depth_frame.is_depth_frame():
                print("Error: Not a valid depth frame")
                continue
            
            # Apply post-processing filters
            depth_frame = decimation.process(depth_frame)
            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)
            depth_frame = hole_filling.process(depth_frame)
            
            # Convert filtered depth to numpy
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Convert to numpy arrays for display
            color_image = np.asanyarray(color_frame.get_data())
            h, w = color_image.shape[:2]
            center_x, center_y = w//2, h//2
            
            # REAL-TIME DISTANCE MEASUREMENT AT CENTER
            center_distance = measure_distance_at_point(depth_frame, center_x, center_y)
            
            # CREATE COLORIZED DEPTH MAP - OPTIMIZED FOR GOLF DIVOT MEASUREMENT
            # Normalize depth image for close-range visualization (0-50cm range)
            depth_meters = depth_image * depth_scale
            depth_normalized = np.clip(depth_meters, 0, 0.5) / 0.5  # 0-50cm range
            depth_normalized = (depth_normalized * 255).astype(np.uint8)
            
            # Apply colormap for better divot visualization (INFERNO gives better contrast for small depths)
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
            
            # Add crosshair to depth map as well
            cv2.line(depth_colormap, (center_x-20, center_y), (center_x+20, center_y), (255, 255, 255), 2)
            cv2.line(depth_colormap, (center_x, center_y-20), (center_x, center_y+20), (255, 255, 255), 2)
            cv2.circle(depth_colormap, (center_x, center_y), 5, (255, 255, 255), 2)
            
            # Add distance info to depth map with higher precision
            if center_distance > 0:
                cv2.putText(depth_colormap, f"{center_distance:.1f}mm", (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(depth_colormap, f"{center_distance/10:.1f}cm", (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Add color scale legend optimized for divot measurement
            cv2.putText(depth_colormap, "DIVOT DEPTH SCALE:", (10, h-100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(depth_colormap, "Dark Blue=Closest (divot bottom)", (10, h-80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(depth_colormap, "Yellow/Red=Surface level", (10, h-60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(depth_colormap, "Range: 0-50cm (divot precision)", (10, h-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Add crosshair at center with BIGGER, MORE VISIBLE design
            cv2.line(color_image, (center_x-20, center_y), (center_x+20, center_y), (0, 255, 0), 2)
            cv2.line(color_image, (center_x, center_y-20), (center_x, center_y+20), (0, 255, 0), 2)
            cv2.circle(color_image, (center_x, center_y), 5, (0, 255, 0), 2)
            
            # REAL-TIME DISTANCE DISPLAY - Big, visible text
            if center_distance > 0:
                distance_text = f"Distance: {center_distance:.1f}mm ({center_distance/10:.1f}cm)"
                cv2.putText(color_image, distance_text, (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                cv2.putText(color_image, distance_text, (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)  # Black outline
            else:
                cv2.putText(color_image, "No depth data at center", (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Instructions
            cv2.putText(color_image, "Point center crosshair at objects to test accuracy", (10, h-40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(color_image, "ESC to exit", (10, h-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show both windows
            cv2.imshow('Distance Measurement', color_image)
            cv2.imshow('Depth Map', depth_colormap)
            
            # Check for ESC key in both windows
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera stopped")

if __name__ == "__main__":
    main() 