import numpy as np
import cv2
import pyrealsense2 as rs

def setup_camera():
    """Setup RealSense camera"""
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Activate sensors
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Stereo depth sensor
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB camera
    
    # Start the camera
    profile = pipeline.start(config)
    
    # Get depth sensor properties
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    
    print(f"Camera started! Depth scale: {depth_scale}")
    return pipeline, depth_scale

def measure_distance_at_point(depth_frame, x, y):
    """Measure distance at specific pixel"""
    distance_meters = depth_frame.get_distance(x, y)
    
    if distance_meters == 0:
        # Try averaging surrounding pixels if exact pixel has no depth
        distances = []
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if 0 <= x + dx < depth_frame.width and 0 <= y + dy < depth_frame.height:
                    d = depth_frame.get_distance(x + dx, y + dy)
                    if d > 0:
                        distances.append(d)
        distance_meters = np.mean(distances) if distances else 0
    
    distance_mm = distance_meters * 1000
    print(f"Distance at pixel ({x}, {y}): {distance_mm:.1f}mm")
    return distance_mm

def main():
    """Distance measurement tool with movable measurement point"""
    print("=== FLEXIBLE DISTANCE MEASUREMENT ===")
    print("1. Setting up camera...")
    
    try:
        pipeline, depth_scale = setup_camera()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure RealSense camera is connected!")
        return
    
    print("2. Camera ready!")
    print("\nControls:")
    print("- Arrow keys: Move measurement point")
    print("- SPACE: Reset to center")
    print("- 'c': Print current coordinates")
    print("- ESC: Exit")
    
    cv2.namedWindow('Flexible Distance Measurement', cv2.WINDOW_AUTOSIZE)
    
    # Initial measurement point (center)
    measure_x = 320  # Center x
    measure_y = 240  # Center y
    
    try:
        while True:
            # Get frames from camera
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # Convert to numpy arrays for display
            color_image = np.asanyarray(color_frame.get_data())
            h, w = color_image.shape[:2]
            
            # Ensure measurement point is within bounds
            measure_x = max(10, min(w-10, measure_x))
            measure_y = max(10, min(h-10, measure_y))
            
            # MEASURE DISTANCE AT CURRENT POINT
            distance = measure_distance_at_point(depth_frame, measure_x, measure_y)
            
            # Draw measurement point - BIG and VISIBLE
            cv2.circle(color_image, (measure_x, measure_y), 10, (0, 255, 0), 3)  # Green circle
            cv2.circle(color_image, (measure_x, measure_y), 3, (0, 0, 255), -1)   # Red center dot
            
            # Draw crosshair at measurement point
            cv2.line(color_image, (measure_x-15, measure_y), (measure_x+15, measure_y), (0, 255, 0), 2)
            cv2.line(color_image, (measure_x, measure_y-15), (measure_x, measure_y+15), (0, 255, 0), 2)
            
            # DISTANCE DISPLAY - Big, visible text
            if distance > 0:
                distance_text = f"Distance: {distance:.1f}mm ({distance/10:.1f}cm)"
                cv2.putText(color_image, distance_text, (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                cv2.putText(color_image, distance_text, (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 1)  # Black outline
            else:
                cv2.putText(color_image, "No depth data at this point", (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Show coordinates
            coord_text = f"Position: ({measure_x}, {measure_y})"
            cv2.putText(color_image, coord_text, (10, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Instructions
            cv2.putText(color_image, "Arrow keys: Move | SPACE: Center | ESC: Exit", (10, h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Flexible Distance Measurement', color_image)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == 82 or key == 0:  # Up arrow
                measure_y -= 10
                print(f"Moved up to ({measure_x}, {measure_y})")
            elif key == 84 or key == 1:  # Down arrow
                measure_y += 10
                print(f"Moved down to ({measure_x}, {measure_y})")
            elif key == 81 or key == 2:  # Left arrow
                measure_x -= 10
                print(f"Moved left to ({measure_x}, {measure_y})")
            elif key == 83 or key == 3:  # Right arrow
                measure_x += 10
                print(f"Moved right to ({measure_x}, {measure_y})")
            elif key == 32:  # SPACE - Reset to center
                measure_x = w // 2
                measure_y = h // 2
                print(f"Reset to center ({measure_x}, {measure_y})")
            elif key == ord('c'):  # Print coordinates
                print(f"Current position: ({measure_x}, {measure_y}), Distance: {distance:.1f}mm")
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Camera stopped")

if __name__ == "__main__":
    main() 