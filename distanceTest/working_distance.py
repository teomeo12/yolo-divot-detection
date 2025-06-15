import numpy as np
import cv2
import pyrealsense2 as rs

def main():
    print("=== SIMPLE WORKING DISTANCE TOOL ===")
    
    # Setup camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start camera
    pipeline.start(config)
    
    print("Camera started! Press ESC to exit")
    
    try:
        while True:
            # Get frames
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # Convert to images
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Get center point
            h, w = color_image.shape[:2]
            center_x, center_y = w//2, h//2
            
            # Get distance at center
            distance_mm = depth_frame.get_distance(center_x, center_y) * 1000
            
            # CREATE DEPTH MAP for visualization
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            # Draw crosshair on both images
            cv2.line(color_image, (center_x-20, center_y), (center_x+20, center_y), (0, 255, 0), 2)
            cv2.line(color_image, (center_x, center_y-20), (center_x, center_y+20), (0, 255, 0), 2)
            cv2.line(depth_colormap, (center_x-20, center_y), (center_x+20, center_y), (255, 255, 255), 2)
            cv2.line(depth_colormap, (center_x, center_y-20), (center_x, center_y+20), (255, 255, 255), 2)
            
            # Show distance and debug info
            if distance_mm > 0:
                text = f"Distance: {distance_mm:.1f}mm ({distance_mm/10:.1f}cm)"
                cv2.putText(color_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(depth_colormap, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(color_image, "NO DEPTH DATA - Try pointing at objects!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(depth_colormap, "NO DEPTH DATA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show both images
            cv2.imshow('Color + Distance', color_image)
            cv2.imshow('Depth Map', depth_colormap)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Done!")

if __name__ == "__main__":
    main() 