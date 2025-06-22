import depthai as dai
import cv2
import numpy as np

def main():
    # Create a pipeline
    pipeline = dai.Pipeline()

    # Create mono cameras for stereo depth
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    depth = pipeline.create(dai.node.StereoDepth)
    
    # Create RGB camera
    camRgb = pipeline.create(dai.node.ColorCamera)

    # Configure mono cameras for OV7251 sensors (only supports 480P)
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    # Configure RGB camera
    camRgb.setPreviewSize(640, 480)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # Simple depth configuration
    depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    depth.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
    depth.setLeftRightCheck(False)
    depth.setSubpixel(False)
    depth.setExtendedDisparity(False)

    # Link cameras to depth
    monoLeft.out.link(depth.left)
    monoRight.out.link(depth.right)

    # Create outputs
    depthOut = pipeline.create(dai.node.XLinkOut)
    depthOut.setStreamName("depth")
    depth.depth.link(depthOut.input)

    rgbOut = pipeline.create(dai.node.XLinkOut)
    rgbOut.setStreamName("rgb")
    camRgb.preview.link(rgbOut.input)

    # Connect to device and start pipeline
    try:
        with dai.Device(pipeline) as device:
            print("OAK Camera connected for divot measurement!")
            print("- Click on divot edges to measure depth")
            print("- Press 'q' to exit, 'c' to clear points")
            
            # Get camera info
            calibData = device.readCalibration()
            baseline = calibData.getBaselineDistance(dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C)
            print(f"Camera baseline: {baseline:.2f} mm")
            
            # Get output queues
            qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            
            # Store measurement points
            measurement_points = []
            
            # Mouse callback for measurement
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN and param is not None:
                    depth_frame = param
                    if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
                        depth_value = depth_frame[y, x]
                        
                        if depth_value > 0:
                            distance_cm = depth_value / 10.0
                            measurement_points.append((x, y, depth_value))
                            print(f"Point {len(measurement_points)}: ({x}, {y}) = {distance_cm:.1f} cm")
                            
                            # Calculate depth difference
                            if len(measurement_points) >= 2:
                                depths = [point[2] for point in measurement_points]
                                min_depth = min(depths)
                                max_depth = max(depths)
                                divot_depth = (max_depth - min_depth) / 10.0
                                print(f"  -> Divot depth: {divot_depth:.1f} cm")
                                
                                # Keep only last 5 points
                                if len(measurement_points) > 5:
                                    measurement_points.pop(0)
                        else:
                            print(f"No depth data at ({x}, {y})")
            
            cv2.namedWindow('Divot Measurement')
            current_depth = None
            cv2.setMouseCallback('Divot Measurement', mouse_callback, current_depth)
            
            while True:
                # Get frames
                inDepth = qDepth.get()
                inRgb = qRgb.get()
                
                if inDepth is not None and inRgb is not None:
                    # Get frames
                    depth_frame = inDepth.getFrame()
                    rgb_frame = inRgb.getCvFrame()
                    
                    # Resize depth to match RGB
                    depth_resized = cv2.resize(depth_frame, (rgb_frame.shape[1], rgb_frame.shape[0]))
                    current_depth = depth_resized
                    cv2.setMouseCallback('Divot Measurement', mouse_callback, current_depth)
                    
                    # Simple depth visualization
                    # Create a copy and handle invalid values
                    depth_vis = depth_resized.copy()
                    
                    # Set invalid depth to a specific value for normalization
                    invalid_mask = depth_vis == 0
                    if np.any(~invalid_mask):  # If we have valid depth data
                        # Normalize only valid depth values
                        valid_depth = depth_vis[~invalid_mask]
                        min_val = np.min(valid_depth)
                        max_val = np.max(valid_depth)
                        
                        if max_val > min_val:
                            # Normalize to 0-255 range
                            depth_normalized = np.zeros_like(depth_vis, dtype=np.uint8)
                            depth_normalized[~invalid_mask] = ((depth_vis[~invalid_mask] - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                            
                            # Apply colormap
                            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                            
                            # Make invalid areas black
                            depth_colored[invalid_mask] = [0, 0, 0]
                        else:
                            depth_colored = np.zeros((depth_resized.shape[0], depth_resized.shape[1], 3), dtype=np.uint8)
                    else:
                        depth_colored = np.zeros((depth_resized.shape[0], depth_resized.shape[1], 3), dtype=np.uint8)
                    
                    # Check data quality
                    valid_pixels = np.count_nonzero(depth_resized)
                    total_pixels = depth_resized.shape[0] * depth_resized.shape[1]
                    valid_percentage = (valid_pixels / total_pixels) * 100
                    
                    # Add crosshair at center
                    center_x, center_y = rgb_frame.shape[1] // 2, rgb_frame.shape[0] // 2
                    cv2.line(rgb_frame, (center_x - 15, center_y), (center_x + 15, center_y), (0, 255, 0), 2)
                    cv2.line(rgb_frame, (center_x, center_y - 15), (center_x, center_y + 15), (0, 255, 0), 2)
                    
                    # Draw measurement points
                    for i, (px, py, depth_val) in enumerate(measurement_points):
                        color = (0, 255, 255) if i == len(measurement_points) - 1 else (255, 255, 0)
                        cv2.circle(rgb_frame, (px, py), 5, color, -1)
                        cv2.putText(rgb_frame, f"{i+1}", (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Center distance
                    center_depth = depth_resized[center_y, center_x]
                    if center_depth > 0:
                        cv2.putText(rgb_frame, f"Center: {center_depth/10:.1f} cm", 
                                   (center_x - 80, center_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Add info text
                    cv2.putText(rgb_frame, f"Valid depth: {valid_percentage:.1f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(rgb_frame, f"Points: {len(measurement_points)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(rgb_frame, "Click to measure | 'c': clear | 'q': quit", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Create side-by-side view
                    combined = np.hstack((rgb_frame, depth_colored))
                    
                    # Add labels
                    cv2.putText(combined, "RGB", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(combined, "Depth", (rgb_frame.shape[1] + 10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    cv2.imshow('Divot Measurement', combined)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Exiting...")
                    break
                elif key == ord('c'):
                    measurement_points.clear()
                    print("Cleared measurement points")
                    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 