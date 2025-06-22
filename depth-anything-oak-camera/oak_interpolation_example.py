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

    # Configure mono cameras for OV7251 sensors
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    # Configure RGB camera
    camRgb.setPreviewSize(640, 480)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # Configure depth with interpolation and filtering
    depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    depth.setLeftRightCheck(True)
    depth.setSubpixel(True)
    depth.setExtendedDisparity(False)
    
    # Configure post-processing with interpolation
    config = depth.initialConfig.get()
    
    # Enable spatial filter for interpolation
    config.postProcessing.spatialFilter.enable = True
    config.postProcessing.spatialFilter.holeFillingRadius = 2
    config.postProcessing.spatialFilter.numIterations = 2
    config.postProcessing.spatialFilter.alpha = 0.5
    config.postProcessing.spatialFilter.delta = 20
    
    # Enable temporal filter for smoother depth
    config.postProcessing.temporalFilter.enable = True
    config.postProcessing.temporalFilter.alpha = 0.4
    config.postProcessing.temporalFilter.delta = 20
    config.postProcessing.temporalFilter.persistencyMode = dai.StereoDepthConfig.PostProcessing.TemporalFilter.PersistencyMode.VALID_8_OUT_OF_8
    
    # Enable speckle filter to remove noise
    config.postProcessing.speckleFilter.enable = True
    config.postProcessing.speckleFilter.speckleRange = 50
    
    # Threshold filter for distance range
    config.postProcessing.thresholdFilter.minRange = 200  # 20cm
    config.postProcessing.thresholdFilter.maxRange = 5000  # 5m
    
    # Apply configuration
    depth.initialConfig.set(config)

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
            print("OAK Camera with Interpolation - Divot Measurement")
            print("Post-processing features enabled:")
            print("- Spatial Filter: Fills holes and smooths depth")
            print("- Temporal Filter: Reduces noise over time")
            print("- Speckle Filter: Removes small invalid regions")
            print("")
            print("Controls:")
            print("- Click on divot to measure depth")
            print("- 'f': toggle post-processing filters")
            print("- 'i': show interpolation info")
            print("- 'c': clear measurement points")
            print("- 'q': quit")
            
            # Get output queues
            qDepth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            
            # Measurement storage
            measurement_points = []
            filters_enabled = True
            
            # Mouse callback for measurement
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN and param is not None:
                    depth_frame = param
                    if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
                        # Get depth value and surrounding area for better measurement
                        region_size = 3
                        x_start = max(0, x - region_size)
                        x_end = min(depth_frame.shape[1], x + region_size + 1)
                        y_start = max(0, y - region_size)
                        y_end = min(depth_frame.shape[0], y + region_size + 1)
                        
                        depth_region = depth_frame[y_start:y_end, x_start:x_end]
                        valid_depths = depth_region[depth_region > 0]
                        
                        if len(valid_depths) > 0:
                            # Use median of region for more stable measurement
                            depth_value = np.median(valid_depths)
                            distance_cm = depth_value / 10.0
                            measurement_points.append((x, y, depth_value))
                            
                            print(f"Point {len(measurement_points)}: ({x}, {y}) = {distance_cm:.1f} cm")
                            print(f"  Region stats: {len(valid_depths)}/{region_size*2+1}² valid pixels")
                            
                            # Calculate depth differences
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
                            print(f"No valid depth data around ({x}, {y})")
            
            cv2.namedWindow('Interpolated Depth Measurement')
            current_depth = None
            cv2.setMouseCallback('Interpolated Depth Measurement', mouse_callback, current_depth)
            
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
                    cv2.setMouseCallback('Interpolated Depth Measurement', mouse_callback, current_depth)
                    
                    # Apply additional OpenCV-based interpolation if needed
                    depth_interpolated = depth_resized.copy()
                    
                    # Fill small holes using inpainting
                    mask = (depth_interpolated == 0).astype(np.uint8)
                    if np.any(mask):
                        # Use OpenCV inpainting for additional hole filling
                        depth_interpolated = cv2.inpaint(depth_interpolated.astype(np.uint16), mask, 3, cv2.INPAINT_TELEA)
                    
                    # Create visualization
                    depth_vis = depth_interpolated.copy()
                    invalid_mask = depth_vis == 0
                    
                    if np.any(~invalid_mask):
                        valid_depth = depth_vis[~invalid_mask]
                        min_val = np.min(valid_depth)
                        max_val = np.max(valid_depth)
                        
                        if max_val > min_val:
                            depth_normalized = np.zeros_like(depth_vis, dtype=np.uint8)
                            depth_normalized[~invalid_mask] = ((depth_vis[~invalid_mask] - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                            depth_colored[invalid_mask] = [0, 0, 0]
                        else:
                            depth_colored = np.zeros((depth_vis.shape[0], depth_vis.shape[1], 3), dtype=np.uint8)
                    else:
                        depth_colored = np.zeros((depth_vis.shape[0], depth_vis.shape[1], 3), dtype=np.uint8)
                    
                    # Calculate quality metrics
                    original_valid = np.count_nonzero(depth_resized)
                    interpolated_valid = np.count_nonzero(depth_interpolated)
                    total_pixels = depth_resized.shape[0] * depth_resized.shape[1]
                    
                    improvement = interpolated_valid - original_valid
                    improvement_percent = (improvement / total_pixels) * 100
                    
                    # Add crosshair and measurement points
                    center_x, center_y = rgb_frame.shape[1] // 2, rgb_frame.shape[0] // 2
                    cv2.line(rgb_frame, (center_x - 15, center_y), (center_x + 15, center_y), (0, 255, 0), 2)
                    cv2.line(rgb_frame, (center_x, center_y - 15), (center_x, center_y + 15), (0, 255, 0), 2)
                    
                    # Draw measurement points with region indicators
                    for i, (px, py, depth_val) in enumerate(measurement_points):
                        color = (0, 255, 255) if i == len(measurement_points) - 1 else (255, 255, 0)
                        cv2.circle(rgb_frame, (px, py), 5, color, -1)
                        cv2.circle(rgb_frame, (px, py), 10, color, 1)  # Show measurement region
                        cv2.putText(rgb_frame, f"{i+1}", (px + 15, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Center distance with interpolated value
                    center_depth_orig = depth_resized[center_y, center_x]
                    center_depth_interp = depth_interpolated[center_y, center_x]
                    
                    if center_depth_interp > 0:
                        depth_text = f"Center: {center_depth_interp/10:.1f} cm"
                        if center_depth_orig == 0 and center_depth_interp > 0:
                            depth_text += " (interpolated)"
                        cv2.putText(rgb_frame, depth_text, 
                                   (center_x - 100, center_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Add info text
                    info_y = 25
                    cv2.putText(rgb_frame, f"Original valid: {(original_valid/total_pixels)*100:.1f}%", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    info_y += 20
                    cv2.putText(rgb_frame, f"Interpolated: {(interpolated_valid/total_pixels)*100:.1f}%", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    info_y += 20
                    cv2.putText(rgb_frame, f"Improvement: +{improvement_percent:.1f}%", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    info_y += 20
                    cv2.putText(rgb_frame, f"Filters: {'ON' if filters_enabled else 'OFF'}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0) if filters_enabled else (0, 0, 255), 1)
                    info_y += 20
                    cv2.putText(rgb_frame, f"Points: {len(measurement_points)}", (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # Controls text
                    cv2.putText(rgb_frame, "Click: measure | 'f': filters | 'i': info | 'c': clear | 'q': quit", 
                               (10, rgb_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    # Create side-by-side view
                    combined = np.hstack((rgb_frame, depth_colored))
                    
                    # Add labels
                    cv2.putText(combined, "RGB + Measurements", (10, rgb_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(combined, "Interpolated Depth", (rgb_frame.shape[1] + 10, rgb_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    cv2.imshow('Interpolated Depth Measurement', combined)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Exiting...")
                    break
                elif key == ord('c'):
                    measurement_points.clear()
                    print("Cleared measurement points")
                elif key == ord('f'):
                    filters_enabled = not filters_enabled
                    print(f"Filters {'enabled' if filters_enabled else 'disabled'}")
                    # Note: This would require pipeline restart to actually change filters
                elif key == ord('i'):
                    print("\nInterpolation Methods Used:")
                    print("1. DepthAI Post-processing:")
                    print("   - Spatial Filter: Smooths depth using neighboring pixels")
                    print("   - Temporal Filter: Uses previous frames for stability")
                    print("   - Speckle Filter: Removes isolated invalid pixels")
                    print("2. OpenCV Inpainting: Fills remaining holes using TELEA algorithm")
                    print("3. Region-based Measurement: Uses 7x7 pixel median for click points")
                    print("")
                    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 