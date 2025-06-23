# File: divot_depth_color_ground.py
# Purpose: Automatically segment a divot using color, then calculate its volume
# relative to a color-segmented ground plane.

import pyrealsense2 as rs
import cv2
import numpy as np

# --- Tweakable Parameters ---
# HSV Color Thresholds for the divot (adjust for your lighting/soil color)
LOWER_DIVOT_COLOR = np.array([0, 0, 0])
UPPER_DIVOT_COLOR = np.array([180, 255, 60]) # The last value (Value) is key for darkness

# HSV Color Thresholds for the ground plane (the green sheet)
LOWER_GROUND_COLOR = np.array([35, 50, 50]) # Example for green, you MUST tune this
UPPER_GROUND_COLOR = np.array([85, 255, 255])

# Morphological Kernel size for cleaning up the mask
MORPH_KERNEL = np.ones((5, 5), np.uint8)
# --- End of Parameters ---


# 1. Initialize the RealSense pipeline
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

print("Pipeline started. Press 'q' to quit.")

try:
    while True:
        # 2. Get aligned frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        display_image = color_image.copy()

        # --- 3. DIVOT AND GROUND SEGMENTATION ---
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        
        # Segment the divot (black/brown area)
        divot_mask = cv2.inRange(hsv, LOWER_DIVOT_COLOR, UPPER_DIVOT_COLOR)
        # Segment the ground (green area)
        ground_mask = cv2.inRange(hsv, LOWER_GROUND_COLOR, UPPER_GROUND_COLOR)

        # Clean up both masks
        divot_mask = cv2.morphologyEx(divot_mask, cv2.MORPH_OPEN, MORPH_KERNEL)
        divot_mask = cv2.morphologyEx(divot_mask, cv2.MORPH_CLOSE, MORPH_KERNEL)
        
        ground_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_OPEN, MORPH_KERNEL)
        ground_mask = cv2.morphologyEx(ground_mask, cv2.MORPH_CLOSE, MORPH_KERNEL)


        # --- 4. GROUND PLANE DETECTION FROM COLOR ---
        ground_depths = depth_image[ground_mask == 255]
        ground_depths_mm = ground_depths[ground_depths != 0]

        if ground_depths_mm.size == 0:
            cv2.putText(display_image, "No ground plane detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            # This is our reference plane, based on the green sheet!
            ground_level_mm = np.mean(ground_depths_mm)
            
            # --- Now process the divot relative to the ground ---
            contours, _ = cv2.findContours(divot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Assume the largest contour is our divot
                divot_contour = max(contours, key=cv2.contourArea)

                # Create a filled mask of just the largest divot contour
                divot_mask_filled = np.zeros(divot_mask.shape, dtype="uint8")
                cv2.drawContours(divot_mask_filled, [divot_contour], -1, 255, -1)

                # --- 5. VOLUME CALCULATION ---
                total_volume_cm3 = 0.0
                total_area_cm2 = 0.0
                
                # Get all pixel depths inside the divot mask
                divot_depths = depth_image[divot_mask_filled == 255]
                divot_depths_mm = divot_depths[divot_depths != 0]

                # Calculate depth difference relative to the ground
                depth_differences_mm = divot_depths_mm - ground_level_mm

                # Consider only points that are deeper than the ground
                hole_depths_mm = depth_differences_mm[depth_differences_mm > 0]
                
                if hole_depths_mm.size > 0:
                    # Estimate the area of one pixel at the ground depth
                    pixel_area_m2 = ((ground_level_mm / 1000) / fx) * ((ground_level_mm / 1000) / fy)
                    pixel_area_cm2 = pixel_area_m2 * 10000

                    # Volume = sum of all (depth * area)
                    total_volume_cm3 = np.sum(hole_depths_mm / 10) * pixel_area_cm2
                    total_area_cm2 = hole_depths_mm.size * pixel_area_cm2

                # --- 6. VISUALIZATION ---
                # Draw the divot contour
                cv2.drawContours(display_image, [divot_contour], -1, (255, 0, 0), 2) # Blue
                # Draw the detected ground for visual feedback
                display_image[ground_mask == 255] = [0, 255, 0] # Green

                # Display the results
                cv2.putText(display_image, f"Ground: {ground_level_mm:.1f} mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_image, f"Area: {total_area_cm2:.1f} cm^2", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display_image, f"Volume: {total_volume_cm3:.2f} cm^3", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Display the images
        cv2.imshow("RealSense Feed", display_image)
        cv2.imshow("Divot Mask", divot_mask)
        cv2.imshow("Ground Mask", ground_mask)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("\nStopping pipeline.")
    cv2.destroyAllWindows()
    pipeline.stop() 