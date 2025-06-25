# File: automatic_volume_calculator.py
# Purpose: Automatically segment a divot using color, then calculate its volume
# relative to the surrounding ground.

import pyrealsense2 as rs
import cv2
import numpy as np

# --- Tweakable Parameters ---
# Calibration factor for 1D distance, based on your previous tests.
CALIBRATION_FACTOR = 0.67
# The area calibration factor is the square of the linear one.
AREA_CALIBRATION_FACTOR = CALIBRATION_FACTOR ** 2

# HSV Color Thresholds for the divot (adjust for your lighting/soil color)
# For black/dark brown soil. You can use a color picker to find these.
LOWER_BLACK_COLOR = np.array([0, 0, 0])
UPPER_BLACK_COLOR = np.array([180, 255, 60]) # The last value (Value) is key for darkness

# Add a range for brown, converted from your RGB values
LOWER_BROWN_COLOR = np.array([10, 40, 20])
UPPER_BROWN_COLOR = np.array([25, 255, 200])

# Morphological Kernel size for cleaning up the mask
MORPH_KERNEL = np.ones((5, 5), np.uint8)

# How many pixels to expand the mask to find the surrounding "ground"
GROUND_RING_WIDTH = 15
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

        # --- 3. DIVOT SEGMENTATION (Your code, slightly adapted) ---
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        #mask = cv2.inRange(hsv, LOWER_BROWN_COLOR, UPPER_BROWN_COLOR)
        mask = cv2.inRange(hsv, LOWER_BLACK_COLOR, UPPER_BLACK_COLOR)

        # Clean mask using morphological operations
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Assume the largest contour is our divot
            divot_contour = max(contours, key=cv2.contourArea)

            # --- 4. AUTOMATIC GROUND PLANE DETECTION ---
            # Create a mask for just the divot
            divot_mask = np.zeros(mask.shape, dtype="uint8")
            cv2.drawContours(divot_mask, [divot_contour], -1, 255, -1)

            # Dilate the divot mask to get a "ring" around it
            ring_kernel = np.ones((GROUND_RING_WIDTH, GROUND_RING_WIDTH), np.uint8)
            dilated_mask = cv2.dilate(divot_mask, ring_kernel, iterations=1)
            # The ground ring is the dilated area minus the divot area
            ground_ring_mask = cv2.bitwise_and(dilated_mask, cv2.bitwise_not(divot_mask))
            
            # Get depth values for the ground ring
            ground_depths = depth_image[ground_ring_mask == 255]
            ground_depths_mm = ground_depths[ground_depths != 0] # Filter out invalid 0 values

            if ground_depths_mm.size == 0:
                cv2.putText(display_image, "No ground plane detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                # This is our reference plane!
                ground_level_mm = np.mean(ground_depths_mm)

                # --- 5. VOLUME CALCULATION AND SINGLE-POINT DEPTH CHECK ---
                total_volume_cm3 = 0.0
                total_area_cm2 = 0.0
                divot_point_depth_mm = 0

                # --- New: Get a single point depth reference from inside the divot ---
                # Calculate the centroid of the contour to get a sample point
                M = cv2.moments(divot_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Get the depth at the centroid
                    centroid_depth_mm = depth_image[cy, cx]
                    if centroid_depth_mm > 0:
                        # Calculate depth relative to the ground plane
                        divot_point_depth_mm = centroid_depth_mm - ground_level_mm
                else:
                    cx, cy = 0, 0
                
                # Get all pixel depths inside the divot mask
                divot_depths = depth_image[divot_mask == 255]
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

                # --- 6. VISUALIZATION ---
                # Draw the divot contour
                cv2.drawContours(display_image, [divot_contour], -1, (255, 0, 0), 2) # Blue
                # Draw the ground ring for visual feedback
                display_image[ground_ring_mask == 255] = [0, 255, 0] # Green
                # Draw a red dot at the centroid for the point depth measurement
                if 'cx' in locals() and cx > 0:
                    cv2.circle(display_image, (cx, cy), 5, (0, 0, 255), -1)

                # Display the results
                cv2.putText(display_image, f"Ground: {ground_level_mm:.1f} mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_image, f"Area: {total_area_cm2:.1f} cm^2 (calibrated)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(display_image, f"Volume: {total_volume_cm3:.2f} cm^3 (calibrated)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                if divot_point_depth_mm > 0:
                    cv2.putText(display_image, f"Point Depth: {divot_point_depth_mm:.1f} mm", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Display the images
        cv2.imshow("RealSense Feed", display_image)
        cv2.imshow("Divot Mask", mask)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("\nStopping pipeline.")
    cv2.destroyAllWindows()
    pipeline.stop()