# File: divot_area_calculator.py
# Purpose: Automatically segment a divot using color, then calculate its projected area
# relative to the surrounding ground.

import pyrealsense2 as rs
import cv2
import numpy as np

# --- Tweakable Parameters ---
# Calibration factor for 1D distance, based on your previous tests.
CALIBRATION_FACTOR = 0.52 # for a circle area
#CALIBRATION_FACTOR = 0.67 # for a square area
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

# Get camera intrinsics for area calculation
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

        # --- 3. DIVOT SEGMENTATION ---
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        # Choose the color range to use for segmentation
        mask = cv2.inRange(hsv, LOWER_BLACK_COLOR, UPPER_BLACK_COLOR)
        # mask = cv2.inRange(hsv, LOWER_BROWN_COLOR, UPPER_BROWN_COLOR)

        # Clean mask using morphological operations
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Assume the largest contour is our divot
            divot_contour = max(contours, key=cv2.contourArea)
            
            # --- 4. AREA CALCULATION ---
            # Create a mask for just the divot to get its depths
            divot_mask = np.zeros(mask.shape, dtype="uint8")
            cv2.drawContours(divot_mask, [divot_contour], -1, 255, -1)

            # To calculate the real-world area, we need a reference distance.
            # We'll use a "ground ring" around the divot for this.
            ring_kernel = np.ones((GROUND_RING_WIDTH, GROUND_RING_WIDTH), np.uint8)
            dilated_mask = cv2.dilate(divot_mask, ring_kernel, iterations=1)
            ground_ring_mask = cv2.bitwise_and(dilated_mask, cv2.bitwise_not(divot_mask))
            
            # Get depth values for the ground ring
            ground_depths = depth_image[ground_ring_mask == 255]
            ground_depths_mm = ground_depths[ground_depths != 0] # Filter out invalid 0 values

            if ground_depths_mm.size == 0:
                cv2.putText(display_image, "No ground plane detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                # Use the average ground depth as our reference distance
                ground_level_mm = np.mean(ground_depths_mm)
                
                # Get the area of the contour in pixels
                contour_pixel_area = cv2.contourArea(divot_contour)
                
                # Estimate the area of a single pixel at the ground depth
                # This is an approximation but good enough for this purpose
                pixel_area_m2 = ((ground_level_mm / 1000) / fx) * ((ground_level_mm / 1000) / fy)
                pixel_area_cm2 = pixel_area_m2 * 10000

                # Calculate the total area in real-world units
                total_area_cm2 = (contour_pixel_area * pixel_area_cm2) * AREA_CALIBRATION_FACTOR

                # --- 5. VISUALIZATION ---
                # Draw the divot contour
                cv2.drawContours(display_image, [divot_contour], -1, (255, 0, 0), 2) # Blue
                # Draw the ground ring for visual feedback
                display_image[ground_ring_mask == 255] = [0, 255, 0] # Green

                # Display the results
                cv2.putText(display_image, f"Ground Depth: {ground_level_mm:.1f} mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_image, f"Divot Area: {total_area_cm2:.1f} cm^2 (calibrated)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

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