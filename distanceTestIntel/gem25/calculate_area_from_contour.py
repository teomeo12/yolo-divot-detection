# File: calculate_area_from_contour.py
# Purpose: To segment a black object, find its contour, and calculate its
# real-world surface area in cm^2 for validation.

import pyrealsense2 as rs
import cv2
import numpy as np

# --- Tweakable Parameters ---
# Calibration factor for 1D distance, based on your previous tests.
CALIBRATION_FACTOR = 0.63
# The area calibration factor is the square of the linear one.
AREA_CALIBRATION_FACTOR = CALIBRATION_FACTOR ** 2

# HSV Color Thresholds for the black rectangle
LOWER_BLACK_COLOR = np.array([0, 0, 0])
UPPER_BLACK_COLOR = np.array([180, 255, 50]) # Tune the last value (Value) for lighting

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

# Get camera intrinsics for area calculation
intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
fx, fy = intrinsics.fx, intrinsics.fy
ppx, ppy = intrinsics.ppx, intrinsics.ppy
camera_matrix = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])
dist_coeffs = np.array(intrinsics.coeffs)

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

        # --- NEW: Undistort images ---
        color_image = cv2.undistort(color_image, camera_matrix, dist_coeffs)
        # We don't undistort the depth image directly here to maintain alignment
        # provided by the RealSense SDK's align object. Calculations will be
        # done on undistorted color image, but depth is sampled from aligned frame.

        display_image = color_image.copy()

        # 3. SEGMENT THE BLACK RECTANGLE
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_BLACK_COLOR, UPPER_BLACK_COLOR)

        # Clean mask using morphological operations
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)

        # 4. FIND CONTOUR AND CALCULATE AREA
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Assume the largest contour is our rectangle
            rect_contour = max(contours, key=cv2.contourArea)

            # Create a filled mask of just the largest contour
            contour_mask = np.zeros(mask.shape, dtype="uint8")
            cv2.drawContours(contour_mask, [rect_contour], -1, 255, -1)
            
            # Get the average distance to the object to calculate pixel area
            object_depths = depth_image[contour_mask == 255]
            object_depths_mm = object_depths[object_depths != 0]

            if object_depths_mm.size > 0:
                avg_dist_mm = np.mean(object_depths_mm)
                
                # Estimate the area of one pixel at the object's distance
                pixel_area_m2 = ((avg_dist_mm / 1000) / fx) * ((avg_dist_mm / 1000) / fy)
                pixel_area_cm2 = pixel_area_m2 * 10000

                # Calculate the raw, uncalibrated area
                raw_area_cm2 = object_depths_mm.size * pixel_area_cm2
                
                # Apply the area calibration factor
                calibrated_area_cm2 = raw_area_cm2 * AREA_CALIBRATION_FACTOR
                
                # --- VISUALIZATION ---
                # Draw the detected contour
                cv2.drawContours(display_image, [rect_contour], -1, (0, 255, 0), 2) # Green

                # Display the results
                cv2.putText(display_image, f"Area: {calibrated_area_cm2:.2f} cm^2 (calibrated)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display_image, f"Distance: {avg_dist_mm:.1f} mm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


        # Display the images
        cv2.imshow("RealSense Feed", display_image)
        cv2.imshow("Object Mask", mask)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("\nStopping pipeline.")
    cv2.destroyAllWindows()
    pipeline.stop() 