# File: interactive_area_calculator.py
# Purpose: Allow a user to select points on screen to define a boundary
# and calculate the calibrated top surface area of that region.

import pyrealsense2 as rs
import cv2
import numpy as np

# --- Tweakable Parameters ---
# Calibration factor for 1D distance, based on your previous tests.
CALIBRATION_FACTOR = 0.63 #big box
#CALIBRATION_FACTOR = 0.58 #small box   
# The area calibration factor is the square of the linear one.
AREA_CALIBRATION_FACTOR = CALIBRATION_FACTOR ** 2
# --- End of Parameters ---

# Global list to store the points the user clicks
selected_points = []

def mouse_callback(event, x, y, flags, param):
    """Appends the (x, y) coordinates to a global list on left-click."""
    global selected_points
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append((x, y))

# 1. Initialize everything
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
print("Starting pipeline...")
profile = pipeline.start(config)
align = rs.align(rs.stream.color)
intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
fx, fy = intrinsics.fx, intrinsics.fy

# Setup OpenCV window and mouse callback
window_name = "RealSense - Click to define area, 'r' to reset, 'q' to quit"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

print("--- Instructions ---")
print("Left-click around the edge of an object to define its boundary.")
print("Press 'r' to clear the selected points and start over.")
print("Press 'q' to quit the application.")
print("--------------------")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        display_image = color_image.copy()

        # Draw the points the user has selected
        for point in selected_points:
            cv2.circle(display_image, point, 5, (0, 0, 255), -1)

        # If we have enough points (at least 3 to form a polygon)
        if len(selected_points) >= 3:
            # --- 1. Create a mask from the user-defined polygon ---
            hull_points = np.array(selected_points)
            convex_hull = cv2.convexHull(hull_points)
            cv2.drawContours(display_image, [convex_hull], 0, (0, 255, 0), 2)

            mask = np.zeros(color_image.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, convex_hull, 255)

            # --- 2. Calculate the area from the mask ---
            # Get all depth pixels inside the user-defined area
            object_depths = depth_image[mask == 255]
            object_depths_mm = object_depths[object_depths != 0]

            if object_depths_mm.size == 0:
                cv2.putText(display_image, "Cannot get depth of selected area", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                # Get the average distance to the object
                avg_dist_mm = np.mean(object_depths_mm)

                # Estimate the area of one pixel at the object's distance
                pixel_area_m2 = ((avg_dist_mm / 1000) / fx) * ((avg_dist_mm / 1000) / fy)
                pixel_area_cm2 = pixel_area_m2 * 10000

                # Calculate the raw, uncalibrated area
                raw_area_cm2 = object_depths_mm.size * pixel_area_cm2
                
                # Apply the area calibration factor
                calibrated_area_cm2 = raw_area_cm2 * AREA_CALIBRATION_FACTOR

                # Display the results
                cv2.putText(display_image, f"Distance: {avg_dist_mm:.1f} mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_image, f"Area: {calibrated_area_cm2:.2f} cm^2 (calibrated)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show the final image
        cv2.imshow(window_name, display_image)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('r'):
            selected_points.clear()
            print("Points cleared.")

finally:
    print("\nStopping pipeline.")
    cv2.destroyAllWindows()
    pipeline.stop() 