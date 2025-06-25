# File: manual_distance_measurement.py
# Purpose: Manually select two points on the color stream and calculate
# the real-world 3D distance between them.

import pyrealsense2 as rs
import cv2
import numpy as np

# --- Tweakable Parameters ---
# Calibration factor to correct for systemic measurement errors.
# Calculated as: real_distance / measured_distance
CALIBRATION_FACTOR = 0.63 #big box
#CALIBRATION_FACTOR = 0.58 #small box
# --- End of Parameters ---

# List to store the two points we select
points = []

def mouse_callback(event, x, y, flags, param):
    """Mouse callback function to capture clicks"""
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append((x, y))

# 1. Initialize Pipeline and get Intrinsics
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
print("Starting pipeline...")
profile = pipeline.start(config)
align = rs.align(rs.stream.color)
intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# 2. Setup OpenCV window
window_name = "RealSense - Click 2 points, press 'r' to reset, 'q' to quit"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

print("Pipeline started. Click two points on the image.")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        display_image = np.asanyarray(color_frame.get_data())

        # If we have two points, calculate the distance
        if len(points) == 2:
            points_3d = []
            valid_points = True
            for point in points:
                u, v = point
                depth = depth_frame.get_distance(u, v)
                if depth == 0:
                    # Point is invalid, depth could not be determined
                    valid_points = False
                    break
                
                # Deproject 2D pixel to 3D point in camera's coordinate system
                point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth)
                points_3d.append(point_3d)
            
            if valid_points:
                # Calculate Euclidean distance
                distance_meters = np.linalg.norm(np.subtract(points_3d[0], points_3d[1]))
                distance_cm = distance_meters * 100

                # Apply the calibration factor
                corrected_distance_cm = distance_cm * CALIBRATION_FACTOR
                
                # Draw line between points and display distance
                cv2.line(display_image, points[0], points[1], (0, 255, 0), 2)
                mid_point = ((points[0][0] + points[1][0]) // 2, (points[0][1] + points[1][1]) // 2)
                cv2.putText(display_image, f"{corrected_distance_cm:.2f} cm (calibrated)", (mid_point[0], mid_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw circles at the clicked points
        for point in points:
            cv2.circle(display_image, point, 2, (0, 0, 255), -1)

        cv2.imshow(window_name, display_image)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('r'):
            points = [] # Reset points

finally:
    print("\nStopping pipeline.")
    cv2.destroyAllWindows()
    pipeline.stop() 