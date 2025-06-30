# File: real_world_dimensions.py
# Purpose: To calculate and display the real-world dimensions (in cm)
# of the camera's view based on the distance to the center object.

import pyrealsense2 as rs
import cv2
import numpy as np
import math

# --- Parameters ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
# --- End of Parameters ---

# 1. Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# 2. Configure streams for depth and color
config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, 30)
config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, 30)

print("Starting pipeline...")
profile = pipeline.start(config)

# Create an align object to map depth to color
align = rs.align(rs.stream.color)

# --- 3. Get Camera's Field of View (FoV) ---
# This is a fixed property of the camera lens
intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
# The rs.rs2_fov function calculates the FoV angles in degrees
fov = rs.rs2_fov(intrinsics)
horizontal_fov = fov[0]
vertical_fov = fov[1]

print("-" * 30)
print("Camera Properties:")
print(f"  Horizontal FoV: {horizontal_fov:.2f} degrees")
print(f"  Vertical FoV:   {vertical_fov:.2f} degrees")
print("-" * 30)
print("Point camera at a surface. Press 'q' to quit.")


try:
    while True:
        # 4. Get aligned frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        display_image = np.asanyarray(color_frame.get_data())
        height, width, _ = display_image.shape
        center_x, center_y = width // 2, height // 2

        # 5. Get the distance to the center pixel
        distance_m = depth_frame.get_distance(center_x, center_y)

        # --- 6. Calculate Real-World Dimensions ---
        view_width_cm = 0
        view_height_cm = 0

        # We can only calculate if we have a valid distance reading
        if distance_m > 0:
            # Formula: dimension = 2 * distance * tan(FoV_angle / 2)
            # We convert FoV from degrees to radians for the math.tan function
            view_width_cm = 2 * (distance_m * 100) * math.tan(math.radians(horizontal_fov / 2))
            view_height_cm = 2 * (distance_m * 100) * math.tan(math.radians(vertical_fov / 2))


        # --- 7. Display the Information ---
        # Draw the center crosshair
        cv2.line(display_image, (center_x, 0), (center_x, height), (0, 0, 255), 2)
        cv2.line(display_image, (0, center_y), (width, center_y), (0, 0, 255), 2)

        # Create a semi-transparent background for the text
        text_bg = np.zeros_like(display_image, np.uint8)
        cv2.rectangle(text_bg, (0, 0), (450, 100), (0, 0, 0), -1)
        display_image = cv2.addWeighted(display_image, 1.0, text_bg, 0.5, 0)
        
        # Display the text
        cv2.putText(display_image, f"Distance to Center: {distance_m * 100:.1f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_image, f"Frame Width: {view_width_cm:.1f} cm", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_image, f"Frame Height: {view_height_cm:.1f} cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Real-World Dimensions", display_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("\nStopping pipeline.")
    cv2.destroyAllWindows()
    pipeline.stop()