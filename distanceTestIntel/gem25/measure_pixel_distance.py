# File: measure_pixel_distance.py
# Purpose: To get the distance (depth) of the center pixel from the camera.
# This is the primary way you will interact with the camera for depth data.

import pyrealsense2 as rs
import cv2
import numpy as np

# 1. Initialize the RealSense pipeline
pipeline = rs.pipeline()

# 2. Configure the streams we want to use
config = rs.config()
# We want the depth stream, 640x480 is a common resolution
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print("Starting pipeline...")
# 3. Start the pipeline with our configuration
profile = pipeline.start(config)

try:
    # Loop to continuously get frames
    while True:
        # 4. Wait for a new set of frames from the camera
        frames = pipeline.wait_for_frames()

        # 5. Get the depth frame
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # If the frame is not valid, skip it
        if not depth_frame or not color_frame:
            continue

        # 6. Get the dimensions of the frame
        width = depth_frame.get_width()
        height = depth_frame.get_height()

        # 7. Define the pixel we want to measure. Let's use the center pixel.
        center_x = width // 2
        center_y = height // 2

        # 8. Get the distance of this pixel from the camera
        # The get_distance() function does all the work for you!
        # It returns the distance in meters.
        distance_in_meters = depth_frame.get_distance(center_x, center_y)

        # Convert distance to millimeters
        distance_in_mm = distance_in_meters * 1000

        # Convert color frame to a numpy array that OpenCV can use
        color_image = np.asanyarray(color_frame.get_data())

        # Draw a circle at the center pixel
        cv2.circle(color_image, (center_x, center_y), 3, (0, 0, 255), -1)

        # Display the distance on the image
        cv2.putText(color_image, f"Distance: {distance_in_mm:.3f} mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 9. Show the color image
        cv2.imshow("RealSense - Press 'q' to quit", color_image)
        key = cv2.waitKey(1)

        # Press 'q' to exit
        if key & 0xFF == ord('q'):
            break

finally:
    # 10. Always stop the pipeline when you're done!
    print("\nStopping pipeline.")
    cv2.destroyAllWindows()
    pipeline.stop()