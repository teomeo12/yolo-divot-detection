# File: visualize_ir_streams.py
# Purpose: To show the raw video from the left and right infrared sensors.
# This helps you understand the "stereo" part of the camera.

import pyrealsense2 as rs
import numpy as np
import cv2 # We need OpenCV to display the images

# 1. Initialize the RealSense pipeline
pipeline = rs.pipeline()

# 2. Configure the streams we want to use
config = rs.config()
# For this script, we ask for the two INFRARED streams instead of the depth stream
# Stream index 1 is the left IR sensor
# Stream index 2 is the right IR sensor
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

print("Starting pipeline to show IR streams...")
# 3. Start the pipeline
pipeline.start(config)

try:
    while True:
        # 4. Wait for a new set of frames
        frames = pipeline.wait_for_frames()

        # 5. Get the left and right IR frames
        left_ir_frame = frames.get_infrared_frame(1)
        right_ir_frame = frames.get_infrared_frame(2)

        if not left_ir_frame or not right_ir_frame:
            continue

        # 6. Convert the IR frames to a format that OpenCV can display (numpy array)
        left_image = np.asanyarray(left_ir_frame.get_data())
        right_image = np.asanyarray(right_ir_frame.get_data())

        # You can clearly see the dot pattern from the IR projector in these images!
        # Notice the slight difference in perspective between the two images.
        # This difference is the "disparity" the camera uses.

        # 7. Display the two images in windows
        cv2.imshow('Left IR Sensor', left_image)
        cv2.imshow('Right IR Sensor', right_image)

        # 8. Check if the 'q' key is pressed to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 9. Stop the pipeline and close all windows
    print("Stopping pipeline.")
    pipeline.stop()
    cv2.destroyAllWindows()