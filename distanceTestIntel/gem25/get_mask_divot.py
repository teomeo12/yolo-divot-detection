# File: measure_pixel_distance.py
# Purpose: Segment a black divot area from green grass, then measure depth.

import pyrealsense2 as rs
import cv2
import numpy as np

# 1. Initialize the RealSense pipeline
pipeline = rs.pipeline()

# 2. Configure the streams we want to use
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print("Starting pipeline...")
profile = pipeline.start(config)

try:
    while True:
        # 3. Grab frames
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # 4. Compute center‐pixel distance
        w, h = depth_frame.get_width(), depth_frame.get_height()
        cx, cy = w // 2, h // 2
        dist_mm = depth_frame.get_distance(cx, cy) * 1000

        # 5. Get color image
        img = np.asanyarray(color_frame.get_data())

        # 6. Annotate distance
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(img,
                    f"{dist_mm:.1f} mm",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2)

        # ─── DIVOT SEGMENTATION ─────────────────────────────────────────
        # Convert to HSV and threshold for black
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        mask = cv2.inRange(hsv, lower_black, upper_black)

        # Clean mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find the largest contour (assumed divot) and its convex hull
        contours, _ = cv2.findContours(mask,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # pick the biggest
            cnt = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(cnt)
            # create a filled overlay
            overlay = img.copy()
            cv2.drawContours(overlay, [hull], -1, (0, 255, 255), cv2.FILLED)
            # blend it
            img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

        # Display both windows automatically
        cv2.imshow("RealSense Feed", img)
        cv2.imshow("Divot Mask", mask)
        # ────────────────────────────────────────────────────────────────

        # 7. Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("\nStopping pipeline.")
    cv2.destroyAllWindows()
    pipeline.stop()
