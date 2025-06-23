# File: interactive_volume_calculator.py
# Purpose: Allow a user to select points on screen to define a divot's boundary,
# then calculate the volume of that divot.

import pyrealsense2 as rs
import cv2
import numpy as np

# --- Global list to store the points the user clicks ---
selected_points = []

def mouse_callback(event, x, y, flags, param):
    """
    OpenCV mouse callback function.
    Appends the (x, y) coordinates to a global list on left-click.
    """
    global selected_points
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append((x, y))
        print(f"Point added: ({x}, {y})")

# 1. Initialize everything
pipeline = rs.pipeline()
config = rs.config()

# Enable streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

print("Starting pipeline...")
profile = pipeline.start(config)

# Create an align object.
# This aligns the depth frame to the color frame, so (x,y) in color is the
# same physical point as (x,y) in depth.
align_to = rs.stream.color
align = rs.align(align_to)

# Get the camera's intrinsic properties. This is crucial for volume calculation.
# Intrinsics tell us about the camera's lens (focal length, principal point).
intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
fx = intrinsics.fx  # Focal length in x
fy = intrinsics.fy  # Focal length in y

# Setup OpenCV window and mouse callback
cv2.namedWindow("RealSense - Click to define divot, 'c' to clear, 'q' to quit")
cv2.setMouseCallback("RealSense - Click to define divot, 'c' to clear, 'q' to quit", mouse_callback)

print("--- Instructions ---")
print("Left-click around the edge of the divot to select points.")
print("Press 'c' to clear the selected points and start over.")
print("Press 'q' to quit the application.")
print("--------------------")

try:
    while True:
        # Get aligned frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        display_image = color_image.copy()

        # Draw the points the user has selected
        for point in selected_points:
            cv2.circle(display_image, point, 5, (0, 0, 255), -1)

        # If we have enough points (at least 3 to form a polygon)
        if len(selected_points) >= 3:
            # --- 1. Define the reference ground plane ---
            ground_depths = []
            for point in selected_points:
                # Get depth of each point that defines the boundary
                depth = depth_frame.get_distance(point[0], point[1]) * 1000 # in mm
                if depth > 0: # a depth of 0 means the camera couldn't see it
                    ground_depths.append(depth)

            if not ground_depths: # If all points had invalid depth
                cv2.putText(display_image, "Cannot measure depth at boundary points", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                ground_level_mm = np.mean(ground_depths)

                # --- 2. Create a mask of the area inside the points ---
                hull_points = np.array(selected_points)
                convex_hull = cv2.convexHull(hull_points)
                cv2.drawContours(display_image, [convex_hull], 0, (0, 255, 0), 2) # Draw the boundary

                mask = np.zeros(color_image.shape[:2], dtype=np.uint8)
                cv2.fillConvexPoly(mask, convex_hull, 255)

                # --- 3. Calculate the volume ---
                total_volume_cm3 = 0.0
                
                # Get the coordinates of all pixels inside the mask
                divot_pixel_coords = np.where(mask == 255)
                
                for y, x in zip(*divot_pixel_coords):
                    # Get the depth of this pixel in the divot
                    pixel_depth_mm = depth_frame.get_distance(x, y) * 1000
                    
                    if pixel_depth_mm == 0: continue # Skip invalid pixels

                    # Calculate the depth difference from the ground plane
                    depth_diff_mm = pixel_depth_mm - ground_level_mm

                    # We only care about pixels that are DEEPER than the ground
                    if depth_diff_mm > 0:
                        # Calculate the real-world area of this single pixel
                        # Area = (distance/focal_length)^2
                        pixel_area_m2 = ((pixel_depth_mm / 1000) / fx) * ((pixel_depth_mm / 1000) / fy)
                        
                        # Calculate the volume of this tiny pixel "column"
                        pixel_volume_m3 = pixel_area_m2 * (depth_diff_mm / 1000)
                        
                        # Add to total, converting from m^3 to cm^3 (1 m^3 = 1,000,000 cm^3)
                        total_volume_cm3 += pixel_volume_m3 * 1e6

                # Display the results
                cv2.putText(display_image, f"Ground Level: {ground_level_mm:.1f} mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display_image, f"Divot Volume: {total_volume_cm3:.2f} cm^3", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


        # Show the final image
        cv2.imshow("RealSense - Click to define divot, 'c' to clear, 'q' to quit", display_image)
        key = cv2.waitKey(1)

        # Handle key presses
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('c'):
            selected_points.clear()
            print("Points cleared.")

finally:
    print("\nStopping pipeline.")
    pipeline.stop()
    cv2.destroyAllWindows()