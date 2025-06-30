# File: divot_area_live.py
# Purpose: Use a YOLOv11 model to detect divots in a live RealSense camera feed
# and calculate their real-world area using the depth stream.

import cv2
import numpy as np
import pyrealsense2 as rs
import supervision as sv
from ultralytics import YOLO
import os

# --- Tweakable Parameters ---
# Use the calibration factor you found previously for your test object
CALIBRATION_FACTOR = 0.52 # for a circle area
# The area calibration factor is the square of the linear one.
AREA_CALIBRATION_FACTOR = CALIBRATION_FACTOR ** 2

# How many pixels to expand the mask to find the surrounding "ground"
GROUND_RING_WIDTH = 15
# --- End of Parameters ---

def main():
    # --- 1. INITIALIZE MODELS AND CAMERA ---

    # Load YOLO Model
    # Construct the path to the model relative to this script's location
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, '1600s_aug_100ep.pt')

        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            # Try a relative path from the CWD as a fallback
            workspace_dir = os.path.dirname(script_dir) # Go up one level
            model_path = os.path.join(workspace_dir, 'yolo11s_1600_100ep', '1600s_aug_100ep.pt')
            if not os.path.exists(model_path):
                 raise FileNotFoundError(f"Model not found in primary or fallback path.")

        model = YOLO(model_path)
        print("YOLO model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return

    # Initialize Supervision Annotators
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.CLASS, opacity=0.5)
    box_annotator = sv.BoxAnnotator()

    # Initialize RealSense Pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    print("Starting RealSense pipeline...")
    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"Error starting RealSense pipeline: {e}")
        print("Please make sure the RealSense camera is connected and not in use by another application.")
        return

    # Create an align object to map depth to color
    align = rs.align(rs.stream.color)

    # Get camera intrinsics for area calculation
    intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    fx = intrinsics.fx
    fy = intrinsics.fy

    print("Pipeline started. Press 'q' to quit.")

    # --- 2. LIVE PROCESSING LOOP ---
    try:
        while True:
            # Get aligned frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            annotated_frame = color_image.copy()

            # --- 3. YOLO DETECTION ---
            results = model(color_image, verbose=False)
            all_detections = sv.Detections.from_ultralytics(results[0])

            # Get the class names from the model
            class_names = results[0].names
            divot_detections = None
            
            try:
                # Find the class ID for 'divot'
                divot_class_id = [k for k, v in class_names.items() if v == 'divot'][0]
                # Filter detections to only include 'divot'
                divot_detections = all_detections[all_detections.class_id == divot_class_id]
            except IndexError:
                # This error means the model doesn't have a 'divot' class.
                # We can skip the area calculation part.
                pass

            # Annotate the frame with boxes and masks of ALL detected objects for context
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=all_detections)
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=all_detections)
            
            # --- 4. AREA CALCULATION FOR EACH DETECTED DIVOT ---
            if divot_detections and len(divot_detections) > 0 and divot_detections.mask is not None:
                # Iterate through each detected divot
                for i in range(len(divot_detections)):
                    # Get the individual mask for this detection (boolean array)
                    divot_mask_bool = divot_detections.mask[i]
                    
                    # Convert boolean mask to uint8 mask for OpenCV functions
                    divot_mask_uint8 = divot_mask_bool.astype(np.uint8) * 255

                    # Create a "ground ring" around the divot mask
                    ring_kernel = np.ones((GROUND_RING_WIDTH, GROUND_RING_WIDTH), np.uint8)
                    dilated_mask = cv2.dilate(divot_mask_uint8, ring_kernel, iterations=1)
                    ground_ring_mask = cv2.bitwise_and(dilated_mask, cv2.bitwise_not(divot_mask_uint8))

                    # Get depth values for the ground ring from the live depth image
                    ground_depths = depth_image[ground_ring_mask == 255]
                    ground_depths_mm = ground_depths[ground_depths != 0]

                    if ground_depths_mm.size > 0:
                        ground_level_mm = np.mean(ground_depths_mm)

                        # For a boolean mask, the area is the number of true pixels
                        contour_pixel_area = np.sum(divot_mask_bool)

                        # Estimate the area of a single pixel at the ground depth
                        pixel_area_m2 = ((ground_level_mm / 1000) / fx) * ((ground_level_mm / 1000) / fy)
                        pixel_area_cm2 = pixel_area_m2 * 10000

                        # Calculate the total calibrated area
                        total_area_cm2 = (contour_pixel_area * pixel_area_cm2) * AREA_CALIBRATION_FACTOR

                        # --- 5. VISUALIZATION ---
                        # Get the bounding box to position the text
                        box = divot_detections.xyxy[i]
                        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                        # Calculate width and height of the bounding box in pixels
                        pixel_width = x2 - x1
                        pixel_height = y2 - y1
                        
                        # Convert pixel dimensions to real-world dimensions (cm) using the linear calibration factor
                        width_cm = (pixel_width * (ground_level_mm / 1000) / fx) * 100 * CALIBRATION_FACTOR
                        height_cm = (pixel_height * (ground_level_mm / 1000) / fy) * 100 * CALIBRATION_FACTOR
                        
                        # Calculate the area of the bounding box
                        bbox_area_cm2 = width_cm * height_cm

                        # Display the dimensions on the annotated frame
                        dim_text = f"Dims: {width_cm:.1f}x{height_cm:.1f} cm"
                        cv2.putText(annotated_frame, dim_text, (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                        # Display the bounding box area
                        bbox_area_text = f"Bbox Area: {bbox_area_cm2:.1f} cm^2"
                        cv2.putText(annotated_frame, bbox_area_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        # Display the original, more accurate mask area
                        area_text = f"Area: {total_area_cm2:.1f} cm^2"
                        cv2.putText(annotated_frame, area_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Display the final annotated frame
            cv2.imshow("Live Divot Detection and Area Calculation", annotated_frame)

            # Quit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # --- 6. CLEANUP ---
        print("\nStopping pipeline and closing windows.")
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 