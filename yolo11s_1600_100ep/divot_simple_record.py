import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from datetime import datetime

def initialize_camera():
    """Initialize RealSense camera for color stream only"""
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable color stream only
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    print("Starting camera...")
    pipeline.start(config)
    
    print("Camera started. Press 'q' to quit, 'SPACE' to start/stop recording.")
    return pipeline

def create_video_writer(width, height, fps=20):
    """Create a video writer for MP4 format"""
    # Create simple_rec_videos directory if it doesn't exist
    video_dir = "simple_rec_videos"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"simple_recording_{timestamp}.mp4"
    filepath = os.path.join(video_dir, filename)
    
    # Try different codecs for better compatibility
    # First try H264 (best compatibility)
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
    
    # If H264 fails, try XVID
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
    
    # If XVID fails, fallback to mp4v
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
    
    print(f"Recording to: {filepath} at {fps} FPS")
    return writer, filepath

def main():
    # Video recording variables
    recording = False
    video_writer = None
    current_video_path = None
    
    # Frame rate control
    target_fps = 20
    frame_time = 1.0 / target_fps
    last_frame_time = 0
    
    try:
        # Initialize camera
        pipeline = initialize_camera()
        
        while True:
            # Frame rate control
            current_time = time.time()
            if current_time - last_frame_time < frame_time:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                continue
            last_frame_time = current_time
            
            # Wait for frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
            
            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            display_image = color_image.copy()
            
            # Add recording status indicator
            if recording:
                cv2.putText(display_image, "RECORDING", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.circle(display_image, (600, 450), 10, (0, 0, 255), -1)  # Red dot
            else:
                cv2.putText(display_image, "Press SPACE to record", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show the camera feed
            cv2.imshow('Intel RealSense Camera - Simple Recording', display_image)
            
            # Write frame to video if recording
            if recording and video_writer is not None:
                video_writer.write(display_image)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("Exiting program...")
                break
            elif key == ord(' '):  # Space key
                if not recording:
                    # Start recording
                    height, width = display_image.shape[:2]
                    video_writer, current_video_path = create_video_writer(width, height)
                    recording = True
                    print("Recording started!")
                else:
                    # Stop recording
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                        print(f"Recording stopped! Video saved to: {current_video_path}")
                    recording = False
                    current_video_path = None
                
    finally:
        # Clean up
        if video_writer is not None:
            video_writer.release()
            print(f"Final video saved to: {current_video_path}")
        
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 