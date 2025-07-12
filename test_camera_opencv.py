#!/usr/bin/env python3

import cv2
import numpy as np

def test_camera_opencv():
    print("Testing Intel RealSense camera with OpenCV...")
    
    # Try to open different video devices
    for i in range(6):
        print(f"\nTrying /dev/video{i}...")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            # Get some properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"  Device {i}: {width}x{height} @ {fps} FPS")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                print(f"  Successfully read frame: {frame.shape}")
                
                # Show the frame for a moment
                cv2.imshow(f'Camera {i}', frame)
                cv2.waitKey(1000)  # Show for 1 second
                cv2.destroyAllWindows()
            else:
                print(f"  Could not read frame from device {i}")
                
            cap.release()
        else:
            print(f"  Could not open device {i}")

if __name__ == "__main__":
    test_camera_opencv() 