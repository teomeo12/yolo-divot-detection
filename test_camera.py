#!/usr/bin/env python3

import pyrealsense2 as rs
import numpy as np
import cv2

def test_camera():
    print("Testing Intel RealSense camera...")
    
    # Create a context object
    ctx = rs.context()
    devices = ctx.query_devices()
    
    print(f"Found {len(devices)} device(s)")
    
    if len(devices) == 0:
        print("No RealSense devices found!")
        return False
    
    for i, device in enumerate(devices):
        print(f"Device {i}: {device.get_info(rs.camera_info.name)}")
        print(f"  Serial: {device.get_info(rs.camera_info.serial_number)}")
        print(f"  Firmware: {device.get_info(rs.camera_info.firmware_version)}")
    
    # Try to create a pipeline
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Configure streams
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        print("Starting pipeline...")
        profile = pipeline.start(config)
        
        print("Pipeline started successfully!")
        
        # Get a few frames
        for i in range(5):
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if depth_frame and color_frame:
                print(f"Frame {i+1}: Got depth and color frames")
            else:
                print(f"Frame {i+1}: Missing frames")
        
        pipeline.stop()
        print("Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_camera() 