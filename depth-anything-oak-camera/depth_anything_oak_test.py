import depthai as dai
import cv2
import numpy as np
import torch
import os
import sys
import argparse

# Add repository to path and import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Depth-Anything-V2')))
from depth_anything_v2.dpt import DepthAnythingV2

# Device selection for PyTorch
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def initialize_depth_model(model_path='checkpoints/depth_anything_v2_vits.pth'):
    """Initialize the Depth Anything V2 model"""
    # Adjust path to be relative to the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if not os.path.isabs(model_path):
        model_path = os.path.join(project_root, model_path)
        
    print(f"Loading depth model from: {model_path}")
    
    # Configure for vits model (smaller and faster)
    model_config = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    
    # Create model and load weights
    depth_model = DepthAnythingV2(**model_config)
    
    if not os.path.exists(model_path):
        print(f"Searched path: {os.path.abspath(model_path)}")
        return None
        
    depth_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    depth_model = depth_model.to(DEVICE).eval()
    return depth_model

def main():
    parser = argparse.ArgumentParser(description='Display RGB and depth video from OAK-D camera using Depth Anything V2')
    parser.add_argument('--depth-model', default='checkpoints/depth_anything_v2_vits.pth',
                       help='Path to the Depth Anything V2 model')
    
    args = parser.parse_args()

    # Initialize depth model
    depth_model = initialize_depth_model(args.depth_model)
    if depth_model is None:
        return

    # Create a pipeline
    pipeline = dai.Pipeline()

    # Create a ColorCamera node
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setPreviewSize(640, 480)
    camRgb.setInterleaved(False)

    # Create an XLinkOut node to stream data
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    camRgb.preview.link(xoutRgb.input)

    # Connect to device and start the pipeline
    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        
        # Create windows
        cv2.namedWindow('RGB and Depth', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('RGB and Depth', 1280, 480)  # Size for side-by-side display
        
        print("\nPress 'q' to exit\n")
        
        while True:
            inRgb = qRgb.get()  # Blocking call, waits for new data
            color_image = inRgb.getCvFrame()
            
            if color_image is not None:
                # Get depth map from Depth Anything V2
                depth_map = depth_model.infer_image(color_image)
                
                # Normalize depth for visualization
                depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
                
                # Create side by side view
                depth_resized = cv2.resize(depth_colored, (640, 480))
                combined_image = np.hstack((color_image, depth_resized))
                
                # Add labels
                cv2.putText(combined_image, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(combined_image, "Depth", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Show images
                cv2.imshow('RGB and Depth', combined_image)
            
            if cv2.waitKey(1) == ord('q'):
                print("Exiting...")
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()