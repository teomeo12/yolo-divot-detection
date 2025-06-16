import numpy as np
import cv2
import torch
import os
import argparse
import pyrealsense2 as rs
import sys
# Add repository to path and import
sys.path.insert(0, './Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2

# Device selection for PyTorch
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
q
def initialize_depth_model(model_path='checkpoints/depth_anything_v2_vits.pth'):
    """Initialize the Depth Anything V2 model"""
    print(f"Loading depth model from: {model_path}")
    
    # Configure for vits model (smaller and faster)
    model_config = {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    
    # Create model and load weights
    depth_model = DepthAnythingV2(**model_config)
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        print("Please specify the correct path to the model file using the --depth-model argument")
        return None
        
    depth_model.load_state_dict(torch.load(model_path, map_location='cpu'))
    depth_model = depth_model.to(DEVICE).eval()
    return depth_model

def initialize_realsense():
    """Initialize and configure the RealSense camera pipeline"""
    print("Initializing RealSense camera...")
    
    # Create a pipeline
    pipeline = rs.pipeline()
    
    # Create a config
    config = rs.config()
    
    # Enable color stream - lower resolution for better performance
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start the pipeline
    profile = pipeline.start(config)
    
    # Wait for camera to warm up
    print("Warming up camera...")
    for i in range(10):
        pipeline.wait_for_frames()
    
    return pipeline

def main():
    parser = argparse.ArgumentParser(description='Display RGB and depth video from Intel RealSense camera')
    parser.add_argument('--depth-model', default='checkpoints/depth_anything_v2_vits.pth',
                       help='Path to the Depth Anything V2 model')
    
    args = parser.parse_args()
    
    # Initialize depth model
    depth_model = initialize_depth_model(args.depth_model)
    if depth_model is None:
        return
    
    # Initialize camera
    try:
        pipeline = initialize_realsense()
        
        # Create windows
        cv2.namedWindow('RGB and Depth', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('RGB and Depth', 1280, 480)  # Size for side-by-side display
        
        print("\nPress 'q' to exit\n")
        
        while True:
            # Wait for a coherent pair of frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue
            
            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            
            # Get depth map from Depth Anything V2
            depth_map = depth_model.infer_image(color_image)
            
            # Normalize depth for visualization
            depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)
            
            # Create side by side view
            rgb_resized = cv2.resize(color_image, (640, 480))
            depth_resized = cv2.resize(depth_colored, (640, 480))
            combined_image = np.hstack((rgb_resized, depth_resized))
            
            # Add labels
            cv2.putText(combined_image, "RGB", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined_image, "Depth", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show images
            cv2.imshow('RGB and Depth', combined_image)
            
            # Press q to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break
        
        # Cleanup
        pipeline.stop()
        cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 