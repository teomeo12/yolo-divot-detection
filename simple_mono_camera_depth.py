import numpy as np
import cv2
import torch
import torch.nn.functional as F
import os
import argparse
import sys
from torchvision.transforms import Compose
# Add repository to path and import
sys.path.insert(0, './Depth-Anything-V2')
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

# Device selection for PyTorch
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def initialize_depth_model(model_path='checkpoints/depth_anything_v2_vits.pth'):
    """Initialize the Depth Anything V2 model"""
    print(f"Loading depth model from: {model_path}")
    
    # Create model with ViT-Small backbone
    depth_model = DepthAnything(
        encoder='vits',
        features=64,
        out_channels=[48, 96, 192, 384],
        use_pretrained=False,
        checkpointing=False
    )

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return None
        
    # Load the weights
    state_dict = torch.load(model_path, map_location='cpu')
    depth_model.load_state_dict(state_dict)
    depth_model = depth_model.to(DEVICE).eval()
    
    print(f'Total parameters: {sum(param.numel() for param in depth_model.parameters()) / 1e6:.2f}M')
    
    return depth_model

def find_camera():
    """Try to find the USB camera by testing different indices"""
    for index in range(10):  # Try indices 0 through 9
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Found camera at index {index}")
                print(f"Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
                print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
                return cap, index
            cap.release()
    return None, None

def main():
    parser = argparse.ArgumentParser(description='Display RGB and depth video from USB mono camera')
    parser.add_argument('--camera-id', type=int, default=None,
                       help='Camera device ID (will auto-detect if not specified)')
    
    args = parser.parse_args()
    
    # Initialize depth model
    depth_model = initialize_depth_model()
    if depth_model is None:
        return
        
    # Setup image transforms
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    # Initialize camera
    if args.camera_id is not None:
        cap = cv2.VideoCapture(args.camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera device {args.camera_id}")
            return
        camera_id = args.camera_id
    else:
        cap, camera_id = find_camera()
        if cap is None:
            print("Error: Could not find any working camera!")
            return
    
    print(f"Using camera {camera_id}")
    print(f"Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    
    try:
        # Create window
        cv2.namedWindow('RGB and Depth', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('RGB and Depth', 1280, 480)  # Size for side-by-side display
        
        print("\nPress 'q' to exit\n")
        
        while True:
            # Capture frame
            ret, color_image = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Convert BGR to RGB
            image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) / 255.0
            
            # Transform image
            image = transform({'image': image})['image']
            image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)
            
            # Get depth prediction
            with torch.no_grad():
                depth = depth_model(image)
            
            # Process depth map
            h, w = color_image.shape[:2]
            depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.cpu().numpy().astype(np.uint8)
            
            # Colorize depth map
            depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
            
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
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed")

if __name__ == "__main__":
    main() 