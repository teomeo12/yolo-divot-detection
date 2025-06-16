import numpy as np
import cv2
import torch
import os
import argparse
import time
import pyrealsense2 as rs
import open3d as o3d
import sys
# Add repository to path and import
sys.path.insert(0, './Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2

# Device selection for PyTorch
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

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
    
    # Get the intrinsics
    color_stream = profile.get_stream(rs.stream.color)
    color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
    
    print(f"Camera intrinsics: fx={color_intrinsics.fx}, fy={color_intrinsics.fy}, ppx={color_intrinsics.ppx}, ppy={color_intrinsics.ppy}")
    
    # Wait for camera to warm up
    print("Warming up camera...")
    for i in range(10):
        pipeline.wait_for_frames()
    
    return pipeline, color_intrinsics

def create_point_cloud(color_image, depth_map, intrinsics):
    """Create a colored point cloud from color image and depth map"""
    height, width = depth_map.shape
    
    # Create Open3D image objects
    depth_image = o3d.geometry.Image(depth_map.astype(np.float32))
    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    color_o3d = o3d.geometry.Image(color_image_rgb)
    
    # Create Open3D pinhole camera intrinsic
    o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        width, height, 
        intrinsics.fx, intrinsics.fy, 
        intrinsics.ppx, intrinsics.ppy
    )
    
    # Create RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_image, 
        depth_scale=1000.0,  # Adjust depth scale if needed
        depth_trunc=10.0,  # Maximum depth in meters
        convert_rgb_to_intensity=False
    )
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, o3d_intrinsics
    )
    
    # Flip it, as the default coordinate system uses a different convention
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
    return pcd

def visualize_point_cloud(pcd, output_dir=None):
    """Visualize the point cloud in a separate window"""
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window("3D Point Cloud")
    
    # Add geometry
    vis.add_geometry(pcd)
    
    # Add coordinate frame
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(coordinate_frame)
    
    # Set initial viewpoint
    view_control = vis.get_view_control()
    view_control.set_front([0, 0, -1])
    view_control.set_lookat([0, 0, 0])
    view_control.set_up([0, -1, 0])
    view_control.set_zoom(0.8)
    
    # Set rendering options for better visualization
    opt = vis.get_render_option()
    opt.point_size = 2
    opt.background_color = np.array([0.1, 0.1, 0.1])
    
    # Save point cloud if output directory is specified
    if output_dir is not None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        pcd_path = os.path.join(output_dir, f"pointcloud_{timestamp}.ply")
        o3d.io.write_point_cloud(pcd_path, pcd)
        print(f"Saved point cloud to {pcd_path}")
    
    # Run the visualizer
    print("Displaying 3D point cloud. Close the window to continue.")
    vis.run()
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description='Display RGB and depth video from Intel RealSense camera and create 3D point cloud')
    parser.add_argument('--depth-model', default='checkpoints/depth_anything_v2_vits.pth',
                       help='Path to the Depth Anything V2 model')
    parser.add_argument('--output', default='pointclouds',
                       help='Output directory for saved point clouds')
    
    args = parser.parse_args()
    
    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        print(f"Point clouds will be saved to: {args.output}")
    
    # Initialize depth model
    depth_model = initialize_depth_model(args.depth_model)
    if depth_model is None:
        return
    
    # Initialize camera
    try:
        pipeline, intrinsics = initialize_realsense()
        
        # Create windows
        cv2.namedWindow('RGB and Depth', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('RGB and Depth', 1280, 480)  # Size for side-by-side display
        
        print("\nPress SPACE to capture and create 3D point cloud")
        print("Press 'q' to exit\n")
        
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
            cv2.putText(combined_image, "Press SPACE to create 3D mesh", (400, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show images
            cv2.imshow('RGB and Depth', combined_image)
            
            # Handle keyboard input
            key = cv2.waitKey(1)
            
            # Press q to exit
            if key & 0xFF == ord('q'):
                print("Exiting...")
                break
            
            # Press SPACE to create and visualize 3D point cloud
            elif key == 32:  # SPACE
                print("Creating 3D point cloud from current frame...")
                
                # Create point cloud
                pcd = create_point_cloud(color_image, depth_map, intrinsics)
                
                # Save current frame if output directory is specified
                if args.output:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    # Save RGB image
                    rgb_path = os.path.join(args.output, f"rgb_{timestamp}.jpg")
                    cv2.imwrite(rgb_path, color_image)
                    # Save depth image
                    depth_path = os.path.join(args.output, f"depth_{timestamp}.jpg")
                    cv2.imwrite(depth_path, depth_colored)
                    print(f"Saved RGB image to {rgb_path}")
                    print(f"Saved depth image to {depth_path}")
                
                # Visualize point cloud (this will block until the visualization window is closed)
                visualize_point_cloud(pcd, args.output)
                
                print("Continuing video stream. Press SPACE to capture again or 'q' to exit.")
        
        # Cleanup
        pipeline.stop()
        cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 