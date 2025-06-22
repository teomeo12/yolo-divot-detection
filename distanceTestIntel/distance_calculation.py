import numpy as np
import cv2
import pyrealsense2 as rs
import time
import argparse
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

class RealSenseDistanceMeasurement:
    """
    Intel RealSense D435i Distance Measurement and Accuracy Testing Tool
    
    The RealSense D435i has multiple sensors:
    1. Stereo depth sensors (two infrared cameras) - Primary method for depth calculation
    2. RGB camera - For visual reference
    3. IMU (Inertial Measurement Unit) - For motion tracking
    4. Infrared emitter - Projects pattern for stereo matching
    
    This class demonstrates both approaches and tests accuracy.
    """
    
    def __init__(self, enable_depth=True, enable_ir=True, enable_rgb=True):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.enable_depth = enable_depth
        self.enable_ir = enable_ir
        self.enable_rgb = enable_rgb
        
        # Measurement history for accuracy analysis
        self.measurements = defaultdict(list)
        self.known_distances = {}  # For accuracy testing
        
        # Configure streams
        self._configure_streams()
        
        # Start pipeline
        self.profile = self.pipeline.start(self.config)
        
        # Get sensors and their properties
        self._initialize_sensors()
        
        # Create alignment object (align depth to color)
        self.align_to = rs.stream.color if enable_rgb else rs.stream.depth
        self.align = rs.align(self.align_to)
        
        print("RealSense D435i initialized successfully!")
        self._print_sensor_info()
    
    def _configure_streams(self):
        """Configure camera streams based on enabled sensors"""
        if self.enable_rgb:
            # RGB camera stream - HD resolution
            self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
            
        if self.enable_depth:
            # Depth stream (uses stereo IR cameras) - HD resolution
            self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
            
        if self.enable_ir:
            # Infrared streams (left and right IR cameras) - HD resolution
            self.config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)  # Left IR
            self.config.enable_stream(rs.stream.infrared, 2, 1280, 720, rs.format.y8, 30)  # Right IR
    
    def _initialize_sensors(self):
        """Initialize sensor properties and get calibration data"""
        device = self.profile.get_device()
        
        # Get depth sensor properties
        if self.enable_depth:
            self.depth_sensor = device.first_depth_sensor()
            self.depth_scale = self.depth_sensor.get_depth_scale()
            
            # Get depth camera intrinsics
            depth_stream = self.profile.get_stream(rs.stream.depth)
            self.depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        
        # Get color camera intrinsics (if enabled)
        if self.enable_rgb:
            color_stream = self.profile.get_stream(rs.stream.color)
            self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        
        # Get IR camera intrinsics (if enabled)
        if self.enable_ir:
            ir_stream = self.profile.get_stream(rs.stream.infrared, 1)
            self.ir_intrinsics = ir_stream.as_video_stream_profile().get_intrinsics()
    
    def _print_sensor_info(self):
        """Print detailed sensor information"""
        print("\n=== SENSOR INFORMATION ===")
        print(f"Depth scale: {self.depth_scale if hasattr(self, 'depth_scale') else 'N/A'}")
        
        if hasattr(self, 'depth_intrinsics'):
            print(f"Depth camera intrinsics:")
            print(f"  Resolution: {self.depth_intrinsics.width}x{self.depth_intrinsics.height}")
            print(f"  Focal length: fx={self.depth_intrinsics.fx:.2f}, fy={self.depth_intrinsics.fy:.2f}")
            print(f"  Principal point: cx={self.depth_intrinsics.ppx:.2f}, cy={self.depth_intrinsics.ppy:.2f}")
        
        if hasattr(self, 'color_intrinsics'):
            print(f"Color camera intrinsics:")
            print(f"  Resolution: {self.color_intrinsics.width}x{self.color_intrinsics.height}")
            print(f"  Focal length: fx={self.color_intrinsics.fx:.2f}, fy={self.color_intrinsics.fy:.2f}")
    
    def get_frames(self):
        """Get aligned frames from all enabled sensors"""
        frames = self.pipeline.wait_for_frames()
        
        # Align frames
        aligned_frames = self.align.process(frames)
        
        result = {}
        
        if self.enable_depth:
            result['depth'] = aligned_frames.get_depth_frame()
        
        if self.enable_rgb:
            result['color'] = aligned_frames.get_color_frame()
        
        if self.enable_ir:
            result['ir_left'] = frames.get_infrared_frame(1)
            result['ir_right'] = frames.get_infrared_frame(2)
        
        return result
    
    def measure_distance_stereo(self, x, y, depth_frame):
        """
        Measure distance using stereo depth sensor (primary method)
        
        The D435i uses two IR cameras to calculate depth through stereo vision.
        This is the most accurate method for distance measurement.
        """
        if not depth_frame:
            return None
        
        # Get depth value at pixel (x, y) in meters
        distance = depth_frame.get_distance(x, y)
        
        if distance == 0:
            # If exact pixel has no depth, try averaging surrounding pixels
            distances = []
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if 0 <= x + dx < depth_frame.width and 0 <= y + dy < depth_frame.height:
                        d = depth_frame.get_distance(x + dx, y + dy)
                        if d > 0:
                            distances.append(d)
            
            distance = np.mean(distances) if distances else 0
        
        return distance * 1000 if distance > 0 else None  # Convert to mm
    
    def measure_distance_triangulation(self, x, y, ir_left_frame, ir_right_frame):
        """
        Measure distance using IR cameras with manual triangulation
        
        This demonstrates how stereo vision works by manually calculating
        the disparity between left and right IR images.
        """
        if not ir_left_frame or not ir_right_frame:
            return None
        
        # Convert frames to numpy arrays
        left_image = np.asanyarray(ir_left_frame.get_data())
        right_image = np.asanyarray(ir_right_frame.get_data())
        
        # Simple block matching for disparity calculation
        # In practice, the RealSense SDK does this more sophisticatedly
        
        # Define search window
        window_size = 15
        search_range = 50
        
        # Extract template from left image
        half_window = window_size // 2
        if (x - half_window < 0 or x + half_window >= left_image.shape[1] or
            y - half_window < 0 or y + half_window >= left_image.shape[0]):
            return None
        
        template = left_image[y-half_window:y+half_window+1, x-half_window:x+half_window+1]
        
        # Search for best match in right image
        best_match = 0
        best_correlation = -1
        
        for d in range(max(0, x-search_range), min(right_image.shape[1]-window_size, x+search_range)):
            if d - half_window < 0 or d + half_window >= right_image.shape[1]:
                continue
                
            candidate = right_image[y-half_window:y+half_window+1, d-half_window:d+half_window+1]
            
            # Normalized cross-correlation
            correlation = cv2.matchTemplate(template, candidate, cv2.TM_CCOEFF_NORMED)[0, 0]
            
            if correlation > best_correlation:
                best_correlation = correlation
                best_match = d
        
        # Calculate disparity
        disparity = abs(x - best_match)
        
        if disparity == 0 or best_correlation < 0.7:
            return None
        
        # Calculate distance using triangulation formula
        # Distance = (baseline * focal_length) / disparity
        baseline = 50.0  # D435i baseline is approximately 50mm
        focal_length = self.ir_intrinsics.fx if hasattr(self, 'ir_intrinsics') else 380.0
        
        distance_mm = (baseline * focal_length) / disparity
        
        return distance_mm if distance_mm > 0 else None
    
    def add_known_distance(self, name, actual_distance_mm):
        """Add a known distance for accuracy testing"""
        self.known_distances[name] = actual_distance_mm
        print(f"Added known distance: {name} = {actual_distance_mm}mm")
    
    def measure_and_record(self, x, y, frames, label="measurement"):
        """Measure distance using all available methods and record results"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'pixel_x': x,
            'pixel_y': y,
            'label': label
        }
        
        # Stereo depth measurement
        if 'depth' in frames and frames['depth']:
            stereo_distance = self.measure_distance_stereo(x, y, frames['depth'])
            results['stereo_distance_mm'] = stereo_distance
            
            if stereo_distance:
                self.measurements[f"{label}_stereo"].append(stereo_distance)
                print(f"Stereo distance: {stereo_distance:.2f}mm")
        
        # IR triangulation measurement
        if 'ir_left' in frames and 'ir_right' in frames:
            ir_distance = self.measure_distance_triangulation(x, y, frames['ir_left'], frames['ir_right'])
            results['ir_triangulation_mm'] = ir_distance
            
            if ir_distance:
                self.measurements[f"{label}_ir"].append(ir_distance)
                print(f"IR triangulation distance: {ir_distance:.2f}mm")
        
        # Calculate accuracy if known distance exists
        if label in self.known_distances:
            actual = self.known_distances[label]
            
            if 'stereo_distance_mm' in results and results['stereo_distance_mm']:
                stereo_error = abs(results['stereo_distance_mm'] - actual)
                stereo_error_percent = (stereo_error / actual) * 100
                results['stereo_error_mm'] = stereo_error
                results['stereo_error_percent'] = stereo_error_percent
                print(f"Stereo error: {stereo_error:.2f}mm ({stereo_error_percent:.2f}%)")
            
            if 'ir_triangulation_mm' in results and results['ir_triangulation_mm']:
                ir_error = abs(results['ir_triangulation_mm'] - actual)
                ir_error_percent = (ir_error / actual) * 100
                results['ir_error_mm'] = ir_error
                results['ir_error_percent'] = ir_error_percent
                print(f"IR triangulation error: {ir_error:.2f}mm ({ir_error_percent:.2f}%)")
        
        return results
    
    def get_accuracy_statistics(self):
        """Calculate accuracy statistics for all measurements"""
        stats = {}
        
        for measurement_type, values in self.measurements.items():
            if not values:
                continue
                
            stats[measurement_type] = {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'range': np.max(values) - np.min(values)
            }
        
        return stats
    
    def save_measurements(self, filename):
        """Save all measurements to a JSON file"""
        data = {
            'known_distances': self.known_distances,
            'measurements': dict(self.measurements),
            'statistics': self.get_accuracy_statistics(),
            'sensor_info': {
                'depth_scale': getattr(self, 'depth_scale', None),
                'depth_intrinsics': {
                    'fx': self.depth_intrinsics.fx,
                    'fy': self.depth_intrinsics.fy,
                    'ppx': self.depth_intrinsics.ppx,
                    'ppy': self.depth_intrinsics.ppy,
                    'width': self.depth_intrinsics.width,
                    'height': self.depth_intrinsics.height
                } if hasattr(self, 'depth_intrinsics') else None
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Measurements saved to {filename}")
    
    def create_accuracy_plot(self, save_path=None):
        """Create visualization of measurement accuracy"""
        if not self.measurements:
            print("No measurements to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('RealSense D435i Distance Measurement Accuracy Analysis')
        
        # Plot 1: Measurement comparison
        ax1 = axes[0, 0]
        for measurement_type, values in self.measurements.items():
            ax1.hist(values, alpha=0.7, label=measurement_type, bins=20)
        ax1.set_xlabel('Distance (mm)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distance Measurement Distribution')
        ax1.legend()
        
        # Plot 2: Statistics
        ax2 = axes[0, 1]
        stats = self.get_accuracy_statistics()
        types = list(stats.keys())
        means = [stats[t]['mean'] for t in types]
        stds = [stats[t]['std'] for t in types]
        
        x_pos = np.arange(len(types))
        ax2.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
        ax2.set_xlabel('Measurement Type')
        ax2.set_ylabel('Distance (mm)')
        ax2.set_title('Mean Distance ± Standard Deviation')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(types, rotation=45)
        
        # Plot 3: Range comparison
        ax3 = axes[1, 0]
        ranges = [stats[t]['range'] for t in types]
        ax3.bar(types, ranges, alpha=0.7)
        ax3.set_xlabel('Measurement Type')
        ax3.set_ylabel('Range (mm)')
        ax3.set_title('Measurement Range (Max - Min)')
        plt.setp(ax3.get_xticklabels(), rotation=45)
        
        # Plot 4: Precision comparison
        ax4 = axes[1, 1]
        precisions = [stats[t]['std'] for t in types]
        ax4.bar(types, precisions, alpha=0.7)
        ax4.set_xlabel('Measurement Type')
        ax4.set_ylabel('Standard Deviation (mm)')
        ax4.set_title('Measurement Precision (Lower = Better)')
        plt.setp(ax4.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Accuracy plot saved to {save_path}")
        
        plt.show()
    
    def cleanup(self):
        """Stop the pipeline and cleanup resources"""
        self.pipeline.stop()
        cv2.destroyAllWindows()

def interactive_measurement_tool():
    """Interactive tool for measuring distances and testing accuracy"""
    parser = argparse.ArgumentParser(description='RealSense D435i Distance Measurement Tool')
    parser.add_argument('--enable-ir', action='store_true', help='Enable IR camera streams for triangulation')
    parser.add_argument('--output-dir', default='distance_measurements', help='Output directory for results')
    parser.add_argument('--save-images', action='store_true', help='Save captured images')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize RealSense
    try:
        rs_tool = RealSenseDistanceMeasurement(
            enable_depth=True,
            enable_ir=args.enable_ir,
            enable_rgb=True
        )
    except Exception as e:
        print(f"Error initializing RealSense camera: {e}")
        print("Make sure the camera is connected and pyrealsense2 is installed:")
        print("pip install pyrealsense2")
        return
    
    # Create window and set mouse callback
    cv2.namedWindow('Distance Measurement Tool', cv2.WINDOW_AUTOSIZE)
    
    # Mouse callback for clicking to measure
    click_pos = None
    measurement_label = "measurement"
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal click_pos, measurement_label
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pos = (x, y)
            print(f"\nMeasuring at pixel ({x}, {y}) - Label: {measurement_label}")
    
    cv2.setMouseCallback('Distance Measurement Tool', mouse_callback)
    
    print("\n=== DISTANCE MEASUREMENT TOOL ===")
    print("Usage:")
    print("- Click on objects to measure distance")
    print("- Press 'k' to add a known distance for accuracy testing")
    print("- Press 'l' to change measurement label")
    print("- Press 's' to save measurements to file")
    print("- Press 'p' to generate accuracy plot")
    print("- Press 'r' to reset all measurements")
    print("- Press ESC to exit")
    print(f"\nSensor configuration:")
    print(f"- Stereo depth: Enabled")
    print(f"- IR triangulation: {'Enabled' if args.enable_ir else 'Disabled'}")
    print(f"- RGB camera: Enabled")
    
    try:
        while True:
            # Get frames
            frames = rs_tool.get_frames()
            
            if not frames:
                continue
            
            # Display image
            display_image = None
            if 'color' in frames and frames['color']:
                display_image = np.asanyarray(frames['color'].get_data())
            elif 'ir_left' in frames and frames['ir_left']:
                ir_image = np.asanyarray(frames['ir_left'].get_data())
                display_image = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR)
            
            if display_image is None:
                continue
            
            # Add crosshair and instructions
            h, w = display_image.shape[:2]
            cv2.line(display_image, (w//2-10, h//2), (w//2+10, h//2), (0, 255, 0), 1)
            cv2.line(display_image, (w//2, h//2-10), (w//2, h//2+10), (0, 255, 0), 1)
            
            # Add text overlay
            cv2.putText(display_image, f"Label: {measurement_label}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_image, "Click to measure distance", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Process click
            if click_pos:
                x, y = click_pos
                cv2.circle(display_image, (x, y), 5, (0, 0, 255), -1)
                
                # Measure distance
                results = rs_tool.measure_and_record(x, y, frames, measurement_label)
                
                # Display results on image
                text_y = 90
                for key, value in results.items():
                    if 'distance' in key.lower() and value:
                        cv2.putText(display_image, f"{key}: {value:.1f}mm", (10, text_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        text_y += 20
                
                click_pos = None
                
                # Save image if requested
                if args.save_images:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    image_path = os.path.join(args.output_dir, f"measurement_{timestamp}.jpg")
                    cv2.imwrite(image_path, display_image)
            
            cv2.imshow('Distance Measurement Tool', display_image)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            elif key == ord('k'):  # Add known distance
                print("\nAdd known distance:")
                name = input("Enter label name: ").strip()
                try:
                    distance = float(input("Enter actual distance (mm): "))
                    rs_tool.add_known_distance(name, distance)
                    measurement_label = name
                except ValueError:
                    print("Invalid distance value")
            elif key == ord('l'):  # Change label
                new_label = input("\nEnter new measurement label: ").strip()
                if new_label:
                    measurement_label = new_label
                    print(f"Label changed to: {measurement_label}")
            elif key == ord('s'):  # Save measurements
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(args.output_dir, f"measurements_{timestamp}.json")
                rs_tool.save_measurements(filename)
            elif key == ord('p'):  # Generate plot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_path = os.path.join(args.output_dir, f"accuracy_plot_{timestamp}.png")
                rs_tool.create_accuracy_plot(plot_path)
            elif key == ord('r'):  # Reset measurements
                rs_tool.measurements.clear()
                rs_tool.known_distances.clear()
                print("All measurements cleared")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        rs_tool.cleanup()
        print("Camera stopped")

if __name__ == "__main__":
    interactive_measurement_tool()
