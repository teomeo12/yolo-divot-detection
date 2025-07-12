#!/usr/bin/env python3
"""
YOLO11 Divot Detection Inference Script
Use this script to run inference on new images with your trained model
"""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch

def load_model(model_path):
    """Load the trained YOLO11 model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = YOLO(model_path)
    print(f"✓ Model loaded from: {model_path}")
    return model

def run_inference(model, image_path, conf_threshold=0.5, save_results=True):
    """Run inference on a single image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Run inference
    results = model(image_path, conf=conf_threshold)
    
    # Process results
    for i, result in enumerate(results):
        # Get image with detections drawn
        annotated_image = result.plot()
        
        if save_results:
            # Save annotated image
            output_dir = "inference_results"
            os.makedirs(output_dir, exist_ok=True)
            
            input_name = Path(image_path).stem
            output_path = os.path.join(output_dir, f"{input_name}_detected.jpg")
            cv2.imwrite(output_path, annotated_image)
            print(f"✓ Results saved to: {output_path}")
        
        # Print detection details
        if result.boxes is not None:
            print(f"Found {len(result.boxes)} divots in {image_path}")
            for j, box in enumerate(result.boxes):
                confidence = box.conf.item()
                class_id = box.cls.item()
                class_name = model.names[int(class_id)]
                print(f"  Divot {j+1}: {class_name} (confidence: {confidence:.2f})")
        else:
            print(f"No divots detected in {image_path}")
    
    return results

def run_inference_folder(model, folder_path, conf_threshold=0.5):
    """Run inference on all images in a folder"""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found at {folder_path}")
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Find all images in folder
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(folder_path).glob(f"*{ext}"))
        image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    total_divots = 0
    for image_file in image_files:
        print(f"\nProcessing: {image_file}")
        results = run_inference(model, str(image_file), conf_threshold)
        
        # Count divots
        for result in results:
            if result.boxes is not None:
                total_divots += len(result.boxes)
    
    print(f"\n📊 Summary: Found {total_divots} divots across {len(image_files)} images")

def run_webcam_inference(model, conf_threshold=0.5):
    """Run real-time inference using webcam"""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return
    
    print("📹 Starting webcam inference. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model(frame, conf=conf_threshold)
        
        # Draw results
        annotated_frame = results[0].plot()
        
        # Display
        cv2.imshow('YOLO11 Divot Detection', annotated_frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✓ Webcam inference stopped")

def main():
    """Main inference function"""
    print("🎯 YOLO11 Divot Detection Inference")
    print("=" * 40)
    
    # Default model path (adjust as needed)
    model_paths = [
        "trained_models/divot_detection_best.pt",
        "runs/train/divot_detection/weights/best.pt",
        "yolo11n.pt"  # fallback to pretrained model
    ]
    
    model = None
    for model_path in model_paths:
        if os.path.exists(model_path):
            model = load_model(model_path)
            break
    
    if model is None:
        print("❌ No model found. Please train a model first or specify the correct path.")
        return
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ Using CPU (GPU not available)")
    
    print("\nInference options:")
    print("1. Single image")
    print("2. Folder of images")
    print("3. Webcam (real-time)")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    try:
        if choice == "1":
            image_path = input("Enter image path: ").strip()
            conf = float(input("Enter confidence threshold (0.1-1.0, default 0.5): ") or "0.5")
            run_inference(model, image_path, conf)
            
        elif choice == "2":
            folder_path = input("Enter folder path: ").strip()
            conf = float(input("Enter confidence threshold (0.1-1.0, default 0.5): ") or "0.5")
            run_inference_folder(model, folder_path, conf)
            
        elif choice == "3":
            conf = float(input("Enter confidence threshold (0.1-1.0, default 0.5): ") or "0.5")
            run_webcam_inference(model, conf)
            
        else:
            print("Invalid choice")
            
    except KeyboardInterrupt:
        print("\n\n⏹ Inference stopped by user")
    except Exception as e:
        print(f"❌ Error during inference: {e}")

if __name__ == "__main__":
    main() 