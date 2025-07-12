#!/usr/bin/env python3
"""
YOLO11 GPU Training Script for Divot Detection
This script downloads the dataset from Roboflow and trains a YOLO11 model using GPU
"""

import os
import sys
from pathlib import Path
import subprocess
import torch
from ultralytics import YOLO
from roboflow import Roboflow

def setup_ultralytics():
    """Configure ultralytics settings"""
    import ultralytics
    ultralytics.checks()
    
    # Disable tracking
    os.system("yolo settings sync=False")
    print("✓ Ultralytics configured")

def download_dataset():
    """Download dataset from Roboflow"""
    
    
    print("📥 Downloading dataset from Roboflow...")
    rf = Roboflow(api_key="SqI8eWr5qmxgVkvnKMZO")
    project = rf.workspace("divotdetection").project("divots-instance-category")
    version = project.version(2)
    dataset = version.download("yolov11")
    
    print(f"✓ Dataset downloaded to: {dataset.location}")
    return dataset.location

def check_gpu():
    """Check if GPU is available"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✓ GPU available: {gpu_name} (Count: {gpu_count})")
        return True
    else:
        print("⚠ No GPU detected, training will use CPU (much slower)")
        return False

def train_model(dataset_path, model_size="yolo11n.pt", epochs=100, imgsz=640, batch_size=16):
    """Train YOLO11 model"""
    print(f"🚀 Starting training with {model_size}...")
    
    # Initialize model
    model = YOLO(model_size)
    
    # Find data.yaml file
    data_yaml = os.path.join(dataset_path, "data.yaml")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found at {data_yaml}")
    
    # Training parameters
    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': imgsz,
        'batch': batch_size,
        'device': 0 if torch.cuda.is_available() else 'cpu',  # Use GPU 0 if available
        'project': 'runs/train',
        'name': 'divot_detection',
        'save': True,
        'save_period': 10,  # Save checkpoint every 10 epochs
        'patience': 50,     # Early stopping patience
        'workers': 8,       # Number of dataloader workers
        'cache': True,      # Cache images for faster training
    }
    
    print(f"Training configuration: {train_args}")
    
    # Train the model
    results = model.train(**train_args)
    
    print("✓ Training completed!")
    return results, model

def save_model(model, output_dir="trained_models"):
    """Save the trained model"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save in different formats
    model_path = os.path.join(output_dir, "divot_detection_best.pt")
    onnx_path = os.path.join(output_dir, "divot_detection_best.onnx")
    
    # Copy best weights
    best_weights = "runs/train/divot_detection/weights/best.pt"
    if os.path.exists(best_weights):
        import shutil
        shutil.copy2(best_weights, model_path)
        print(f"✓ Model saved to: {model_path}")
        
        # Export to ONNX for deployment
        try:
            model.export(format="onnx", imgsz=640)
            print(f"✓ ONNX model exported")
        except Exception as e:
            print(f"⚠ ONNX export failed: {e}")
    
    return model_path

def main():
    """Main training pipeline"""
    print("🎯 YOLO11 Divot Detection Training Pipeline")
    print("=" * 50)
    
    # Step 1: Setup ultralytics
    print("\n1. Setting up ultralytics...")
    setup_ultralytics()
    
    # Step 2: Check GPU
    print("\n2. Checking GPU availability...")
    gpu_available = check_gpu()
    
    # Step 3: Download dataset
    print("\n3. Downloading dataset...")
    dataset_path = download_dataset()
    
    # Step 4: Train model
    print("\n4. Training model...")
    try:
        # You can adjust these parameters based on your needs
        results, model = train_model(
            dataset_path=dataset_path,
            model_size="yolo11n.pt",  # Options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
            epochs=100,               # Adjust based on your dataset size
            imgsz=640,               # Image size
            batch_size=16 if gpu_available else 4  # Adjust based on GPU memory
        )
        
        # Step 5: Save model
        print("\n5. Saving model...")
        model_path = save_model(model)
        
        print("\n🎉 Training completed successfully!")
        print(f"📁 Model saved at: {model_path}")
        print(f"📊 Training results saved in: runs/train/divot_detection/")
        print(f"📈 View training plots at: runs/train/divot_detection/results.png")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ All done! Your model is ready for inference.")
    else:
        print("\n❌ Training failed. Please check the error messages above.")