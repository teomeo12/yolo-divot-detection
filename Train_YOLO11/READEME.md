# YOLO11 Divot Detection Training

This project contains scripts to train a YOLO11 model for divot detection using your local GPU instead of Google Colab.

## 📋 Prerequisites

- **NVIDIA GPU** with CUDA support (recommended)
- **Python 3.8+**
- **CUDA and cuDNN** installed (for GPU acceleration)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Alternatively, install individually:
pip install "ultralytics<=8.3.40" supervision roboflow torch torchvision opencv-python
```

### 2. Train the Model

```bash
python train_gpu.py
```

The training script will automatically:
- ✅ Install required packages
- ✅ Download your dataset from Roboflow  
- ✅ Configure GPU training
- ✅ Train YOLO11 model
- ✅ Save the trained model

### 3. Run Inference

```bash
python inference.py
```

Choose from:
- Single image inference
- Batch processing (folder of images)
- Real-time webcam detection

## 📁 Project Structure

```
Train_YOLO11/
├── train_gpu.py          # Main training script
├── inference.py          # Inference script  
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── trained_models/      # Your trained models (created after training)
├── runs/                # Training logs and results (created after training)
└── divots-instance-category-2/  # Downloaded dataset (created after training)
```

## ⚙️ Training Configuration

You can modify training parameters in `train_gpu.py`:

```python
# Training parameters (line ~87)
results, model = train_model(
    dataset_path=dataset_path,
    model_size="yolo11n.pt",    # Model size: n, s, m, l, x
    epochs=100,                 # Number of training epochs
    imgsz=640,                 # Image size
    batch_size=16              # Batch size (adjust for your GPU memory)
)
```

### Model Size Options:
- `yolo11n.pt` - Nano (fastest, smallest)
- `yolo11s.pt` - Small  
- `yolo11m.pt` - Medium
- `yolo11l.pt` - Large
- `yolo11x.pt` - Extra Large (slowest, most accurate)

## 🖥️ GPU Requirements

### Recommended GPU Memory:
- **4GB**: batch_size=8, yolo11n/s
- **8GB**: batch_size=16, yolo11m
- **12GB+**: batch_size=32, yolo11l/x

## 📊 Training Outputs

After training, you'll find:

```
runs/train/divot_detection/
├── weights/
│   ├── best.pt          # Best model weights
│   └── last.pt          # Last epoch weights
├── results.png          # Training curves
├── confusion_matrix.png # Model performance matrix
├── val_batch0_pred.png  # Validation predictions
└── train_batch0.jpg     # Training batch sample
```

## 🔍 Using Your Trained Model

### Load and Use Model:
```python
from ultralytics import YOLO

# Load your trained model
model = YOLO('trained_models/divot_detection_best.pt')

# Run inference
results = model('path/to/image.jpg')

# Display results
results[0].show()
```

### Export for Deployment:
```python
# Export to different formats
model.export(format="onnx")      # ONNX
model.export(format="tensorrt")  # TensorRT (NVIDIA)
model.export(format="coreml")    # CoreML (Apple)
```

## 🛠️ Troubleshooting

### CUDA Out of Memory:
- Reduce `batch_size` in training parameters
- Use smaller model size (`yolo11n.pt`)
- Reduce `imgsz` to 416 or 320

### No GPU Detected:
- Check CUDA installation: `nvidia-smi`
- Verify PyTorch GPU support: `python -c "import torch; print(torch.cuda.is_available())"`
- Install CUDA-compatible PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### Dataset Download Issues:
- Check your Roboflow API key
- Ensure internet connection
- Verify workspace and project names

## 📈 Monitoring Training

Monitor training progress:
- Watch console output for real-time metrics
- View training plots: `runs/train/divot_detection/results.png`
- Use TensorBoard: `tensorboard --logdir runs/train`

## 🎯 Performance Tips

1. **Data Quality**: Ensure good quality, diverse training images
2. **Augmentation**: YOLO11 includes built-in augmentations
3. **Early Stopping**: Training stops automatically if no improvement
4. **Transfer Learning**: Starting from pretrained weights (automatic)
5. **Validation**: Monitor validation metrics to avoid overfitting

## 📞 Support

If you encounter issues:
1. Check the error messages in console output
2. Verify GPU memory availability
3. Ensure all dependencies are correctly installed
4. Check CUDA/PyTorch compatibility

Happy training! 🚀 