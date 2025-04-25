import torch
import os
import argparse
from ultralytics import YOLO
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Convert YOLO model from PyTorch to ONNX format')
    parser.add_argument('--model', type=str, default='373s_seg_noaug_50ep.pt', 
                        help='Path to the PyTorch model')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], 
                        help='Image size for ONNX export (height, width)')
    parser.add_argument('--batch-size', type=int, default=1, 
                        help='Batch size for ONNX export')
    parser.add_argument('--opset', type=int, default=12, 
                        help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', 
                        help='Simplify ONNX model (requires onnx-simplifier)')
    parser.add_argument('--output', type=str, default='', 
                        help='Output file name (default: input_name.onnx)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    # Determine output filename if not specified
    if not args.output:
        model_name = Path(args.model).stem
        args.output = f"{model_name}.onnx"
    
    print(f"Converting model to ONNX format...")
    model_path = model.export(format="onnx", opset=args.opset, simplify=args.simplify, 
                             imgsz=args.img_size, batch=args.batch_size)
    
    print(f"Model successfully converted to ONNX format and saved at: {model_path}")
    print("\nTo use the ONNX model, you can replace the PyTorch model path in your code with the ONNX model path.")
    print("Example usage with ONNXRuntime:")
    print("```python")
    print("import onnxruntime as ort")
    print("import numpy as np")
    print("import cv2")
    print("")
    print("# Load ONNX model")
    print(f"session = ort.InferenceSession('{model_path}')")
    print("")
    print("# Prepare input (example)")
    print("image = cv2.imread('image.jpg')")
    print("image = cv2.resize(image, tuple(args.img_size[::-1]))  # width, height")
    print("image = image.transpose(2, 0, 1)  # HWC to CHW")
    print("image = image.astype(np.float32) / 255.0")
    print("image = np.expand_dims(image, 0)  # add batch dimension")
    print("")
    print("# Run inference")
    print("inputs = {session.get_inputs()[0].name: image}")
    print("outputs = session.run(None, inputs)")
    print("```")

if __name__ == '__main__':
    main() 