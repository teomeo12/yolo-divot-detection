import numpy as np
import cv2
import supervision as sv
import os
from ultralytics import YOLO
import torch
from collections import Counter

def evaluate_model_on_images(model_path, image_dir, output_dir, resize_factor=0.5):
    # Initialize YOLO model
    model = YOLO(model_path)
    
    # Initialize annotators
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.CLASS, opacity=0.5)
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT, text_padding=3)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Counter for detected classes
    detection_counts = Counter()
    
    # Process each image in the directory
    for image_file in sorted(os.listdir(image_dir)):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"\nProcessing image: {image_file}")
            
            image_path = os.path.join(image_dir, image_file)
            output_path = os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_processed.jpg")
            
            # Read and resize image
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Warning: Could not read image {image_file}")
                continue
                
            new_width = int(frame.shape[1] * resize_factor)
            new_height = int(frame.shape[0] * resize_factor)
            resized_frame = cv2.resize(frame, (new_width, new_height))
            
            # Run YOLO detection
            results = model(resized_frame)
            detections = sv.Detections.from_ultralytics(results[0])
            
            # Update detection counts
            for class_id in detections.class_id:
                class_name = model.model.names[class_id]
                detection_counts[class_name] += 1
            
            # Annotate the image
            annotated_frame = resized_frame.copy()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
            
            # Save the processed image
            cv2.imwrite(output_path, annotated_frame)
            print(f"Saved processed image to: {output_path}")

    # Print the performance summary
    print("\n--- Model Performance Summary ---")
    if not detection_counts:
        print("No objects were detected in any of the images.")
    else:
        print("Total detections per class:")
        for class_name, count in detection_counts.items():
            print(f"- {class_name}: {count}")
    print("---------------------------------")

def main():
    # Base workspace directory
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Project and model configuration
    project_folder = 'yolo11s_1600_100ep'
    # model_file = '3318s_aug_150ep.pt'
    model_file = '1600s_aug_100ep.pt'
    
    model_path = os.path.join(workspace_dir, project_folder, model_file)
    image_dir = os.path.join(workspace_dir, project_folder, 'test_images')
    output_dir = os.path.join(workspace_dir, project_folder, 'processed_img_test')
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
        
    # Check if image directory exists
    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found at {image_dir}")
        return
        
    evaluate_model_on_images(model_path, image_dir, output_dir, resize_factor=0.5)

if __name__ == "__main__":
    main()
