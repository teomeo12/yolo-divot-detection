import depthai as dai
import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO

def main():
    # Initialize YOLO model
    # IMPORTANT: Update this path to your local model path
   
    model_path = "/home/teo/Desktop/yolo-divot-detection/yolo11n_1600_100ep/yolo11n_1600_100ep.pt" 
    model = YOLO(model_path)
    
    # Initialize annotators for visualization
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.CLASS, opacity=0.5)
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT, text_padding=3)
    
    # Create a DepthAI pipeline
    pipeline = dai.Pipeline()

    # Configure ColorCamera
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(640, 480)
    cam_rgb.setInterleaved(False)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)

    # Create XLinkOut for RGB frames
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    # Start the pipeline
    with dai.Device(pipeline) as device:
        # Get the output queue
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        
        print("Camera initialized. Press 'q' to exit.")

        while True:
            # Get RGB frame from the queue
            in_rgb = q_rgb.tryGet()
            
            if in_rgb is not None:
                # Get the frame data
                color_image = in_rgb.getCvFrame()
                
                # Run YOLO detection on the frame
                results = model(color_image)
                
                # Convert YOLO results to supervision Detections object
                detections = sv.Detections.from_ultralytics(results[0])
                
                # Create a copy of the image to draw on
                annotated_image = color_image.copy()
                
                # Annotate the image with segmentation masks
                annotated_image = mask_annotator.annotate(
                    scene=annotated_image, 
                    detections=detections
                )
                
                # Annotate the image with bounding boxes
                annotated_image = box_annotator.annotate(
                    scene=annotated_image, 
                    detections=detections
                )
                
                # Annotate the image with labels
                annotated_image = label_annotator.annotate(
                    scene=annotated_image, 
                    detections=detections
                )
                
                # Display the annotated image
                cv2.imshow('OAK-D Divot Detection', annotated_image)
            
            # Check for 'q' key press to exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("Exiting program...")
                break
                
    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 