import depthai as dai
import cv2

def main(): 
    # Create a pipeline
    pipeline = dai.Pipeline()

    # Create a ColorCamera node
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setPreviewSize(640, 480)  # Set resolution
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    # Create an XLinkOut node to stream data
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    camRgb.preview.link(xoutRgb.input)

    # Connect to device and start the pipeline
    try:
        with dai.Device(pipeline) as device:
            print("OAK Camera connected successfully!")
            print("Press 'q' to exit")
            
            qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            
            while True:
                inRgb = qRgb.get()  # Get frame from camera
                frame = inRgb.getCvFrame()
                
                if frame is not None:
                    # Add some info text on the frame
                    cv2.putText(frame, "OAK Camera Test - Press 'q' to exit", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Resolution: {frame.shape[1]}x{frame.shape[0]}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Display the frame
                    cv2.imshow('OAK Camera Test', frame)
                
                # Exit on 'q' key press
                if cv2.waitKey(1) == ord('q'):
                    print("Exiting...")
                    break
                    
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your OAK camera is connected properly.")
    
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 