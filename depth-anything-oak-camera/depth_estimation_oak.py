import depthai as dai
import cv2
import numpy as np

# Create pipeline
pipeline = dai.Pipeline()

# Color camera
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setInterleaved(False)

# Mono cameras
mono_left = pipeline.create(dai.node.MonoCamera)
mono_right = pipeline.create(dai.node.MonoCamera)
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

# Stereo depth
stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)

# Configure stereo depth
stereo.setConfidenceThreshold(200)
stereo.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
stereo.setLeftRightCheck(True)
stereo.setExtendedDisparity(True)
stereo.setSubpixel(True)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

# Link mono cams to stereo
mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

# Outputs
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.video.link(xout_rgb.input)

xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# Start device
with dai.Device(pipeline) as device:
    rgb_queue = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
    depth_queue = device.getOutputQueue(name="depth", maxSize=1, blocking=False)

    calib = device.readCalibration()
    intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B)  # Using left camera intrinsics
    fx = intrinsics[0][0]
    baseline_cm = 7.5

    while True:
        rgb_frame = rgb_queue.get().getCvFrame()
        depth_frame = depth_queue.get().getFrame()

        # Normalize and colorize depth
        depth_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_JET)

        # Show frames
        cv2.imshow("RGB", rgb_frame)
        cv2.imshow("Depth", depth_vis)

        # Print depth at center pixel
        h, w = depth_frame.shape
        center_depth = depth_frame[h//2, w//2]
        if center_depth > 0:
            depth_cm = (fx * baseline_cm) / center_depth
            print(f"Depth @ center: {depth_cm:.2f} cm")

        if cv2.waitKey(1) == ord('q'):
            break
