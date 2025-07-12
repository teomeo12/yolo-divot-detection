import time
import queue
import threading
import numpy as np
import cv2
import supervision as sv
import os
from ultralytics import YOLO
import torch

def process_video(video_path, model_path, output_path,
                  resize_factor=0.5, process_every_n_frames=2,
                  use_async_capture=False):
    # 1) Load and prepare the model
    model = YOLO(model_path)         # returns a YOLO object
    model.fuse()                     # fuse conv+BN layers, in-place
    model = model.half().to('cuda')  # then FP16 + move to GPU

    # 2) Set up annotators
    box_annotator   = sv.BoxAnnotator()
    mask_annotator  = sv.MaskAnnotator(color_lookup=sv.ColorLookup.CLASS, opacity=0.5)
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT, text_padding=3)
    
    # 3) Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return
    
    # 4) Video properties and output setup
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    nw, nh = int(w*resize_factor), int(h*resize_factor)
    print(f"Resizing {w}x{h} → {nw}x{nh}, processing every {process_every_n_frames} frames")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps/process_every_n_frames, (nw, nh))
    
    # 5) Optional async capture
    frame_queue = queue.Queue(maxsize=2)
    def reader():
        while True:
            ret, frm = cap.read()
            if not ret: break
            if not frame_queue.full(): frame_queue.put(frm)
    if use_async_capture:
        threading.Thread(target=reader, daemon=True).start()
    
    # 6) Main loop
    frame_count = 0
    try:
        while True:
            # grab
            if use_async_capture:
                if frame_queue.empty():
                    continue
                frame = frame_queue.get()
            else:
                ret, frame = cap.read()
                if not ret:
                    break
            
            frame_count += 1
            if (frame_count - 1) % process_every_n_frames != 0:
                continue
            
            # preprocess
            t0 = time.time()
            img = cv2.resize(frame, (nw, nh))
            t_pre = (time.time() - t0) * 1000
            
            # inference (stream=True overlaps data transfer + compute)
            t1 = time.time()
            results = model(img, stream=True, imgsz=(nh, nw), half=True)
            res     = next(results)  # batch size = 1
            t_inf   = (time.time() - t1) * 1000
            
            # postprocess + annotate
            t2 = time.time()
            dets = sv.Detections.from_ultralytics(res)
            ann  = img.copy()
            ann  = mask_annotator.annotate(scene=ann, detections=dets)
            ann  = box_annotator.annotate(scene=ann, detections=dets)
            ann  = label_annotator.annotate(scene=ann, detections=dets)
            t_post = (time.time() - t2) * 1000
            
            # output
            out.write(ann)
            cv2.imshow('Divot Detection', ann)
            if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                print("Exiting early…")
                break
            
            print(f"Frame {frame_count}:  preprocess {t_pre:.1f}ms  infer {t_inf:.1f}ms  post {t_post:.1f}ms")
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Processed saved to: {output_path}")

def main():
    # GPU info
    print(f"PyTorch {torch.__version__}  CUDA:{torch.version.cuda}  GPU OK? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print("Device:", torch.cuda.get_device_name(0))
    
    workspace = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models = {'yolo11s_1600_100ep': '1600s_aug_100ep.pt'}
    videos = os.path.join(workspace, 'videos')
    
    for folder, mfile in models.items():
        model_path = os.path.join(workspace, folder, mfile)
        if not os.path.exists(model_path):
            print(f"Missing {model_path}, skipping")
            continue
        
        outdir = os.path.join(workspace, folder, 'processed_videos')
        os.makedirs(outdir, exist_ok=True)
        
        for vid in sorted(os.listdir(videos)):
            if not vid.lower().endswith(('.mp4','.avi','.mov')):
                continue
            print(f"\n=== Processing {vid} with {folder} ===")
            inp  = os.path.join(videos, vid)
            outp = os.path.join(outdir, vid.rsplit('.',1)[0] + '_processed.mp4')
            process_video(
                inp, model_path, outp,
                resize_factor=0.5,
                process_every_n_frames=2,
                use_async_capture=True
            )

if __name__ == "__main__":
    main()
