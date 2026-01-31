# import os
# import cv2
# import torch
# import numpy as np
# from torchvision import models, transforms
# from PIL import Image
# from tqdm import tqdm

# # --- CONFIGURATION ---
# RAW_DATA_ROOT = "/vision/anishn/Egocentric/Datasets"
# OUTPUT_ROOT = "/u/anishn/GTCC_Data_Processed/egoprocel/frames"
# TARGET_FPS = 10
# BATCH_SIZE = 64  # Adjust based on your GPU memory

# # 1. Setup Device and Model
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = models.resnet50(pretrained=True)

# # CRITICAL: Extract from layer3 (before layer4) to get (1024, 14, 14) features
# # children()[:-3] removes: layer4, avgpool, and fc
# model = torch.nn.Sequential(*(list(model.children())[:-3]))
# model = model.to(device)
# model.eval()

# print(f"Model output shape will be: (batch, 1024, 14, 14)")

# # 2. Image Preprocessing (Standard ImageNet normalization)
# preprocess = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# def extract_frames_at_fps(video_path, target_fps):
#     """Decodes video and samples frames at the target FPS (e.g., 10 FPS)."""
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         return None
    
#     video_fps = cap.get(cv2.CAP_PROP_FPS)
#     if video_fps <= 0:
#         video_fps = 30  # Default fallback
    
#     hop = video_fps / target_fps
#     frames = []
#     count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         if int(count % hop) == 0:
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(Image.fromarray(frame_rgb))
#         count += 1
#     cap.release()
#     return frames

# def process_all_videos():
#     os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
#     all_video_paths = []
#     valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    
#     print(f"Searching for videos in {RAW_DATA_ROOT}...")
#     for root, _, files in os.walk(RAW_DATA_ROOT):
#         for file in files:
#             if file.lower().endswith(valid_extensions):
#                 all_video_paths.append(os.path.join(root, file))

#     print(f"Found {len(all_video_paths)} videos. Extracting layer3 features at {TARGET_FPS} FPS...")

#     for v_path in tqdm(all_video_paths, desc="Total Extraction Progress"):
#         handle = os.path.splitext(os.path.basename(v_path))[0]
#         output_file = os.path.join(OUTPUT_ROOT, f"{handle}.npy")
        
#         if os.path.exists(output_file):
#             continue

#         frames = extract_frames_at_fps(v_path, TARGET_FPS)
#         if not frames or len(frames) == 0:
#             continue
        
#         img_tensors = torch.stack([preprocess(f) for f in frames])
        
#         # Batch inference to extract (T, 1024, 14, 14) features from layer3
#         features_list = []
#         with torch.no_grad():
#             for i in range(0, len(img_tensors), BATCH_SIZE):
#                 batch = img_tensors[i:i+BATCH_SIZE].to(device)
#                 feat = model(batch)  # Returns (B, 1024, 14, 14)
#                 features_list.append(feat.cpu().numpy())
        
#         # Save as numpy array with shape (T, 1024, 14, 14)
#         if features_list:
#             video_features = np.concatenate(features_list, axis=0)
#             print(f"Saving {handle}.npy with shape {video_features.shape}")
#             np.save(output_file, video_features)

# if __name__ == "__main__":
#     process_all_videos()

import os
import json
import cv2
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
RAW_DATA_ROOT = "/vision/anishn/Egocentric/Datasets"
OUTPUT_ROOT = '/vision/anishn/GTCC_Data_Processed_1fps/egoprocel/frames'
JSON_PATH = "/vision/anishn/GTCC_CVPR2024/dset_jsons/egoprocel.json"
TARGET_FPS = 1
BATCH_SIZE = 64

# Load JSON to get required handles
with open(JSON_PATH, 'r') as f:
    data_structure = json.load(f)

# Extract all required video handles from JSON
required_handles = set()
for task_name, task_data in data_structure.items():
    required_handles.update(task_data['handles'])

print(f"JSON requires {len(required_handles)} unique videos")

# 1. Setup Device and Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights='IMAGENET1K_V1')  # Fixed deprecation warning
model = torch.nn.Sequential(*(list(model.children())[:-3]))
model = model.to(device)
model.eval()

print(f"Model output shape will be: (batch, 1024, 14, 14)")

# 2. Image Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_frames_at_fps(video_path, target_fps):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30
    
    hop = video_fps / target_fps
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if int(count % hop) == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        count += 1
    cap.release()
    return frames

def process_required_videos():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # Find all video files
    all_video_paths = {}
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    
    print(f"Searching for videos in {RAW_DATA_ROOT}...")
    for root, _, files in os.walk(RAW_DATA_ROOT):
        for file in files:
            if file.lower().endswith(valid_extensions):
                handle = os.path.splitext(file)[0]
                all_video_paths[handle] = os.path.join(root, file)
    
    print(f"Found {len(all_video_paths)} total videos in directory")
    
    # Filter to only required videos
    videos_to_process = []
    missing_videos = []
    
    for handle in required_handles:
        output_file = os.path.join(OUTPUT_ROOT, f"{handle}.npy")
        if os.path.exists(output_file):
            continue  # Already extracted
        
        if handle in all_video_paths:
            videos_to_process.append((handle, all_video_paths[handle]))
        else:
            missing_videos.append(handle)
    
    if missing_videos:
        print(f"WARNING: {len(missing_videos)} videos from JSON not found in {RAW_DATA_ROOT}")
        print(f"First few missing: {missing_videos[:5]}")
    
    print(f"\nExtracting features for {len(videos_to_process)} videos at {TARGET_FPS} FPS...")
    print(f"Estimated storage needed: ~{len(videos_to_process) * 3:.1f} GB")
    
    for handle, v_path in tqdm(videos_to_process, desc="Extraction Progress"):
        output_file = os.path.join(OUTPUT_ROOT, f"{handle}.npy")
        
        frames = extract_frames_at_fps(v_path, TARGET_FPS)
        if not frames or len(frames) == 0:
            print(f"\nSkipping {handle}: No frames extracted")
            continue
        
        img_tensors = torch.stack([preprocess(f) for f in frames])
        
        features_list = []
        with torch.no_grad():
            for i in range(0, len(img_tensors), BATCH_SIZE):
                batch = img_tensors[i:i+BATCH_SIZE].to(device)
                feat = model(batch)
                features_list.append(feat.cpu().numpy())
        
        if features_list:
            video_features = np.concatenate(features_list, axis=0)
            np.save(output_file, video_features.astype(np.float16))
            # Optional: print progress every 10 videos
            if len(os.listdir(OUTPUT_ROOT)) % 10 == 0:
                print(f"\n{len(os.listdir(OUTPUT_ROOT))} files extracted so far")

if __name__ == "__main__":
    process_required_videos()