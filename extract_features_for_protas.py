"""
Extract ResNet50 layer3 features (1024x14x14) at 1fps for all ProTAS videos.
These features will be used as input to GTCC to generate aligned features.

Output: /vision/anishn/GTCC_Data_Processed_1fps/GTCC_1fps_protas/frames/

Usage:
    python extract_features_for_protas.py [--gpu GPU_ID]
"""

import os
import argparse

# Parse arguments before importing torch to set GPU
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU ID to use (default: 1)')
args = parser.parse_args()

# Set GPU before importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import cv2
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import glob

# --- CONFIGURATION ---
RAW_DATA_ROOT = "/vision/anishn/Egocentric/Datasets"
PROTAS_DATA_ROOT = "/vision/anishn/ProTAS/data_1fps"
OUTPUT_ROOT = "/vision/anishn/GTCC_Data_Processed_1fps/GTCC_1fps_protas/frames"
TARGET_FPS = 1
BATCH_SIZE = 64

# Video path mappings for each dataset
VIDEO_PATHS = {
    'CMU_Kitchens': f"{RAW_DATA_ROOT}/CMU_Kitchens",
    'EGTEA-Gaze+': f"{RAW_DATA_ROOT}/EGTEA-Gaze+",
    'EpicTent': f"{RAW_DATA_ROOT}/EpicTent/2ite3tu1u53n42hjfh3886sa86/data",
    'MECCANO': f"{RAW_DATA_ROOT}/MECCANO/MECCANO_RGB_Videos",
    'PC_assembly': f"{RAW_DATA_ROOT}/PC_assembly",
    'PC_disassembly': f"{RAW_DATA_ROOT}/PC_disassembly",
}


def get_video_path(video_name):
    """
    Map ProTAS video name to actual video file path.

    Video naming conventions:
    - CMU (subset1_S): S07_Brownie_7150991-1431 -> CMU_Kitchens/Brownie/S07_Brownie_Video/S07_Brownie_7150991-1431.avi
    - EGTEA (subset2): OP01-R01-PastaSalad -> EGTEA-Gaze+/OP01-R01-PastaSalad.mp4
    - EpicTent (subset3): 01.tent.090617.gopro -> EpicTent/.../data/01/01.tent.090617.gopro.MP4
    - MECCANO (subset4): 0003 -> MECCANO/MECCANO_RGB_Videos/0003.mp4
    - PC_assembly (subset5): Head_11 -> PC_assembly/Head_11.mp4
    """

    # CMU Kitchens (S##_Task_CameraID format)
    if video_name.startswith('S') and '_' in video_name:
        parts = video_name.split('_')
        if len(parts) >= 2:
            subject = parts[0]  # e.g., S07
            task = parts[1]     # e.g., Brownie
            video_folder = f"{subject}_{task}_Video"
            video_path = os.path.join(VIDEO_PATHS['CMU_Kitchens'], task, video_folder, f"{video_name}.avi")
            if os.path.exists(video_path):
                return video_path

    # EGTEA-Gaze+ (OP##-R##-Task format)
    if video_name.startswith('OP') or video_name.startswith('P'):
        video_path = os.path.join(VIDEO_PATHS['EGTEA-Gaze+'], f"{video_name}.mp4")
        if os.path.exists(video_path):
            return video_path

    # EpicTent (##.tent.date.gopro format)
    if '.tent.' in video_name:
        # Extract the folder number from video name (e.g., "01" from "01.tent.090617.gopro")
        folder_num = video_name.split('.')[0]
        # Try both .MP4 and .mp4 extensions
        for ext in ['.MP4', '.mp4']:
            video_path = os.path.join(VIDEO_PATHS['EpicTent'], folder_num, f"{video_name}{ext}")
            if os.path.exists(video_path):
                return video_path

    # MECCANO (0001, 0002, etc.)
    if video_name.isdigit() or (len(video_name) == 4 and video_name.startswith('0')):
        video_path = os.path.join(VIDEO_PATHS['MECCANO'], f"{video_name}.mp4")
        if os.path.exists(video_path):
            return video_path

    # PC_assembly / PC_disassembly (Head_## format)
    if video_name.startswith('Head_'):
        # Try PC_assembly first
        video_path = os.path.join(VIDEO_PATHS['PC_assembly'], f"{video_name}.mp4")
        if os.path.exists(video_path):
            return video_path
        # Try PC_disassembly
        video_path = os.path.join(VIDEO_PATHS['PC_disassembly'], f"{video_name}.mp4")
        if os.path.exists(video_path):
            return video_path

    return None


def get_all_protas_videos():
    """Get all video names from ProTAS data directories."""
    videos = set()
    subsets = [
        'egoprocel_subset1_S',
        'egoprocel_subset2_OP_P',
        'egoprocel_subset3_tent',
        'egoprocel_subset4_numbers',
        'egoprocel_subset5_head'
    ]

    for subset in subsets:
        features_dir = os.path.join(PROTAS_DATA_ROOT, subset, 'features')
        if os.path.exists(features_dir):
            for f in os.listdir(features_dir):
                if f.endswith('.npy'):
                    video_name = f[:-4]  # Remove .npy extension
                    videos.add(video_name)

    return sorted(list(videos))


def extract_frames_at_fps(video_path, target_fps):
    """Decodes video and samples frames at the target FPS."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30  # Default fallback

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


def main():
    # Setup device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = models.resnet50(weights='IMAGENET1K_V1')
    # Extract from layer3 to get (1024, 14, 14) features
    model = torch.nn.Sequential(*(list(model.children())[:-3]))
    model = model.to(device)
    model.eval()
    print(f"Model output shape will be: (batch, 1024, 14, 14)")

    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create output directory
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Get all ProTAS video names
    all_videos = get_all_protas_videos()
    print(f"Found {len(all_videos)} total ProTAS videos")

    # Check which videos already have features
    existing = set(f[:-4] for f in os.listdir(OUTPUT_ROOT) if f.endswith('.npy'))
    videos_to_process = [v for v in all_videos if v not in existing]
    print(f"Already extracted: {len(existing)}")
    print(f"To process: {len(videos_to_process)}")

    # Track statistics
    successful = 0
    failed = 0
    missing_videos = []

    with torch.no_grad():
        for video_name in tqdm(videos_to_process, desc="Extracting features"):
            output_file = os.path.join(OUTPUT_ROOT, f"{video_name}.npy")

            # Skip if already exists
            if os.path.exists(output_file):
                continue

            # Get video path
            video_path = get_video_path(video_name)
            if video_path is None:
                missing_videos.append(video_name)
                failed += 1
                continue

            # Extract frames
            frames = extract_frames_at_fps(video_path, TARGET_FPS)
            if not frames or len(frames) == 0:
                print(f"\nNo frames extracted from: {video_name}")
                failed += 1
                continue

            # Preprocess and batch
            img_tensors = torch.stack([preprocess(f) for f in frames])

            # Extract features in batches
            features_list = []
            for i in range(0, len(img_tensors), BATCH_SIZE):
                batch = img_tensors[i:i+BATCH_SIZE].to(device)
                feat = model(batch)  # Returns (B, 1024, 14, 14)
                features_list.append(feat.cpu().numpy())

            # Save features
            if features_list:
                video_features = np.concatenate(features_list, axis=0)
                np.save(output_file, video_features.astype(np.float16))
                successful += 1

    # Summary
    print(f"\n{'='*80}")
    print(f"Feature extraction complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {OUTPUT_ROOT}")

    if missing_videos:
        print(f"\nMissing videos ({len(missing_videos)}):")
        for v in missing_videos[:20]:
            print(f"  - {v}")
        if len(missing_videos) > 20:
            print(f"  ... and {len(missing_videos) - 20} more")

    # Verify output count
    final_count = len([f for f in os.listdir(OUTPUT_ROOT) if f.endswith('.npy')])
    print(f"\nTotal features in output: {final_count}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
