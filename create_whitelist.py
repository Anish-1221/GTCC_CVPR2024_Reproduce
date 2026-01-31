"""
Generate whitelist with GTCC's test split (35% of 325 = 114 videos)
"""
import os
import json

# Paths
GTCC_ALIGNED_FEAT_DIR = '/vision/anishn/GTCC_Data_Processed_1fps/egoprocel/aligned_features'
OUTPUT_WHITELIST = '/vision/anishn/GTCC_CVPR2024/evaluation_video_whitelist.json'

# GTCC's train/test split
TRAIN_SPLIT = 0.65  # 65% train, 35% test

print("="*80)
print("CREATING TEST SET WHITELIST (35% of videos)")
print("="*80)

# Get all videos that GTCC has aligned features for
feat_files = sorted([f for f in os.listdir(GTCC_ALIGNED_FEAT_DIR) if f.endswith('.npy')])
all_video_names = sorted([os.path.splitext(f)[0] for f in feat_files])

# Calculate split point
total_videos = len(all_video_names)
split_point = int(total_videos * TRAIN_SPLIT)

# Get test split (last 35%)
train_videos = all_video_names[:split_point]
test_videos = all_video_names[split_point:]

print(f"\n[1/2] Video split:")
print(f"  Total videos:     {total_videos}")
print(f"  Train (65%):      {len(train_videos)} videos")
print(f"  Test (35%):       {len(test_videos)} videos")
print(f"\nTest set first 5: {test_videos[:5]}")
print(f"Test set last 5:  {test_videos[-5:]}")

# Save test split as whitelist
whitelist_data = {
    'description': 'GTCC test split for consistent evaluation',
    'train_split': TRAIN_SPLIT,
    'num_total': total_videos,
    'num_train': len(train_videos),
    'num_test': len(test_videos),
    'video_names': test_videos
}

with open(OUTPUT_WHITELIST, 'w') as f:
    json.dump(whitelist_data, f, indent=2)

print(f"\n[2/2] Saved whitelist to: {OUTPUT_WHITELIST}")
print("="*80)
print("SUMMARY")
print("="*80)
print(f"Whitelist contains: {len(test_videos)} test videos (35%)")
print(f"Both GTCC and ProTAS will be evaluated on these {len(test_videos)} videos")
print("="*80)