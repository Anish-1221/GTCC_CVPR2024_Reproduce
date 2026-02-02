"""
Generate whitelist with GTCC's test split (35% of 325 = 114 videos)
Uses STRATIFIED RANDOM SAMPLING to ensure representation from all video types.

Video categories in EgoProceL dataset:
- Numbered (0003-0020): MECCANO dataset
- Tent (XX.tent.*): EPIC-tent dataset
- Head (Head_XX): EGTEA Gaze+ dataset
- S## (S##_Task): CMU-MMAC dataset (Brownie, Eggs, Pizza, Salad, Sandwich)
- OP## (OP##-R##-Recipe): CMU-MMAC optional protocol
- P## (P##-R##-Recipe): CMU-MMAC participant videos
"""
import os
import json
import random
import re
from collections import defaultdict

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Paths
GTCC_ALIGNED_FEAT_DIR = '/vision/anishn/GTCC_Data_Processed_1fps/egoprocel/aligned_features'
OUTPUT_WHITELIST = '/vision/anishn/GTCC_CVPR2024/evaluation_video_whitelist.json'

# GTCC's train/test split
TRAIN_SPLIT = 0.65  # 65% train, 35% test
TEST_SPLIT = 1 - TRAIN_SPLIT  # 35% test

def categorize_video(video_name):
    """Categorize video into one of the dataset groups."""
    # Numbered videos (MECCANO): 0003, 0004, etc.
    if re.match(r'^\d{4}$', video_name):
        return 'meccano'

    # Tent videos (EPIC-tent): XX.tent.XXXXXX.gopro
    if '.tent.' in video_name:
        return 'epic_tent'

    # Head videos (EGTEA): Head_XX
    if video_name.startswith('Head_'):
        return 'egtea'

    # S## videos (CMU-MMAC main): S07_Brownie, S08_Eggs, etc.
    if re.match(r'^S\d+_', video_name):
        return 'cmu_mmac_s'

    # OP## videos (CMU-MMAC optional protocol): OP01-R01-PastaSalad
    if re.match(r'^OP\d+-', video_name):
        return 'cmu_mmac_op'

    # P## videos (CMU-MMAC participants): P01-R01-PastaSalad
    if re.match(r'^P\d+-', video_name):
        return 'cmu_mmac_p'

    return 'other'

print("="*80)
print("CREATING TEST SET WHITELIST (35% of videos)")
print("Using STRATIFIED RANDOM SAMPLING for balanced representation")
print("="*80)

# Get all videos that GTCC has aligned features for
feat_files = sorted([f for f in os.listdir(GTCC_ALIGNED_FEAT_DIR) if f.endswith('.npy')])
all_video_names = [os.path.splitext(f)[0] for f in feat_files]

# Categorize videos
video_by_category = defaultdict(list)
for video_name in all_video_names:
    category = categorize_video(video_name)
    video_by_category[category].append(video_name)

print(f"\n[1/4] Video categories found:")
total_videos = len(all_video_names)
for category, videos in sorted(video_by_category.items()):
    print(f"  {category:15s}: {len(videos):3d} videos ({100*len(videos)/total_videos:.1f}%)")

# Stratified sampling: take 35% from each category
print(f"\n[2/4] Stratified sampling (35% from each category):")
test_videos = []
train_videos = []

for category, videos in sorted(video_by_category.items()):
    # Shuffle videos within category
    shuffled = videos.copy()
    random.shuffle(shuffled)

    # Calculate split point for this category
    n_test = max(1, round(len(videos) * TEST_SPLIT))  # At least 1 video per category

    category_test = shuffled[:n_test]
    category_train = shuffled[n_test:]

    test_videos.extend(category_test)
    train_videos.extend(category_train)

    print(f"  {category:15s}: {len(category_test):3d} test / {len(category_train):3d} train")

# Shuffle the final test set for good measure
random.shuffle(test_videos)

print(f"\n[3/4] Final split:")
print(f"  Total videos:     {total_videos}")
print(f"  Train (65%):      {len(train_videos)} videos ({100*len(train_videos)/total_videos:.1f}%)")
print(f"  Test (35%):       {len(test_videos)} videos ({100*len(test_videos)/total_videos:.1f}%)")

# Verify category distribution in test set
print(f"\n[4/4] Test set category distribution:")
test_by_category = defaultdict(list)
for video_name in test_videos:
    category = categorize_video(video_name)
    test_by_category[category].append(video_name)

for category, videos in sorted(test_by_category.items()):
    orig_count = len(video_by_category[category])
    print(f"  {category:15s}: {len(videos):3d} videos (from {orig_count} total)")

print(f"\nTest set sample (first 10):")
for v in sorted(test_videos)[:10]:
    print(f"  - {v}")

# Save test split as whitelist
whitelist_data = {
    'description': 'GTCC test split with stratified random sampling for balanced representation',
    'train_split': TRAIN_SPLIT,
    'test_split': TEST_SPLIT,
    'random_seed': RANDOM_SEED,
    'num_total': total_videos,
    'num_train': len(train_videos),
    'num_test': len(test_videos),
    'category_counts': {cat: len(vids) for cat, vids in sorted(test_by_category.items())},
    'video_names': sorted(test_videos)  # Sort for easier reading
}

with open(OUTPUT_WHITELIST, 'w') as f:
    json.dump(whitelist_data, f, indent=2)

print(f"\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Whitelist saved to: {OUTPUT_WHITELIST}")
print(f"Whitelist contains: {len(test_videos)} test videos (35%)")
print(f"Categories represented: {len(test_by_category)}")
print(f"Random seed used: {RANDOM_SEED}")
print("="*80)