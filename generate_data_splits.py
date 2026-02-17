#!/usr/bin/env python
"""
One-time script to generate deterministic, stratified data splits.

Generates data_splits.json with 75% train, 10% validation, 15% test splits
stratified by task category (each category represented proportionally).

Usage:
    python generate_data_splits.py

Output:
    Creates /vision/anishn/GTCC_CVPR2024/data_splits.json
"""
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.os_util import get_env_variable
from models.json_dataset import data_json_labels_handles
from utils.data_splits import generate_stratified_splits, save_splits_to_json


def main():
    # Get dataset JSON folder from environment
    dset_json_folder = get_env_variable('JSON_DPATH')

    # Load full data structure
    print("Loading data structure from egoprocel.json...")
    data_structure = data_json_labels_handles(dset_json_folder, dset_name='egoprocel')

    print(f"Found {len(data_structure)} task categories:")
    total_videos = 0
    for task_name, task_data in data_structure.items():
        n_videos = len(task_data['handles'])
        total_videos += n_videos
        print(f"  {task_name}: {n_videos} videos")
    print(f"Total: {total_videos} videos")

    # Generate stratified splits
    print("\nGenerating stratified splits (75% train, 10% val, 15% test)...")
    splits = generate_stratified_splits(
        data_structure,
        train_ratio=0.75,
        val_ratio=0.10,
        test_ratio=0.15,
        seed=42
    )

    # Print split statistics
    print("\nSplit statistics per category:")
    print("-" * 60)
    print(f"{'Category':<35} {'Train':>6} {'Val':>6} {'Test':>6}")
    print("-" * 60)

    total_train, total_val, total_test = 0, 0, 0
    for task_name, split_data in sorted(splits.items()):
        n_train = len(split_data['train'])
        n_val = len(split_data['val'])
        n_test = len(split_data['test'])
        total_train += n_train
        total_val += n_val
        total_test += n_test
        print(f"{task_name:<35} {n_train:>6} {n_val:>6} {n_test:>6}")

    print("-" * 60)
    print(f"{'TOTAL':<35} {total_train:>6} {total_val:>6} {total_test:>6}")
    print(f"{'PERCENTAGE':<35} {100*total_train/total_videos:>5.1f}% {100*total_val/total_videos:>5.1f}% {100*total_test/total_videos:>5.1f}%")

    # Save to JSON
    output_path = '/vision/anishn/GTCC_CVPR2024/data_splits.json'
    print(f"\nSaving splits to {output_path}...")
    save_splits_to_json(splits, output_path)

    print("Done!")

    # Verify no overlap between splits
    print("\nVerifying no overlap between splits...")
    all_train = set(h for t in splits.values() for h in t['train'])
    all_val = set(h for t in splits.values() for h in t['val'])
    all_test = set(h for t in splits.values() for h in t['test'])

    train_val_overlap = all_train & all_val
    train_test_overlap = all_train & all_test
    val_test_overlap = all_val & all_test

    if train_val_overlap:
        print(f"  WARNING: {len(train_val_overlap)} videos in both train and val!")
    else:
        print("  No train/val overlap")

    if train_test_overlap:
        print(f"  WARNING: {len(train_test_overlap)} videos in both train and test!")
    else:
        print("  No train/test overlap")

    if val_test_overlap:
        print(f"  WARNING: {len(val_test_overlap)} videos in both val and test!")
    else:
        print("  No val/test overlap")

    print("\nAll verifications passed!")


if __name__ == '__main__':
    main()
