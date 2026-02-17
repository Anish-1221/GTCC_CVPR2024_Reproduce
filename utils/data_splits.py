"""
Stratified data split generator for GTCC training.

Generates deterministic, stratified train/val/test splits (75/10/15)
with each task category represented proportionally in each split.
"""
import json
import random
from typing import Dict, List


def generate_stratified_splits(
    data_structure: Dict,
    train_ratio: float = 0.75,
    val_ratio: float = 0.10,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, Dict[str, List[str]]]:
    """
    Generate stratified train/val/test splits for each task category.

    Args:
        data_structure: Dict with task names as keys, each containing 'handles' list
        train_ratio: Fraction for training (default 0.75)
        val_ratio: Fraction for validation (default 0.10)
        test_ratio: Fraction for test (default 0.15)
        seed: Random seed for reproducibility

    Returns:
        Dict with structure:
        {
            "task_name": {
                "train": [handle1, handle2, ...],
                "val": [handle3, ...],
                "test": [handle4, ...]
            },
            ...
        }
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    random.seed(seed)
    splits = {}

    for task_name, task_data in data_structure.items():
        handles = task_data['handles'].copy()
        random.shuffle(handles)

        n = len(handles)

        # Ensure minimum 1 video per split where possible
        if n >= 3:
            n_train = max(1, round(n * train_ratio))
            n_val = max(1, round(n * val_ratio))
            n_test = max(1, n - n_train - n_val)

            # Adjust if we over-allocated due to rounding
            while n_train + n_val + n_test > n:
                if n_train > 1:
                    n_train -= 1
                elif n_val > 1:
                    n_val -= 1
        elif n == 2:
            # Only 2 videos: 1 train, 0 val, 1 test
            n_train, n_val, n_test = 1, 0, 1
        else:
            # Only 1 video: put in train
            n_train, n_val, n_test = 1, 0, 0

        splits[task_name] = {
            'train': handles[:n_train],
            'val': handles[n_train:n_train + n_val],
            'test': handles[n_train + n_val:]
        }

    return splits


def save_splits_to_json(splits: Dict, filepath: str):
    """
    Save splits dictionary to JSON file with metadata.

    Args:
        splits: The splits dictionary from generate_stratified_splits
        filepath: Path to save the JSON file
    """
    # Compute statistics
    all_train = [h for t in splits.values() for h in t['train']]
    all_val = [h for t in splits.values() for h in t['val']]
    all_test = [h for t in splits.values() for h in t['test']]

    # Per-category counts
    category_counts = {}
    for task_name, split_data in splits.items():
        category_counts[task_name] = {
            'train': len(split_data['train']),
            'val': len(split_data['val']),
            'test': len(split_data['test']),
            'total': len(split_data['train']) + len(split_data['val']) + len(split_data['test'])
        }

    output = {
        'description': 'Stratified train/val/test splits for GTCC training',
        'ratios': {'train': 0.75, 'val': 0.10, 'test': 0.15},
        'seed': 42,
        'counts': {
            'total': len(all_train) + len(all_val) + len(all_test),
            'train': len(all_train),
            'val': len(all_val),
            'test': len(all_test)
        },
        'category_counts': category_counts,
        'splits': splits
    }

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)


def load_splits_from_json(filepath: str) -> Dict:
    """
    Load splits dictionary from JSON file.

    Args:
        filepath: Path to the JSON file

    Returns:
        The splits dictionary with structure:
        {
            "task_name": {
                "train": [...],
                "val": [...],
                "test": [...]
            }, ...
        }
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['splits']


def get_all_videos_for_split(splits: Dict, split_type: str) -> List[str]:
    """
    Get all video handles for a specific split type across all tasks.

    Args:
        splits: The splits dictionary
        split_type: One of 'train', 'val', 'test'

    Returns:
        List of all video handles in that split
    """
    assert split_type in ['train', 'val', 'test'], f"Invalid split_type: {split_type}"
    return [h for task_data in splits.values() for h in task_data[split_type]]
