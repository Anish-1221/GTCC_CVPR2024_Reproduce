"""
Naive baseline evaluation for action-level OGPE.
Predicts constant values (0, 0.25, 0.5, 0.75, 1.0) or random [0,1] for all action frames,
then computes MAE against ground truth (get_trueprogress_per_action).

No model needed — uses ground truth .txt files from ProTAS and test splits from data_splits.json.

Usage:
    python eval_naive_baselines.py
"""

import os
import json
import random
import numpy as np
import pandas as pd
import torch

from utils.tensorops import get_trueprogress_per_action

# Paths
PROTAS_BASE = '/vision/anishn/ProTAS/data_1fps'
SUBSETS = [
    'egoprocel_subset1_S', 'egoprocel_subset2_OP_P', 'egoprocel_subset3_tent',
    'egoprocel_subset4_numbers', 'egoprocel_subset5_head'
]
SPLITS_PATH = '/vision/anishn/GTCC_CVPR2024/data_splits.json'
DSET_JSON_PATH = '/vision/anishn/GTCC_CVPR2024/dset_jsons/egoprocel.json'
OUTPUT_BASE = '/vision/anishn/GTCC_CVPR2024/output_baselines'

CONSTANT_BASELINES = [0.0, 0.25, 0.5, 0.75, 1.0]
NUM_RANDOM_SEEDS = 10


def parse_groundtruth_to_segments(gt_path):
    """Read ground truth .txt and parse into segments. Same as evaluation_action_level.py."""
    with open(gt_path, 'r') as f:
        action_names = [line.strip() for line in f if line.strip()]

    segments = []
    if not action_names:
        return segments, 0

    current_action = action_names[0]
    start_frame = 0

    for i in range(1, len(action_names)):
        if action_names[i] != current_action:
            segments.append({
                'name': current_action,
                'start': start_frame,
                'end': i - 1
            })
            current_action = action_names[i]
            start_frame = i

    segments.append({
        'name': current_action,
        'start': start_frame,
        'end': len(action_names) - 1
    })

    return segments, len(action_names)


def find_gt_path(video_name):
    """Search all 5 ProTAS subsets for this video's ground truth."""
    for subset in SUBSETS:
        path = f'{PROTAS_BASE}/{subset}/groundTruth/{video_name}.txt'
        if os.path.exists(path):
            return path
    return None


def compute_ogpe_for_video(segments, total_frames, pred_progress):
    """Compute action-level OGPE for a single video given predicted progress tensor."""
    tdict = {
        'step': [seg['name'] for seg in segments],
        'start_frame': [seg['start'] for seg in segments],
        'end_frame': [min(seg['end'], total_frames - 1) for seg in segments],
    }

    true_progress = get_trueprogress_per_action(tdict)

    # Build action mask (exclude SIL/background)
    action_mask = torch.zeros(len(true_progress), dtype=torch.bool)
    for seg in segments:
        if seg['name'] not in ['SIL', 'background']:
            seg_start = max(0, min(seg['start'], len(true_progress) - 1))
            seg_end = max(0, min(seg['end'], len(true_progress) - 1))
            action_mask[seg_start:seg_end + 1] = True

    if not action_mask.any():
        return None

    errors = torch.abs(true_progress - pred_progress[:len(true_progress)])
    return errors[action_mask].mean().item()


def run_constant_baseline(test_videos, constant_value):
    """Run a constant prediction baseline across all test videos."""
    gpe_list = []
    for task, videos in test_videos.items():
        for video_name in videos:
            gt_path = find_gt_path(video_name)
            if gt_path is None:
                print(f"[WARNING] No ground truth for {video_name}, skipping")
                continue

            segments, total_frames = parse_groundtruth_to_segments(gt_path)
            if total_frames == 0:
                continue

            pred_progress = torch.full((total_frames,), constant_value)
            ogpe = compute_ogpe_for_video(segments, total_frames, pred_progress)
            if ogpe is not None:
                gpe_list.append({'task': task, 'video': video_name, 'ogpe': ogpe})

    return gpe_list


def run_random_baseline(test_videos, seed):
    """Run a random [0,1] prediction baseline across all test videos."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    gpe_list = []
    for task, videos in test_videos.items():
        for video_name in videos:
            gt_path = find_gt_path(video_name)
            if gt_path is None:
                print(f"[WARNING] No ground truth for {video_name}, skipping")
                continue

            segments, total_frames = parse_groundtruth_to_segments(gt_path)
            if total_frames == 0:
                continue

            pred_progress = torch.rand(total_frames)
            ogpe = compute_ogpe_for_video(segments, total_frames, pred_progress)
            if ogpe is not None:
                gpe_list.append({'task': task, 'video': video_name, 'ogpe': ogpe})

    return gpe_list


def save_results(results, output_folder):
    """Save per-task OGPE summary to ogpe.csv."""
    os.makedirs(output_folder, exist_ok=True)

    df = pd.DataFrame(results)
    # Per-task summary
    task_summary = df.groupby('task').agg(
        ogpe=('ogpe', 'mean'),
        num_videos=('ogpe', 'count')
    ).reset_index()

    # Add overall mean row
    mean_row = pd.DataFrame([{
        'task': 'MEAN',
        'ogpe': df['ogpe'].mean(),
        'num_videos': len(df)
    }])
    task_summary = pd.concat([task_summary, mean_row], ignore_index=True)

    csv_path = os.path.join(output_folder, 'ogpe.csv')
    task_summary.to_csv(csv_path, index=False)
    return task_summary


def get_test_videos():
    """Load test video names per task from data_splits.json."""
    with open(SPLITS_PATH, 'r') as f:
        splits_data = json.load(f)

    test_videos = {}
    for task, split in splits_data['splits'].items():
        test_videos[task] = split['test']

    return test_videos


if __name__ == '__main__':
    test_videos = get_test_videos()
    total_test = sum(len(v) for v in test_videos.values())
    print(f"Loaded {total_test} test videos across {len(test_videos)} tasks")

    # Constant baselines
    for const_val in CONSTANT_BASELINES:
        results = run_constant_baseline(test_videos, const_val)
        folder = os.path.join(OUTPUT_BASE, f'constant_{const_val}')
        summary = save_results(results, folder)
        mean_ogpe = summary[summary['task'] == 'MEAN']['ogpe'].values[0]
        print(f"Constant {const_val:4.2f}  =>  OGPE = {mean_ogpe:.4f}  ({len(results)} videos)")

    # Random baseline (multiple seeds)
    all_random_means = []
    all_random_results = []
    for seed in range(NUM_RANDOM_SEEDS):
        results = run_random_baseline(test_videos, seed)
        mean_ogpe = np.mean([r['ogpe'] for r in results])
        all_random_means.append(mean_ogpe)
        all_random_results.append(results)

    # Save the first seed's results as the representative CSV
    folder = os.path.join(OUTPUT_BASE, 'random')
    summary = save_results(all_random_results[0], folder)

    # Also save a summary with mean ± std across seeds
    random_mean = np.mean(all_random_means)
    random_std = np.std(all_random_means)
    print(f"Random [0,1]  =>  OGPE = {random_mean:.4f} +/- {random_std:.4f}  ({NUM_RANDOM_SEEDS} seeds)")

    # Save seed-level summary
    seed_df = pd.DataFrame({'seed': range(NUM_RANDOM_SEEDS), 'ogpe': all_random_means})
    seed_df.to_csv(os.path.join(OUTPUT_BASE, 'random', 'ogpe_seeds.csv'), index=False)

    print(f"\nResults saved to {OUTPUT_BASE}/")
