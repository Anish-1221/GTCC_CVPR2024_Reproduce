#!/usr/bin/env python
"""
Calculate Action Geodesic Means - STORES IN EXPERIMENT FOLDER

Usage:
    python calculate_action_means.py --exp_folder /path/to/model

Input:  {exp_folder}/aligned_features/*.npy
Output: {exp_folder}/action_means.json

Examples:
    # Just specify the experiment folder!
    python calculate_action_means.py --exp_folder output_4fps_val/multi-task-setting_val/V1___GTCC_egoprocel.4fps

    # Output will be at: output_4fps_val/multi-task-setting_val/V1___GTCC_egoprocel.4fps/action_means.json
"""
import argparse
import os
import torch
import json
import numpy as np
from tqdm import tqdm


SUBSETS = [
    'egoprocel_subset1_S',
    'egoprocel_subset2_OP_P',
    'egoprocel_subset3_tent',
    'egoprocel_subset4_numbers',
    'egoprocel_subset5_head'
]


def get_segments_from_txt(file_path):
    """Groups frame-wise labels into start/end segments"""
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        return []

    segments = []
    current_action = lines[0]
    start_frame = 0

    for i in range(1, len(lines)):
        if lines[i] != current_action:
            segments.append({'name': current_action, 'start': start_frame, 'end': i - 1})
            current_action = lines[i]
            start_frame = i

    segments.append({'name': current_action, 'start': start_frame, 'end': len(lines) - 1})
    return segments


def calculate_geodesic_distance(features):
    """Calculate geodesic distance (sum of L2 between consecutive frames)"""
    if features.shape[0] < 2:
        return 0.0
    diffs = torch.norm(features[1:] - features[:-1], p=2, dim=1)
    return torch.sum(diffs).item()


def main():
    parser = argparse.ArgumentParser(
        description='Calculate action geodesic means (stored in experiment folder)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--exp_folder', type=str, required=True,
                        help='Path to experiment folder containing aligned_features/')
    parser.add_argument('--fps', type=str, default=None,
                        help='FPS for ground truth (auto-detected if not specified)')

    args = parser.parse_args()

    # Auto-detect FPS
    fps = args.fps
    if fps is None:
        fps = '4fps' if '4fps' in args.exp_folder else '1fps'

    protas_base = f'/vision/anishn/ProTAS/data_{fps}/'
    aligned_feat_dir = os.path.join(args.exp_folder, 'aligned_features')
    output_json = os.path.join(args.exp_folder, 'action_means.json')  # INSIDE experiment folder!

    print("="*80)
    print("Action Means Calculation (Experiment Folder Storage)")
    print("="*80)
    print(f"Experiment folder: {args.exp_folder}")
    print(f"FPS: {fps}")
    print(f"Aligned features: {aligned_feat_dir}")
    print(f"Output: {output_json}")

    if not os.path.exists(aligned_feat_dir):
        print(f"\n[ERROR] Aligned features directory not found: {aligned_feat_dir}")
        print(f"Run first: python generate_aligned_features.py --exp_folder {args.exp_folder}")
        return

    action_distances = {}
    feat_files = sorted([f for f in os.listdir(aligned_feat_dir) if f.endswith('.npy')])
    print(f"\nFound {len(feat_files)} aligned feature files")

    processed, skipped_no_label, skipped_length, total_segments = 0, 0, 0, 0

    for f_name in tqdm(feat_files, desc="Calculating Action Means"):
        video_name = os.path.splitext(f_name)[0]

        # Find ground truth in subsets
        label_path = None
        for s in SUBSETS:
            potential = os.path.join(protas_base, s, 'groundTruth', f"{video_name}.txt")
            if os.path.exists(potential):
                label_path = potential
                break

        if not label_path:
            skipped_no_label += 1
            continue

        # Load aligned features
        aligned_feats = torch.from_numpy(np.load(os.path.join(aligned_feat_dir, f_name))).float()
        T_feat = aligned_feats.shape[0]

        # Parse segments
        segments = get_segments_from_txt(label_path)

        # Verify length match
        with open(label_path, 'r') as f:
            T_label = len([l.strip() for l in f if l.strip()])

        if T_feat != T_label:
            skipped_length += 1
            continue

        # Calculate geodesic distance per segment
        for seg in segments:
            start, end = seg['start'], min(seg['end'] + 1, T_feat)
            if end - start < 2:
                continue

            geo_dist = calculate_geodesic_distance(aligned_feats[start:end])

            if seg['name'] not in action_distances:
                action_distances[seg['name']] = []
            action_distances[seg['name']].append(geo_dist)
            total_segments += 1

        processed += 1

    # Aggregate statistics
    final_stats = {}
    for action_name, dists in sorted(action_distances.items()):
        final_stats[action_name] = {
            "mean": float(np.mean(dists)),
            "std": float(np.std(dists)),
            "min": float(np.min(dists)),
            "max": float(np.max(dists)),
            "count": len(dists)
        }

    # Save
    with open(output_json, 'w') as f:
        json.dump(final_stats, f, indent=4)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Videos processed:      {processed}/{len(feat_files)}")
    print(f"Skipped (no label):    {skipped_no_label}")
    print(f"Skipped (len mismatch):{skipped_length}")
    print(f"Total segments:        {total_segments}")
    print(f"Unique actions:        {len(final_stats)}")
    print(f"\nAction means saved to: {output_json}")
    print(f"Ready for evaluation: python eval.py -f {args.exp_folder} --level action")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
