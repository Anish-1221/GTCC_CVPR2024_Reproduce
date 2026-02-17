import os
import torch
import json
import numpy as np
from tqdm import tqdm

# --- PATH CONFIGURATION (4 FPS VERSION) ---
ALIGNED_FEAT_DIR = '/vision/anishn/GTCC_Data_Processed_4fps/egoprocel/vava_aligned_features'
PROTAS_BASE = '/vision/anishn/ProTAS/data_4fps/'
OUTPUT_JSON = '/vision/anishn/GTCC_CVPR2024/vava_action_means_4fps.json'
SUBSETS = [
    'egoprocel_subset1_S',
    'egoprocel_subset2_OP_P',
    'egoprocel_subset3_tent',
    'egoprocel_subset4_numbers',
    'egoprocel_subset5_head'
]

def get_segments_from_txt(file_path):
    """
    Groups frame-wise labels into start/end segments.
    Returns list of dicts: [{'name': action, 'start': int, 'end': int}, ...]
    """
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    segments = []
    if not lines:
        return segments

    current_action = lines[0]
    start_frame = 0

    for i in range(1, len(lines)):
        if lines[i] != current_action:
            segments.append({
                'name': current_action,
                'start': start_frame,
                'end': i - 1  # Inclusive end
            })
            current_action = lines[i]
            start_frame = i

    # Add final segment
    segments.append({
        'name': current_action,
        'start': start_frame,
        'end': len(lines) - 1  # Inclusive end
    })

    return segments


def calculate_geodesic_distance(features):
    """
    Calculate geodesic distance for a feature sequence.
    Geodesic = sum of L2 distances between consecutive frames.

    Args:
        features: torch.Tensor of shape [T, D]

    Returns:
        float: total geodesic distance
    """
    if features.shape[0] < 2:
        return 0.0

    # Calculate L2 distance between consecutive frames
    diffs = torch.norm(features[1:] - features[:-1], p=2, dim=1)
    total_dist = torch.sum(diffs).item()

    return total_dist


def main():
    print("="*80)
    print("VAVA ACTION-WISE GEODESIC MEAN CALCULATION")
    print("="*80)

    action_distances = {}

    # Get list of aligned feature files
    feat_files = sorted([f for f in os.listdir(ALIGNED_FEAT_DIR) if f.endswith('.npy')])
    print(f"\n[1/4] Found {len(feat_files)} aligned feature files")
    print(f"First 3 files: {feat_files[:3]}")

    # Counters for debugging
    processed = 0
    skipped_no_label = 0
    skipped_length_mismatch = 0
    total_segments = 0
    skipped_short_segments = 0

    # Track which subset each video is found in
    subset_distribution = {s: 0 for s in SUBSETS}

    print(f"\n[2/4] Processing files...")

    for file_idx, f_name in enumerate(tqdm(feat_files, desc="Calculating Action Means")):
        video_name = os.path.splitext(f_name)[0]

        # Debug first file in detail
        debug_this_file = (file_idx == 0)

        if debug_this_file:
            print(f"\n[DEBUG] Detailed analysis of first file: {video_name}")
            print(f"[DEBUG]   Searching for ground truth in all {len(SUBSETS)} subsets...")

        # 1. CRITICAL: Search ALL subsets to find the ground truth file
        # Each feature file's ground truth could be in ANY of the 5 subsets
        label_path = None
        found_subset = None
        for s in SUBSETS:
            potential = os.path.join(PROTAS_BASE, s, 'groundTruth', f"{video_name}.txt")
            if debug_this_file:
                print(f"[DEBUG]     Checking {s}: {os.path.exists(potential)}")
            if os.path.exists(potential):
                label_path = potential
                found_subset = s
                subset_distribution[s] += 1
                if debug_this_file:
                    print(f"[DEBUG]   ✓ Found label in subset: {s}")
                    print(f"[DEBUG]   Label path: {label_path}")
                break

        if not label_path:
            skipped_no_label += 1
            if debug_this_file:
                print(f"[DEBUG]   ✗ No label found in any of the {len(SUBSETS)} subsets")
                print(f"[DEBUG]   Searched subsets: {SUBSETS}")
            continue

        # 2. Load aligned features
        feat_path = os.path.join(ALIGNED_FEAT_DIR, f_name)
        aligned_feats = np.load(feat_path)
        aligned_feats = torch.from_numpy(aligned_feats).float()
        T_feat = aligned_feats.shape[0]

        if debug_this_file:
            print(f"[DEBUG]   Features shape: {aligned_feats.shape}")
            print(f"[DEBUG]   Number of frames: {T_feat}")

        # 3. Parse segments from ground truth
        segments = get_segments_from_txt(label_path)

        if debug_this_file:
            print(f"[DEBUG]   Number of segments: {len(segments)}")
            print(f"[DEBUG]   First 3 segments:")
            for seg in segments[:3]:
                duration = seg['end'] - seg['start'] + 1
                print(f"[DEBUG]     - {seg['name']}: frames {seg['start']}-{seg['end']} (duration: {duration})")

        # 4. Verify label length matches feature length
        with open(label_path, 'r') as f:
            T_label = len([line.strip() for line in f if line.strip()])

        if T_feat != T_label:
            skipped_length_mismatch += 1
            if debug_this_file:
                print(f"[DEBUG]   ✗ Length mismatch! Features: {T_feat}, Labels: {T_label}")
            continue

        # 5. Calculate geodesic distance for each segment
        for seg_idx, seg in enumerate(segments):
            action_name = seg['name']
            start = seg['start']
            end = seg['end'] + 1  # Convert inclusive end to exclusive for slicing

            # Bounds checking
            start = max(0, min(start, T_feat))
            end = max(0, min(end, T_feat))

            segment_length = end - start

            if segment_length < 2:
                skipped_short_segments += 1
                if debug_this_file and seg_idx < 3:
                    print(f"[DEBUG]     ✗ Segment {seg_idx} too short ({segment_length} frames)")
                continue

            # Extract segment features
            segment_feats = aligned_feats[start:end]

            # Calculate geodesic distance
            geo_dist = calculate_geodesic_distance(segment_feats)

            if debug_this_file and seg_idx < 3:
                print(f"[DEBUG]     ✓ Segment {seg_idx} ({action_name}): geodesic = {geo_dist:.4f}")

            # Store distance
            if action_name not in action_distances:
                action_distances[action_name] = []
            action_distances[action_name].append(geo_dist)
            total_segments += 1

        processed += 1

    print(f"\n[3/4] Aggregating statistics...")

    # 4. Aggregate statistics per action
    final_stats = {}
    for action_name in sorted(action_distances.keys()):
        dists = action_distances[action_name]
        final_stats[action_name] = {
            "mean": float(np.mean(dists)),
            "std": float(np.std(dists)),
            "min": float(np.min(dists)),
            "max": float(np.max(dists)),
            "count": len(dists)
        }

    # Display statistics
    print(f"\n[DEBUG] Action statistics:")
    sorted_actions = sorted(final_stats.items(), key=lambda x: x[1]['count'], reverse=True)

    # Show background separately if it exists
    if 'background' in final_stats:
        bg_stats = final_stats['background']
        print(f"\n  Background class:")
        print(f"    {'background':30s}: mean={bg_stats['mean']:8.4f}, std={bg_stats['std']:8.4f}, n={bg_stats['count']:4d}")
        print(f"\n  Other actions (top 10 by count):")
        other_actions = [item for item in sorted_actions if item[0] != 'background'][:10]
    else:
        print(f"\n  Top 10 actions by count:")
        other_actions = sorted_actions[:10]

    for action_name, stats in other_actions:
        print(f"    {action_name:30s}: mean={stats['mean']:8.4f}, std={stats['std']:8.4f}, n={stats['count']:4d}")

    # 5. Save to JSON
    print(f"\n[4/4] Saving results...")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_stats, f, indent=4)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total videos processed:        {processed}/{len(feat_files)}")
    print(f"Videos missing labels:         {skipped_no_label}")
    print(f"Videos with length mismatch:   {skipped_length_mismatch}")
    print(f"Total segments processed:      {total_segments}")
    print(f"Segments skipped (too short):  {skipped_short_segments}")
    print(f"Unique actions found:          {len(final_stats)}")
    if 'background' in final_stats:
        print(f"  - Including 'background':    Yes ({final_stats['background']['count']} segments)")
        print(f"  - Action classes (non-bg):   {len(final_stats) - 1}")

    print(f"\n[SUBSET DISTRIBUTION]")
    print(f"Videos found per subset:")
    for subset_name in SUBSETS:
        count = subset_distribution[subset_name]
        pct = (count / processed * 100) if processed > 0 else 0
        print(f"  {subset_name:30s}: {count:3d} videos ({pct:5.1f}%)")

    print(f"\n✓ Results saved to: {OUTPUT_JSON}")
    print("="*80)


if __name__ == "__main__":
    main()
