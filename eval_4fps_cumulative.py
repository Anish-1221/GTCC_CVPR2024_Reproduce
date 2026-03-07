"""
Evaluate cumulative L2 progress at 4fps using a 1fps-trained model.

Saves all 4fps artifacts to separate paths to avoid overwriting 1fps results:
  - {exp_folder}/aligned_features_4fps/
  - {exp_folder}/action_means_4fps.json
  - {exp_folder}/ogpe_4fps.csv

Usage:
    python eval_4fps_cumulative.py --exp_folder output_l2_progress_v2/multi-task-setting_val/V1___GTCC_egoprocel
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from generate_aligned_features import initialize_model
from utils.tensorops import get_cum_matrix, get_trueprogress_per_action
from utils.logging import configure_logging_format

logger = configure_logging_format()

RAW_4FPS_DIR = '/vision/anishn/GTCC_Data_Processed_4fps/egoprocel/frames/'
PROTAS_4FPS = '/vision/anishn/ProTAS/data_4fps'
SPLITS_PATH = '/vision/anishn/GTCC_CVPR2024/data_splits.json'
SUBSETS = [
    'egoprocel_subset1_S', 'egoprocel_subset2_OP_P', 'egoprocel_subset3_tent',
    'egoprocel_subset4_numbers', 'egoprocel_subset5_head'
]


def parse_groundtruth_to_segments(gt_path):
    """Same as evaluation_action_level.py"""
    with open(gt_path, 'r') as f:
        action_names = [line.strip() for line in f if line.strip()]
    segments = []
    if not action_names:
        return segments, 0
    current_action = action_names[0]
    start_frame = 0
    for i in range(1, len(action_names)):
        if action_names[i] != current_action:
            segments.append({'name': current_action, 'start': start_frame, 'end': i - 1})
            current_action = action_names[i]
            start_frame = i
    segments.append({'name': current_action, 'start': start_frame, 'end': len(action_names) - 1})
    return segments, len(action_names)


def find_gt_path(video_name):
    for subset in SUBSETS:
        path = f'{PROTAS_4FPS}/{subset}/groundTruth/{video_name}.txt'
        if os.path.exists(path):
            return path
    return None


# ──────────────────────────────────────────────
# Phase 1: Generate 4fps aligned features
# ──────────────────────────────────────────────
def generate_aligned_features_4fps(exp_folder, device):
    out_dir = os.path.join(exp_folder, 'aligned_features_4fps')
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
        logger.info(f"[SKIP] aligned_features_4fps/ already exists with {len(os.listdir(out_dir))} files")
        return out_dir

    os.makedirs(out_dir, exist_ok=True)
    model, config, model_type = initialize_model(exp_folder, device)

    feat_files = sorted([f for f in os.listdir(RAW_4FPS_DIR) if f.endswith('.npy')])
    logger.info(f"Generating 4fps aligned features for {len(feat_files)} videos...")

    successful, failed = 0, 0
    with torch.no_grad():
        for f_name in tqdm(feat_files, desc="4fps aligned features"):
            try:
                raw = np.load(os.path.join(RAW_4FPS_DIR, f_name))
                feat_tensor = torch.from_numpy(raw).float().to(device)
                out_dict = model([feat_tensor])
                aligned = out_dict['outputs'][0].cpu().numpy()
                np.save(os.path.join(out_dir, f_name), aligned)
                successful += 1
            except Exception as e:
                failed += 1
                if failed <= 3:
                    logger.error(f"Failed {f_name}: {e}")

    logger.info(f"Aligned features: {successful} ok, {failed} failed → {out_dir}")
    return out_dir


# ──────────────────────────────────────────────
# Phase 2: Calculate 4fps action means
# ──────────────────────────────────────────────
def calculate_action_means_4fps(exp_folder, aligned_dir):
    out_json = os.path.join(exp_folder, 'action_means_4fps.json')
    if os.path.exists(out_json):
        logger.info(f"[SKIP] action_means_4fps.json already exists")
        with open(out_json, 'r') as f:
            return json.load(f)

    feat_files = sorted([f for f in os.listdir(aligned_dir) if f.endswith('.npy')])
    logger.info(f"Calculating 4fps action means from {len(feat_files)} files...")

    action_distances = {}
    for f_name in tqdm(feat_files, desc="4fps action means"):
        video_name = os.path.splitext(f_name)[0]
        gt_path = find_gt_path(video_name)
        if gt_path is None:
            continue

        aligned = torch.from_numpy(np.load(os.path.join(aligned_dir, f_name))).float()
        segments, total_frames = parse_groundtruth_to_segments(gt_path)

        if aligned.shape[0] != total_frames:
            continue

        for seg in segments:
            start, end = seg['start'], min(seg['end'] + 1, aligned.shape[0])
            if end - start < 2:
                continue
            diffs = torch.norm(aligned[start+1:end] - aligned[start:end-1], p=2, dim=1)
            geo_dist = torch.sum(diffs).item()
            if seg['name'] not in action_distances:
                action_distances[seg['name']] = []
            action_distances[seg['name']].append(geo_dist)

    action_means = {}
    for name, dists in sorted(action_distances.items()):
        action_means[name] = {
            'mean': float(np.mean(dists)),
            'std': float(np.std(dists)),
            'count': len(dists)
        }

    with open(out_json, 'w') as f:
        json.dump(action_means, f, indent=4)
    logger.info(f"Saved {len(action_means)} action means → {out_json}")
    return action_means


# ──────────────────────────────────────────────
# Phase 3: Evaluate OGPE at 4fps
# ──────────────────────────────────────────────
def evaluate_ogpe_4fps(exp_folder, aligned_dir, action_means):
    with open(SPLITS_PATH, 'r') as f:
        splits = json.load(f)['splits']

    test_videos = {}
    for task, split in splits.items():
        test_videos[task] = split['test']

    results = []
    for task, videos in sorted(test_videos.items()):
        gpe_list = []
        for video_name in videos:
            gt_path = find_gt_path(video_name)
            if gt_path is None:
                continue

            feat_path = os.path.join(aligned_dir, f'{video_name}.npy')
            if not os.path.exists(feat_path):
                continue

            outputs = torch.from_numpy(np.load(feat_path)).float()
            segments, total_frames = parse_groundtruth_to_segments(gt_path)
            current_len = outputs.shape[0]

            if current_len != total_frames:
                logger.warning(f"Length mismatch {video_name}: features={current_len}, gt={total_frames}")
                continue

            tdict = {
                'step': [seg['name'] for seg in segments],
                'start_frame': [seg['start'] for seg in segments],
                'end_frame': [min(seg['end'], current_len - 1) for seg in segments],
            }
            true_progress = get_trueprogress_per_action(tdict)

            pred_progress = torch.zeros(current_len)
            for seg in segments:
                action_name = seg['name']
                start = max(0, min(seg['start'], current_len - 1))
                end = max(0, min(seg['end'], current_len - 1))
                if start >= current_len or action_name in ['SIL', 'background']:
                    continue

                segment_outputs = outputs[start:end + 1]
                segment_cum = get_cum_matrix(segment_outputs)

                if action_name in action_means and action_means[action_name]['mean'] > 0:
                    pred_progress[start:end + 1] = segment_cum / action_means[action_name]['mean']
                # else: stays at 0

            # Action mask
            action_mask = torch.zeros(len(true_progress), dtype=torch.bool)
            for seg in segments:
                if seg['name'] not in ['SIL', 'background']:
                    s = max(0, min(seg['start'], len(true_progress) - 1))
                    e = max(0, min(seg['end'], len(true_progress) - 1))
                    action_mask[s:e + 1] = True

            if action_mask.any():
                gpe = torch.abs(true_progress - pred_progress)[action_mask].mean().item()
                gpe_list.append(gpe)

        if gpe_list:
            results.append({'task': task, 'ogpe': np.mean(gpe_list), 'num_videos': len(gpe_list)})
            logger.info(f"  {task}: OGPE={np.mean(gpe_list):.4f} ({len(gpe_list)} videos)")

    # Save
    df = pd.DataFrame(results)
    mean_row = pd.DataFrame([{'task': 'MEAN', 'ogpe': df['ogpe'].mean(), 'num_videos': int(df['num_videos'].sum())}])
    df = pd.concat([df, mean_row], ignore_index=True)

    csv_path = os.path.join(exp_folder, 'ogpe_4fps.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"\nMEAN OGPE (4fps): {df[df['task']=='MEAN']['ogpe'].values[0]:.4f}")
    logger.info(f"Saved → {csv_path}")
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate cumulative L2 progress at 4fps')
    parser.add_argument('--exp_folder', type=str, required=True)
    parser.add_argument('--skip_features', action='store_true',
                        help='Skip feature generation (use existing aligned_features_4fps/)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = args.exp_folder

    logger.info(f"{'='*60}")
    logger.info(f"4fps Cumulative L2 Evaluation")
    logger.info(f"Experiment: {exp}")
    logger.info(f"{'='*60}")

    # Phase 1
    if args.skip_features:
        aligned_dir = os.path.join(exp, 'aligned_features_4fps')
        if not os.path.exists(aligned_dir):
            logger.error(f"--skip_features but {aligned_dir} not found!")
            exit(1)
    else:
        aligned_dir = generate_aligned_features_4fps(exp, device)

    # Phase 2
    action_means = calculate_action_means_4fps(exp, aligned_dir)

    # Phase 3
    evaluate_ogpe_4fps(exp, aligned_dir, action_means)
