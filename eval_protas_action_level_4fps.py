"""
Action-Level OGPE Evaluation for ProTAS Model (4 FPS Version)

Uses get_trueprogress_per_action() from utils.tensorops for action-level
progress (each action goes 0->1 independently).

Results are saved PER TASK (Brownie.cmu, Eggs.cmu, etc.) to match
the GTCC output format exactly.

Each subset has its own ProTAS model with different num_classes.
Videos are evaluated using the model trained on their respective subset.

OGPE = mean(|true_progress - pred_progress|)
"""

import os
import json
import numpy as np
import pandas as pd
import torch

from models.protas_model import MultiStageModel
# Import action-level progress function for action-level evaluation
from utils.tensorops import get_trueprogress_per_action
from utils.logging import configure_logging_format

logger = configure_logging_format()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths - UPDATED FOR 4FPS
WHITELIST_PATH = '/vision/anishn/GTCC_CVPR2024/evaluation_video_whitelist.json'
EGOPROCEL_JSON_PATH = '/vision/anishn/GTCC_CVPR2024/dset_jsons/egoprocel.json'
PROTAS_BASE = '/vision/anishn/ProTAS/data_4fps/'
OUTPUT_FOLDER = '/vision/anishn/GTCC_CVPR2024/output_4fps/multi-task-setting/protas_eval_egoprocel_act_level_4fps'

# Per-subset configuration - UPDATED FOR 4FPS MODELS
# Each subset has different num_classes and its own trained model
SUBSET_CONFIGS = {
    'egoprocel_subset1_S': {
        'num_classes': 30,
        'model_path': '/u/anishn/models/egoprocel_subset1_S_4fps/egoprocel_subset1_S/split_1/epoch-50.model',
        'graph_path': '/vision/anishn/ProTAS/data_4fps/egoprocel_subset1_S/graph/graph.pkl',
    },
    'egoprocel_subset2_OP_P': {
        'num_classes': 50,
        'model_path': '/u/anishn/models/egoprocel_subset2_OP_P_4fps/egoprocel_subset2_OP_P/split_1/epoch-50.model',
        'graph_path': '/vision/anishn/ProTAS/data_4fps/egoprocel_subset2_OP_P/graph/graph.pkl',
    },
    'egoprocel_subset3_tent': {
        'num_classes': 12,
        'model_path': '/u/anishn/models/egoprocel_subset3_tent_4fps/egoprocel_subset3_tent/split_1/epoch-50.model',
        'graph_path': '/vision/anishn/ProTAS/data_4fps/egoprocel_subset3_tent/graph/graph.pkl',
    },
    'egoprocel_subset4_numbers': {
        'num_classes': 18,
        'model_path': '/u/anishn/models/egoprocel_subset4_numbers_4fps/egoprocel_subset4_numbers/split_1/epoch-50.model',
        'graph_path': '/vision/anishn/ProTAS/data_4fps/egoprocel_subset4_numbers/graph/graph.pkl',
    },
    'egoprocel_subset5_head': {
        'num_classes': 19,
        'model_path': '/u/anishn/models/egoprocel_subset5_head_4fps/egoprocel_subset5_head/split_1/epoch-50.model',
        'graph_path': '/vision/anishn/ProTAS/data_4fps/egoprocel_subset5_head/graph/graph.pkl',
    },
}

# Common ProTAS params (num_classes and init_graph_path are subset-specific)
COMMON_PROTAS_PARAMS = {
    'num_stages': 4,
    'num_layers': 10,
    'num_f_maps': 64,
    'dim': 2048,
    'causal': True,
    'use_graph': True,
    'learnable': True,
}

# Cache for loaded models (avoid reloading for each video)
_model_cache = {}
_mapping_cache = {}


def load_model_for_subset(subset_name):
    """Load ProTAS model for a specific subset, with caching."""
    if subset_name in _model_cache:
        return _model_cache[subset_name]

    config = SUBSET_CONFIGS[subset_name]

    # Build full params for this subset
    params = {
        **COMMON_PROTAS_PARAMS,
        'num_classes': config['num_classes'],
        'init_graph_path': config['graph_path'],
    }

    logger.info(f"Loading ProTAS model for {subset_name} (num_classes={config['num_classes']})")
    model = MultiStageModel(**params)
    state_dict = torch.load(config['model_path'], map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _model_cache[subset_name] = model
    return model


def load_action_mapping(subset_name):
    """Load action name -> ID mapping from mapping.txt with caching"""
    if subset_name in _mapping_cache:
        return _mapping_cache[subset_name]

    mapping_path = f'{PROTAS_BASE}/{subset_name}/mapping.txt'
    mapping = {}
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        action_id = int(parts[0])
                        action_name = parts[1]
                        mapping[action_name] = action_id
    _mapping_cache[subset_name] = mapping
    return mapping


def parse_groundtruth_to_tdict(gt_path):
    """
    Read ground truth file and create tdict in the EXACT format
    expected by get_trueprogress() from utils.tensorops.

    tdict must have:
    - 'step': list of action names (use 'SIL' for background, not 'background')
    - 'start_frame': list of segment start frames
    - 'end_frame': list of segment end frames
    """
    with open(gt_path, 'r') as f:
        action_names_per_frame = [line.strip() for line in f if line.strip()]

    if not action_names_per_frame:
        return None, None

    # Group consecutive identical actions into segments
    segments = []
    current_action = action_names_per_frame[0]
    start_frame = 0

    for i in range(1, len(action_names_per_frame)):
        if action_names_per_frame[i] != current_action:
            segments.append({
                'name': current_action,
                'start': start_frame,
                'end': i - 1
            })
            current_action = action_names_per_frame[i]
            start_frame = i

    # Add final segment
    segments.append({
        'name': current_action,
        'start': start_frame,
        'end': len(action_names_per_frame) - 1
    })

    # Convert to tdict format
    # IMPORTANT: get_trueprogress() checks for 'SIL', so map 'background' -> 'SIL'
    tdict = {
        'step': ['SIL' if seg['name'] == 'background' else seg['name'] for seg in segments],
        'start_frame': [seg['start'] for seg in segments],
        'end_frame': [seg['end'] for seg in segments]
    }

    return tdict, action_names_per_frame


def find_video_subset(video_name):
    """Find which subset contains this video's ground truth"""
    for subset_name in SUBSET_CONFIGS.keys():
        gt_path = f'{PROTAS_BASE}/{subset_name}/groundTruth/{video_name}.txt'
        if os.path.exists(gt_path):
            return subset_name, gt_path
    return None, None


def evaluate_single_video(video_name, subset_name, gt_path):
    """
    Evaluate a single video and return its OGPE.
    Returns None if evaluation fails.
    """
    # Load features
    features_path = f'{PROTAS_BASE}/{subset_name}/features/{video_name}.npy'
    if not os.path.exists(features_path):
        logger.warning(f"No features found for {video_name}, skipping")
        return None

    try:
        # Load the correct model for this subset
        model = load_model_for_subset(subset_name)
        action_to_id = load_action_mapping(subset_name)

        # Load features: shape (T, 2048)
        features = np.load(features_path)

        # Parse ground truth into tdict format
        tdict, action_names_per_frame = parse_groundtruth_to_tdict(gt_path)
        if tdict is None:
            logger.warning(f"{video_name}: Could not parse ground truth, skipping")
            return None

        # Check if video has any non-SIL actions
        non_sil_count = sum([1 if step != 'SIL' else 0 for step in tdict['step']])
        if non_sil_count == 0:
            logger.warning(f"{video_name}: No non-background actions, skipping")
            return None

        # Check length alignment
        T_features = features.shape[0]
        T_gt = len(action_names_per_frame)

        if T_features != T_gt:
            T = min(T_features, T_gt)
            features = features[:T]
            action_names_per_frame = action_names_per_frame[:T]
            # Rebuild tdict with adjusted length
            tdict, action_names_per_frame = parse_groundtruth_to_tdict(gt_path)
            tdict['end_frame'][-1] = T - 1
        else:
            T = T_features
            # Match evaluation.py line 279: tdict['end_frame'][-1] = outputs.shape[0]-1
            tdict['end_frame'][-1] = T - 1

        # Prepare input for ProTAS: [Batch, Channels, Time]
        x = torch.from_numpy(features).float().T.unsqueeze(0).to(device)  # (1, 2048, T)
        mask = torch.ones(1, 1, T).to(device)

        # Run ProTAS inference
        with torch.no_grad():
            predictions, progress_predictions = model(x, mask)

        # Get final stage progress: [Batch, Classes, Time] -> [Classes, Time]
        prog_matrix = progress_predictions[-1].squeeze(0)  # (num_classes, T)

        # Create GT class labels per frame
        gt_class_per_frame = torch.zeros(T, dtype=torch.long).to(device)
        for t, action_name in enumerate(action_names_per_frame):
            gt_class_per_frame[t] = action_to_id.get(action_name, 0)

        # Extract predicted progress for GT class at each frame
        pred_progress = torch.gather(
            prog_matrix, 0, gt_class_per_frame.unsqueeze(0)
        ).squeeze(0)  # (T,)

        # Clamp to [0, 1] range
        pred_progress = torch.clamp(pred_progress, 0, 1)

        # Get ground truth progress (action-level: each action goes 0->1 independently)
        true_progress = get_trueprogress_per_action(tdict).to(device)

        # Create mask to exclude background frames (only evaluate on action frames)
        action_mask = torch.tensor([label != 'background' for label in action_names_per_frame],
                                   dtype=torch.bool, device=device)

        # OGPE calculation - ONLY on action frames (exclude background)
        if action_mask.any():
            errors = torch.abs(true_progress - pred_progress)
            gpe = errors[action_mask].mean()
        else:
            gpe = torch.tensor(0.0)  # No action frames
        return gpe.item()

    except Exception as e:
        logger.error(f"Error processing {video_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    logger.info("=" * 60)
    logger.info("Action-Level OGPE Evaluation for ProTAS (4 FPS)")
    logger.info("Results organized by TASK (matching GTCC format)")
    logger.info("=" * 60)

    # Load whitelist
    with open(WHITELIST_PATH, 'r') as f:
        whitelist_data = json.load(f)
    video_whitelist = set(whitelist_data['video_names'])
    logger.info(f"[INFO] Test set whitelist: {len(video_whitelist)} videos")

    # Load egoprocel.json to get task structure (same as GTCC uses)
    with open(EGOPROCEL_JSON_PATH, 'r') as f:
        egoprocel_data = json.load(f)

    # Create video -> task mapping
    video_to_task = {}
    for task, task_data in egoprocel_data.items():
        for handle in task_data['handles']:
            video_to_task[handle] = task
    logger.info(f"[INFO] Loaded {len(video_to_task)} video-to-task mappings from egoprocel.json")

    # Group whitelist videos by task
    task_videos = {}
    unmapped_videos = []
    for video_name in video_whitelist:
        task = video_to_task.get(video_name)
        if task:
            if task not in task_videos:
                task_videos[task] = []
            task_videos[task].append(video_name)
        else:
            unmapped_videos.append(video_name)

    if unmapped_videos:
        logger.warning(f"[WARNING] {len(unmapped_videos)} videos not mapped to any task: {unmapped_videos[:5]}...")

    logger.info(f"[INFO] Found {len(task_videos)} tasks with whitelist videos")

    # Results storage (matching GTCC format)
    results = {'task': [], 'ogpe': [], 'CoV': [], 'num_videos': []}

    total_processed = 0
    total_skipped = 0

    # Process each task (sorted to match GTCC output order)
    for task in sorted(task_videos.keys()):
        videos = task_videos[task]
        logger.info(f"\n{'*' * 40}")
        logger.info(f"Processing task: {task} ({len(videos)} videos)")
        logger.info(f"{'*' * 40}")

        gpe_list = []
        task_skipped = 0

        for video_name in videos:
            # Find which ProTAS subset this video belongs to
            subset_name, gt_path = find_video_subset(video_name)
            if subset_name is None:
                logger.warning(f"  No ground truth found for {video_name}, skipping")
                task_skipped += 1
                continue

            # Evaluate this video
            gpe = evaluate_single_video(video_name, subset_name, gt_path)
            if gpe is not None:
                gpe_list.append(gpe)
            else:
                task_skipped += 1

        # Compute task-level metrics
        if gpe_list:
            task_ogpe = np.mean(gpe_list)
            task_std = np.std(gpe_list)
            task_cov = task_std / task_ogpe if task_ogpe > 0 else 0

            results['task'].append(task)
            results['ogpe'].append(task_ogpe)
            results['CoV'].append(task_cov)
            results['num_videos'].append(len(gpe_list))

            logger.info(f"  Task {task}: OGPE={task_ogpe:.4f}, CoV={task_cov:.4f}, n={len(gpe_list)}")
            total_processed += len(gpe_list)
        else:
            logger.warning(f"  Task {task}: No videos processed successfully")

        total_skipped += task_skipped

    # Add MEAN row (same as GTCC format)
    if results['ogpe']:
        mean_ogpe = np.mean(results['ogpe'])
        mean_cov = np.mean(results['CoV'])
        mean_num_videos = np.mean(results['num_videos'])

        results['task'].append('MEAN')
        results['ogpe'].append(mean_ogpe)
        results['CoV'].append(mean_cov)
        results['num_videos'].append(mean_num_videos)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total processed: {total_processed} videos")
    logger.info(f"Total skipped: {total_skipped} videos")
    logger.info(f"Tasks evaluated: {len(results['task']) - 1}")  # -1 for MEAN row

    if results['ogpe']:
        logger.info(f"\nMean OGPE across tasks: {mean_ogpe:.4f}")

    # Save results
    eval_folder = f'{OUTPUT_FOLDER}/EVAL/ProTAS_ActionLevel'
    os.makedirs(eval_folder, exist_ok=True)

    # Save in same CSV format as GTCC evaluation.py
    df = pd.DataFrame(results)
    df.to_csv(f'{eval_folder}/ogpe.csv')

    logger.info(f"\nResults saved to {eval_folder}/ogpe.csv")
    logger.info("=" * 60)
    logger.info("Evaluation complete")


if __name__ == '__main__':
    main()
