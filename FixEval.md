# Evaluation Commands and Action-Level OGPE Setup Plan

## Summary

This plan provides:
1. Fix for eval.py to use fixed splits from `data_splits.json` (not random splits) - **DONE**
2. Remove obsolete whitelist filtering from evaluation_action_level.py - **DONE**
3. Add 4fps support to evaluation_action_level.py - **DONE**
4. **Store aligned_features/ and action_means.json INSIDE each experiment folder**
5. **Update unified scripts to use experiment folder storage**
6. **Update evaluation to load action_means.json from experiment folder**
7. Commands to generate aligned features and action means for all models
8. All eval commands

## NEW: Experiment Folder Storage Structure

Each experiment folder will contain its own aligned features and action means:

```
output_4fps_val/multi-task-setting_val/V1___GTCC_egoprocel.4fps/
├── ckpt/
│   └── best_model.pt
├── config.json
├── aligned_features/          ← NEW: model-specific aligned features
│   ├── video1.npy
│   ├── video2.npy
│   └── ...
└── action_means.json          ← NEW: model-specific action means
```

**Benefits:**
- No naming conflicts between experiments
- Delete experiment = delete all artifacts
- Evaluation automatically finds correct files
- Clear ownership of which model produced which features

---

## IMPORTANT: Learnable Progress Models Don't Need Prerequisites!

**Learnable progress models (`output_learnable_progress/`) do NOT require:**
- ❌ No aligned_features/ directory
- ❌ No action_means.json file
- ❌ No preprocessing steps

**Why?** Learnable models use `ProgressHead` (a GRU neural network) that directly predicts progress from embeddings. They don't use cumulative L2 distance or action means at all.

**For learnable models, just run evaluation directly:**
```bash
python eval.py -f output_learnable_progress/multi-task-setting_val/V1___GTCC_egoprocel --level action
```

**Models that DO require aligned_features/ and action_means.json:**
- `output_val/` (baseline models)
- `output_l2_progress/` (L2 progress loss models)
- `output_4fps_val/` (4fps baseline models)

---

## Do Progress Loss Models Need Aligned Features / Action Means?

### CRITICAL FINDING: Training vs Evaluation Mismatch

**There is an inconsistency between how progress is computed during TRAINING vs EVALUATION!**

---

### Training Analysis (loss_entry.py + tensorops.py)

**Cumulative L2 Progress Loss** (`--progress_loss cumulative_l2`):
```python
# tensorops.py, line 360-363:
segment_cum = get_cum_matrix(segment_features)
max_val = segment_cum[-1]
pred_progress[start:end+1] = segment_cum / max_val  # NORMALIZE BY MAX!
```
- Normalizes by **max value within the segment itself** (self-normalization)
- Progress ALWAYS goes 0→1 for each action segment
- **NO external action means used during training**

**Learnable Progress Loss** (`--progress_loss learnable`):
```python
# loss_entry.py, line 121:
pred_prog = progress_head(seg_emb)  # Direct prediction from neural network
```
- ProgressHead (GRU) directly outputs progress in [0,1]
- **NO cumulative L2 distance computed at all**
- **NO action means used during training**

---

### Evaluation Analysis (evaluation_action_level.py)

**Current evaluation code (line 388-396):**
```python
segment_cum = get_cum_matrix(segment_outputs)
if action_name in action_means:
    action_mean = action_means[action_name]['mean']
    pred2_progress[start:end+1] = segment_cum / action_mean  # NORMALIZE BY ACTION MEAN!
```

**ALL models currently use the same evaluation method:**
- Cumulative L2 distance normalized by **action-specific means from JSON**
- This is DIFFERENT from training normalization!

---

### The Mismatch Problem

| Model Type | Training Method | Current Eval Method | Match? |
|------------|-----------------|---------------------|--------|
| GTCC/TCC/VAVA/LAV (baseline) | N/A (no progress) | cum_l2 / action_mean | ✓ (expected) |
| L2 Progress Loss | cum_l2 / **max_val** | cum_l2 / **action_mean** | ✗ **MISMATCH** |
| Learnable Progress Loss | **ProgressHead(emb)** | cum_l2 / action_mean | ✗✗ **BIG MISMATCH** |

---

### Why This Matters

**For L2 Progress Model:**
- Training: segment_cum / max_val → progress ends at exactly 1.0
- Evaluation: segment_cum / action_mean → progress can be >1 or <1

The difference: if an action instance has higher geodesic distance than the mean, eval shows progress >1.

**For Learnable Progress Model:**
- Training: ProgressHead neural network directly predicts progress
- Evaluation: **Completely ignores the ProgressHead** and uses cumulative L2 instead!

This means the learned ProgressHead is never used during evaluation, which defeats the purpose of training it.

---

### Recommended Fix

The evaluation code should be modified to match training:

**Option 1: For Learnable Models - Use ProgressHead directly**
```python
if hasattr(model, 'progress_head') and model.progress_head is not None:
    # Use the trained progress head for prediction
    for seg in segments:
        segment_outputs = outputs[seg['start']:seg['end']+1]
        for t in range(len(segment_outputs)):
            seg_emb = segment_outputs[:t+1]
            if seg_emb.shape[0] >= 2:
                pred2_progress[seg['start'] + t] = model.progress_head(seg_emb).item()
```

**Option 2: For L2 Models - Use max_val normalization (same as training)**
```python
# Instead of: segment_cum / action_mean
# Use: segment_cum / max_val (same as training)
pred_progress = get_normalized_predicted_progress_action(outputs, tdict)
```

**Option 3: Keep action means for fair cross-model comparison**
- Using action means allows comparing baseline models vs progress loss models fairly
- But we should acknowledge this is different from training

---

### Summary Table (Corrected Analysis)

| Model Type | Needs Aligned Features? | Needs Action Means JSON? | Notes |
|------------|------------------------|--------------------------|-------|
| GTCC (baseline) | YES | YES | Cumulative L2 / action_mean |
| TCC (baseline) | YES | YES | Cumulative L2 / action_mean |
| VAVA (baseline) | YES | YES | Cumulative L2 / action_mean |
| LAV (baseline) | YES | YES | Cumulative L2 / action_mean |
| L2 Progress Loss | YES | YES | Action means computed from THIS model's aligned features |
| Learnable Progress Loss | **NO** | **NO** | ProgressHead predicts directly |

**Key Points:**
1. Each model generates its OWN aligned features and action means - they're model-specific
2. L2 progress models use cumulative L2 / action_mean (action means from same model, so consistent)
3. Learnable progress models should use ProgressHead directly (eval code needs update)

### Required Eval Code Fix for Learnable Models

The evaluation code needs to check if model has a ProgressHead and use it:

```python
# In evaluation_action_level.py OnlineGeoProgressError.__call__:

# Check if this is a learnable progress model
has_progress_head = hasattr(model, 'progress_head') and model.progress_head is not None

if has_progress_head:
    # Use ProgressHead directly for prediction
    for seg in segments:
        action_name = seg['name']
        start = max(0, min(seg['start'], current_len-1))
        end = max(0, min(seg['end'], current_len-1))

        if action_name not in ['SIL', 'background']:
            segment_outputs = outputs[start:end+1]
            # Predict progress for each frame using cumulative segment up to that point
            for t in range(len(segment_outputs)):
                seg_emb = segment_outputs[:t+1]
                if seg_emb.shape[0] >= 2:
                    with torch.no_grad():
                        pred2_progress[start + t] = model.progress_head(seg_emb).item()
                else:
                    pred2_progress[start + t] = 0.0
else:
    # Use cumulative L2 with action means (for baseline and L2 progress models)
    # ... existing code ...
```

---

## Critical Issue Found

### Inconsistent Test Sets

| File | Split Type | Test Videos |
|------|-----------|-------------|
| `data_splits.json` | Fixed stratified (75/10/15) | **48 videos** |
| `evaluation_video_whitelist.json` | Old random (65/35) | **113 videos** |

**Problem:**
- Training uses `jsondataset_from_splits` → reads from `data_splits.json` ✓
- Evaluation uses `get_test_dataloaders` → uses OLD random split ✗

**Solution:** Remove whitelist, update eval.py to use fixed splits.

---

## Part 1: Fix eval.py to Use Fixed Splits

### File: `/vision/anishn/GTCC_CVPR2024/eval.py`

**Add import at top (around line 214):**
```python
from models.json_dataset import get_test_dataloaders, jsondataset_from_splits, data_json_labels_handles
from torch.utils.data import DataLoader
from utils.collate_functions import jsondataset_collate_fn
import json
```

**Add function to load splits (after imports):**
```python
def load_splits_from_json(splits_path='/vision/anishn/GTCC_CVPR2024/data_splits.json'):
    """Load pre-computed train/val/test splits."""
    with open(splits_path, 'r') as f:
        splits_data = json.load(f)
    return splits_data['splits']
```

**Replace `get_test_dataloaders` call (around line 376) with:**
```python
    # Load fixed splits from data_splits.json
    splits_dict = load_splits_from_json()

    data_subfolder_name, datafile_extension = get_data_subfolder_and_extension(architecture=config.BASEARCH.ARCHITECTURE)
    data_folder = f'{config.DATAFOLDER}/{data_subfolder_name}'

    test_dataloaders = {}
    for task in testTASKS:
        test_set = jsondataset_from_splits(
            task=task,
            task_json=data_structure[task],
            data_folder=data_folder,
            splits_dict=splits_dict,
            split_type='test',
            extension=datafile_extension,
            lazy_loading=config.LAZY_LOAD
        )
        logger.info(f'{len(test_set)} videos in test set for {task}')
        batch_size = config.BATCH_SIZE if config.BATCH_SIZE else len(test_set)
        test_dataloaders[task] = DataLoader(
            test_set,
            batch_size=batch_size,
            collate_fn=jsondataset_collate_fn,
            drop_last=False,  # Don't drop last batch for evaluation
            shuffle=False
        )
```

---

## Part 2: Remove Whitelist from evaluation_action_level.py

### File: `/vision/anishn/GTCC_CVPR2024/utils/evaluation_action_level.py`

**Remove whitelist loading (delete lines 232, 238-244):**
```python
# DELETE THESE LINES:
WHITELIST_PATH = '/vision/anishn/GTCC_CVPR2024/evaluation_video_whitelist.json'
# ...
video_whitelist = None
if os.path.exists(WHITELIST_PATH):
    with open(WHITELIST_PATH, 'r') as f:
        data = json.load(f)
    video_whitelist = set(data['video_names'])
    print(f"[INFO] Test set whitelist: {len(video_whitelist)} videos")
```

**Remove whitelist filtering (delete lines 324-333):**
```python
# DELETE THESE LINES:
if video_whitelist is not None:
    filtered_dl = []
    for output_dict, tdict in embedded_dl:
        video_name = output_dict.get('name', '')
        video_name = os.path.splitext(os.path.basename(video_name))[0]
        if video_name in video_whitelist:
            filtered_dl.append((output_dict, tdict))
    print(f"[INFO] GTCC: Using {len(filtered_dl)}/{len(embedded_dl)} whitelisted videos")
    embedded_dl = filtered_dl
```

---

## Part 3: Update evaluation_action_level.py to Load from Experiment Folder

### File: `/vision/anishn/GTCC_CVPR2024/utils/evaluation_action_level.py`

**Replace OnlineGeoProgressError.__call__ start with:**

```python
    def __call__(self, model, config_obj, epoch, test_dataloaders, testfolder, tasks, normalize=False, plotting=False):
        # [4FPS FIX] Detect if model is 4fps based on folder path
        is_4fps = '4fps' in testfolder

        if is_4fps:
            PROTAS_BASE = '/vision/anishn/ProTAS/data_4fps/'
            print(f"[INFO] Detected 4fps model, using 4fps paths")
        else:
            PROTAS_BASE = '/vision/anishn/ProTAS/data_1fps/'

        plot_folder = testfolder + '/plotting_progress'
        validate_folder(plot_folder)

        # [NEW] Load action means from EXPERIMENT FOLDER
        # Each model has its own action_means.json stored alongside its checkpoint
        action_means_path = os.path.join(testfolder, 'action_means.json')

        action_means = None
        if os.path.exists(action_means_path):
            with open(action_means_path, 'r') as f:
                action_means = json.load(f)
            print(f"[INFO] Loaded action means from experiment folder: {len(action_means)} actions")
        else:
            print(f"[ERROR] Action means not found at {action_means_path}")
            print(f"[ERROR] Run: python generate_aligned_features.py --exp_folder {testfolder}")
            print(f"[ERROR] Then: python calculate_action_means.py --exp_folder {testfolder}")
            return None
```

---

## Part 4: Update generate_aligned_features.py (Store in Experiment Folder)

### File: `/vision/anishn/GTCC_CVPR2024/generate_aligned_features.py`

**Key Change:** Output to `{exp_folder}/aligned_features/` instead of central location.

```python
#!/usr/bin/env python
"""
Unified Aligned Feature Generation Script - STORES IN EXPERIMENT FOLDER

Usage:
    python generate_aligned_features.py --exp_folder /path/to/model

Output: {exp_folder}/aligned_features/*.npy
"""
import argparse
import os
import torch
import numpy as np
import glob
from tqdm import tqdm

from utils.train_util import get_config_for_folder, ckpt_restore_mprong
from utils.logging import configure_logging_format

logger = configure_logging_format()


def get_input_path(fps: str):
    """Get input features path based on FPS"""
    return f'/vision/anishn/GTCC_Data_Processed_{fps}/egoprocel/frames/'


def detect_model_type(config):
    """Detect model type from config"""
    if config.LOSS_TYPE.get('GTCC', False):
        return 'gtcc'
    elif config.LOSS_TYPE.get('tcc', False):
        return 'tcc'
    elif config.LOSS_TYPE.get('VAVA', False):
        return 'vava'
    elif config.LOSS_TYPE.get('LAV', False):
        return 'lav'
    return 'gtcc'  # default


def initialize_model(exp_folder: str, device):
    """Initialize and load the trained model"""
    logger.info(f"[1/4] Loading config from {exp_folder}")
    config = get_config_for_folder(exp_folder)

    model_type = detect_model_type(config)
    logger.info(f"[DEBUG] Detected model type: {model_type.upper()}")

    # Find checkpoint
    ckpt_path = os.path.join(exp_folder, 'ckpt', 'best_model.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(exp_folder, 'ckpt', 'epoch-50.pt')
    if not os.path.exists(ckpt_path):
        ckpt_files = sorted(glob.glob(os.path.join(exp_folder, 'ckpt', 'epoch-*.pt')))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoints found in {exp_folder}/ckpt/")
        ckpt_path = ckpt_files[-1]

    logger.info(f"[2/4] Loading checkpoint from {ckpt_path}")

    # Detect number of heads
    ckpt_data = torch.load(ckpt_path, map_location='cpu')
    model_state = ckpt_data.get('model', ckpt_data)
    head_keys = [k for k in model_state.keys() if k.startswith('head_models.')]
    num_heads = max([int(k.split('.')[1]) for k in head_keys]) + 1 if head_keys else 16

    # GTCC uses dropout layers, others don't
    use_dropout = (model_type == 'gtcc')

    logger.info(f"[3/4] Restoring model with {num_heads} heads (dropout={use_dropout})...")
    model, epoch, loss, _, _ = ckpt_restore_mprong(
        ckpt_path,
        num_heads=num_heads,
        dropout=use_dropout,
        device=device
    )

    model.eval()
    logger.info(f"[4/4] Model loaded from epoch {epoch}")
    return model, config, model_type


def extract_features(model, input_dir: str, output_dir: str, device):
    """Extract aligned features from raw features"""
    os.makedirs(output_dir, exist_ok=True)

    feat_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
    logger.info(f"Found {len(feat_files)} feature files in {input_dir}")

    successful, failed = 0, 0

    with torch.no_grad():
        for idx, f_name in enumerate(tqdm(feat_files, desc="Extracting Aligned Features")):
            try:
                raw_feats = np.load(os.path.join(input_dir, f_name))
                feat_tensor = torch.from_numpy(raw_feats).float().to(device)

                out_dict = model([feat_tensor])
                aligned_feats = out_dict['outputs'][0].cpu().numpy()

                np.save(os.path.join(output_dir, f_name), aligned_feats)
                successful += 1

            except Exception as e:
                failed += 1
                logger.error(f"Failed to process {f_name}: {e}")

    logger.info(f"Extraction complete: {successful}/{len(feat_files)} successful")
    return successful, failed


def main():
    parser = argparse.ArgumentParser(description='Generate aligned features (stored in experiment folder)')
    parser.add_argument('--exp_folder', type=str, required=True,
                        help='Path to experiment folder containing ckpt/')
    parser.add_argument('--fps', type=str, default=None,
                        help='FPS of the data (auto-detected from folder name if not specified)')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-detect FPS from folder name
    fps = args.fps
    if fps is None:
        fps = '4fps' if '4fps' in args.exp_folder else '1fps'

    input_dir = get_input_path(fps)
    output_dir = os.path.join(args.exp_folder, 'aligned_features')  # INSIDE experiment folder!

    logger.info("="*80)
    logger.info("Aligned Feature Generation (Experiment Folder Storage)")
    logger.info("="*80)
    logger.info(f"Using device: {device}")
    logger.info(f"FPS: {fps}")
    logger.info(f"Experiment folder: {args.exp_folder}")
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")

    model, config, model_type = initialize_model(args.exp_folder, device)
    extract_features(model, input_dir, output_dir, device)

    logger.info(f"\nAligned features saved to: {output_dir}")
    logger.info(f"Next step: python calculate_action_means.py --exp_folder {args.exp_folder}")


if __name__ == "__main__":
    main()
```

### Usage (Simplified!)

```bash
cd /vision/anishn/GTCC_CVPR2024

# Just specify the experiment folder - FPS and model type auto-detected!
python generate_aligned_features.py --exp_folder output_4fps_val/multi-task-setting_val/V1___GTCC_egoprocel.4fps

# Output will be at: output_4fps_val/multi-task-setting_val/V1___GTCC_egoprocel.4fps/aligned_features/
```

---

## Part 5: Update calculate_action_means.py (Store in Experiment Folder)

### File: `/vision/anishn/GTCC_CVPR2024/calculate_action_means.py`

**Key Change:** Read from `{exp_folder}/aligned_features/`, output to `{exp_folder}/action_means.json`

```python
#!/usr/bin/env python
"""
Calculate Action Geodesic Means - STORES IN EXPERIMENT FOLDER

Usage:
    python calculate_action_means.py --exp_folder /path/to/model

Input:  {exp_folder}/aligned_features/*.npy
Output: {exp_folder}/action_means.json
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
    parser = argparse.ArgumentParser(description='Calculate action geodesic means (stored in experiment folder)')
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
    print(f"Found {len(feat_files)} aligned feature files")

    processed, total_segments = 0, 0

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

    print(f"\nProcessed: {processed}/{len(feat_files)} videos")
    print(f"Total segments: {total_segments}")
    print(f"Unique actions: {len(final_stats)}")
    print(f"\nAction means saved to: {output_json}")
    print(f"Ready for evaluation: python eval.py -f {args.exp_folder} --level action")


if __name__ == "__main__":
    main()
```

### Usage (Simplified!)

```bash
cd /vision/anishn/GTCC_CVPR2024

# Just specify the experiment folder!
python calculate_action_means.py --exp_folder output_4fps_val/multi-task-setting_val/V1___GTCC_egoprocel.4fps

# Output will be at: output_4fps_val/multi-task-setting_val/V1___GTCC_egoprocel.4fps/action_means.json
```

---

## Part 6: Complete Workflow for All Models

### Which models need aligned features and action means?

| Output Directory | Model Types | Need Aligned Features? | Need Action Means? |
|------------------|-------------|----------------------|-------------------|
| output_val | GTCC, TCC, LAV, VAVA (baseline) | YES | YES |
| output_l2_progress | GTCC/TCC/VAVA/LAV + L2 progress | YES | YES |
| output_learnable_progress | GTCC/TCC/VAVA/LAV + learnable | **NO** | **NO** |
| output_4fps_val | GTCC/TCC/VAVA/LAV (baseline) | YES | YES |

**Note:** Learnable progress models use ProgressHead directly - no aligned features or action means needed!

---

### COMPLETE COMMANDS FOR ALL MODELS

```bash
cd /vision/anishn/GTCC_CVPR2024

#############################################
# output_val (1fps baseline models)
#############################################

# GTCC baseline
python generate_aligned_features.py --exp_folder output_val/multi-task-setting_val/Vtest1___GTCC_egoprocel
python calculate_action_means.py --exp_folder output_val/multi-task-setting_val/Vtest1___GTCC_egoprocel

# TCC baseline
python generate_aligned_features.py --exp_folder output_val/multi-task-setting_val/Vtest1___tcc_egoprocel
python calculate_action_means.py --exp_folder output_val/multi-task-setting_val/Vtest1___tcc_egoprocel

# LAV baseline
python generate_aligned_features.py --exp_folder output_val/multi-task-setting_val/VtestLavLoss1___LAV_egoprocel
python calculate_action_means.py --exp_folder output_val/multi-task-setting_val/VtestLavLoss1___LAV_egoprocel

# VAVA baseline
python generate_aligned_features.py --exp_folder output_val/multi-task-setting_val/VtestVavaLoss1___VAVA_egoprocel
python calculate_action_means.py --exp_folder output_val/multi-task-setting_val/VtestVavaLoss1___VAVA_egoprocel

#############################################
# output_l2_progress (1fps L2 progress models)
#############################################

python generate_aligned_features.py --exp_folder output_l2_progress/multi-task-setting_val/V1___GTCC_egoprocel
python calculate_action_means.py --exp_folder output_l2_progress/multi-task-setting_val/V1___GTCC_egoprocel

python generate_aligned_features.py --exp_folder output_l2_progress/multi-task-setting_val/V1___tcc_egoprocel
python calculate_action_means.py --exp_folder output_l2_progress/multi-task-setting_val/V1___tcc_egoprocel

python generate_aligned_features.py --exp_folder output_l2_progress/multi-task-setting_val/V1___VAVA_egoprocel
python calculate_action_means.py --exp_folder output_l2_progress/multi-task-setting_val/V1___VAVA_egoprocel

python generate_aligned_features.py --exp_folder output_l2_progress/multi-task-setting_val/V1___LAV_egoprocel
python calculate_action_means.py --exp_folder output_l2_progress/multi-task-setting_val/V1___LAV_egoprocel

#############################################
# output_learnable_progress (NO preprocessing needed!)
#############################################
# Skip - these use ProgressHead directly

#############################################
# output_4fps_val (4fps baseline models)
#############################################

python generate_aligned_features.py --exp_folder output_4fps_val/multi-task-setting_val/V1___GTCC_egoprocel.4fps
python calculate_action_means.py --exp_folder output_4fps_val/multi-task-setting_val/V1___GTCC_egoprocel.4fps

python generate_aligned_features.py --exp_folder output_4fps_val/multi-task-setting_val/V1___tcc_egoprocel.4fps
python calculate_action_means.py --exp_folder output_4fps_val/multi-task-setting_val/V1___tcc_egoprocel.4fps

python generate_aligned_features.py --exp_folder output_4fps_val/multi-task-setting_val/V1___VAVA_egoprocel.4fps
python calculate_action_means.py --exp_folder output_4fps_val/multi-task-setting_val/V1___VAVA_egoprocel.4fps

python generate_aligned_features.py --exp_folder output_4fps_val/multi-task-setting_val/V1___LAV_egoprocel.4fps
python calculate_action_means.py --exp_folder output_4fps_val/multi-task-setting_val/V1___LAV_egoprocel.4fps

# V2 GTCC if exists
python generate_aligned_features.py --exp_folder output_4fps_val/multi-task-setting_val/V2___GTCC_egoprocel.4fps
python calculate_action_means.py --exp_folder output_4fps_val/multi-task-setting_val/V2___GTCC_egoprocel.4fps
```

---

## Part 7: Evaluation Commands (Run After Prerequisites)

```bash
cd /vision/anishn/GTCC_CVPR2024

#############################################
# output_val (1fps baseline models)
#############################################
python eval.py -f output_val/multi-task-setting_val/Vtest1___GTCC_egoprocel --level action
python eval.py -f output_val/multi-task-setting_val/Vtest1___tcc_egoprocel --level action
python eval.py -f output_val/multi-task-setting_val/VtestLavLoss1___LAV_egoprocel --level action
python eval.py -f output_val/multi-task-setting_val/VtestVavaLoss1___VAVA_egoprocel --level action

#############################################
# output_l2_progress (L2 Progress Loss)
#############################################
python eval.py -f output_l2_progress/multi-task-setting_val/V1___GTCC_egoprocel --level action
python eval.py -f output_l2_progress/multi-task-setting_val/V1___tcc_egoprocel --level action
python eval.py -f output_l2_progress/multi-task-setting_val/V1___VAVA_egoprocel --level action
python eval.py -f output_l2_progress/multi-task-setting_val/V1___LAV_egoprocel --level action

#############################################
# output_learnable_progress (NO prerequisites needed!)
#############################################
python eval.py -f output_learnable_progress/multi-task-setting_val/V1___GTCC_egoprocel --level action
python eval.py -f output_learnable_progress/multi-task-setting_val/V1___tcc_egoprocel --level action
python eval.py -f output_learnable_progress/multi-task-setting_val/V1___VAVA_egoprocel --level action
python eval.py -f output_learnable_progress/multi-task-setting_val/V1___LAV_egoprocel --level action

#############################################
# output_4fps_val (4fps baseline models)
#############################################
python eval.py -f output_4fps_val/multi-task-setting_val/V1___GTCC_egoprocel.4fps --level action
python eval.py -f output_4fps_val/multi-task-setting_val/V2___GTCC_egoprocel.4fps --level action
python eval.py -f output_4fps_val/multi-task-setting_val/V1___tcc_egoprocel.4fps --level action
python eval.py -f output_4fps_val/multi-task-setting_val/V1___VAVA_egoprocel.4fps --level action
python eval.py -f output_4fps_val/multi-task-setting_val/V1___LAV_egoprocel.4fps --level action

#############################################
# Video-Level Evaluation (Optional)
#############################################
python eval.py -f output_l2_progress/multi-task-setting_val/V1___GTCC_egoprocel --level video
```

---

## Output Locations

Results saved to:
```
{model_folder}/EVAL_action_level/{checkpoint_name}/OnlineGeoProgressError.csv
{model_folder}/EVAL_video_level/{checkpoint_name}/OnlineGeoProgressError.csv
```

---

## Verification

```bash
# Check that evaluation uses correct test set size (should be 48 total, distributed across tasks)
# Watch for log messages like: "12 videos in test set for BaconAndEggs.egtea"

# View results
cat output_l2_progress/multi-task-setting_val/V1___GTCC_egoprocel/EVAL_action_level/*/OnlineGeoProgressError.csv
```

---

## Summary of Required Actions

| Step | Action | Status |
|------|--------|--------|
| 1 | Update eval.py to use `jsondataset_from_splits` | **DONE** |
| 2 | Remove whitelist from evaluation_action_level.py | **DONE** |
| 3 | Add 4fps detection to evaluation_action_level.py | **DONE** |
| 4 | Update evaluation_action_level.py to load action_means.json from exp folder | **TODO** |
| 5 | Update `generate_aligned_features.py` to store in exp folder | **TODO** |
| 6 | Update `calculate_action_means.py` to store in exp folder | **TODO** |
| 7 | Generate aligned features for all models (Part 6 commands) | After steps 4-6 |
| 8 | Generate action means for all models (Part 6 commands) | After step 7 |
| 9 | Run evaluations (Part 7 commands) | After step 8 |

---

## Critical Files to Modify

1. `/vision/anishn/GTCC_CVPR2024/eval.py` - **DONE**
2. `/vision/anishn/GTCC_CVPR2024/utils/evaluation_action_level.py`:
   - Remove whitelist filtering - **DONE**
   - Add 4fps path detection - **DONE**
   - Load action_means.json from experiment folder - **TODO**
3. `/vision/anishn/GTCC_CVPR2024/generate_aligned_features.py` - Update to store in exp folder
4. `/vision/anishn/GTCC_CVPR2024/calculate_action_means.py` - Update to store in exp folder

### Can Delete After Updates:
- `generate_gtcc_aligned_features.py`, `generate_gtcc_aligned_features_4fps.py`
- `generate_tcc_aligned_features.py`, `generate_tcc_aligned_features_4fps.py`
- `generate_vava_aligned_features.py`, `generate_vava_aligned_features_4fps.py`
- `generate_lav_aligned_features.py`, `generate_lav_aligned_features_4fps.py`
- `calculate_action_geodesic_means.py`, `calculate_action_geodesic_means_4fps.py`
- Old action means JSON files in project root (gtcc_action_means.json, etc.)
