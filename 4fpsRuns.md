# Plan: Train and Evaluate GTCC/TCC/VAVA/LAV at 4 FPS

## Overview
Train GTCC, TCC, VAVA, and LAV alignment models using 4 fps data (same pipeline as 1fps), evaluate them at both action_level and video_level, and evaluate already-trained 4fps ProTAS models.

## Key Configuration Changes
- **Batch size**: 2
- **GPU**: Single GPU only (no DDP)
- **Disable pairwise batch skipping**: Preserve all training data
- **Output folders**: New directories (no overwriting)
- **Architecture**: `resnet50` (same as 1fps pipeline, uses raw frames)

---

## Part 1: Code Modifications

### 1.1 Disable Pairwise Batch Skipping
**File**: `/vision/anishn/GTCC_CVPR2024/models/alignment_training_loop.py`

**Lines 185-191**: Raise thresholds to effectively disable skipping:
```python
MAX_PAIRWISE_THRESHOLDS = {
    'LAV': 999_999_999_999,      # Disabled
    'GTCC': 999_999_999_999,     # Disabled
    'tcc': 999_999_999_999,      # Disabled
    'VAVA': 999_999_999_999,     # Disabled
    'default': 999_999_999_999,  # Disabled
}
```

### 1.2 Create 4fps ProTAS Evaluation Scripts

**Create**: `/vision/anishn/GTCC_CVPR2024/eval_protas_action_level_4fps.py`
- Copy from `eval_protas_action_level.py`
- Change line 33: `PROTAS_BASE = '/vision/anishn/ProTAS/data_4fps/'`
- Change line 34: `OUTPUT_FOLDER = '/vision/anishn/GTCC_CVPR2024/output_4fps/multi-task-setting/protas_eval_egoprocel_act_level_4fps'`
- Update SUBSET_CONFIGS (lines 38-63) with 4fps model paths:
  - `egoprocel_subset1_S`: `/u/anishn/models/egoprocel_subset1_S_4fps/egoprocel_subset1_S/split_1/epoch-50.model`
  - `egoprocel_subset2_OP_P`: `/u/anishn/models/egoprocel_subset2_OP_P_4fps/egoprocel_subset2_OP_P/split_1/epoch-50.model`
  - `egoprocel_subset3_tent`: `/u/anishn/models/egoprocel_subset3_tent_4fps/egoprocel_subset3_tent/split_1/epoch-50.model`
  - `egoprocel_subset4_numbers`: `/u/anishn/models/egoprocel_subset4_numbers_4fps/egoprocel_subset4_numbers/split_1/epoch-50.model`
  - `egoprocel_subset5_head`: `/u/anishn/models/egoprocel_subset5_head_4fps/egoprocel_subset5_head/split_1/epoch-50.model`
- Update graph_path entries to use `/vision/anishn/ProTAS/data_4fps/`

**Create**: `/vision/anishn/GTCC_CVPR2024/eval_protas_video_level_4fps.py`
- Copy from `eval_protas_video_level.py`
- Same path modifications as above

### 1.3 Create 4fps Aligned Feature Generation Scripts

**Create**: `/vision/anishn/GTCC_CVPR2024/generate_gtcc_aligned_features_4fps.py`
- Copy from `generate_gtcc_aligned_features.py`
- Line 15: `EXP_FOLDER = '/vision/anishn/GTCC_CVPR2024/output_4fps/multi-task-setting/V1___GTCC_egoprocel.4fps'`
- Line 17: `INPUT_FEAT_DIR = '/vision/anishn/GTCC_Data_Processed_4fps/egoprocel/frames/'`
- Line 18: `OUTPUT_DIR = '/vision/anishn/GTCC_Data_Processed_4fps/egoprocel/aligned_features/'`

Similar for TCC, VAVA, LAV aligned feature generation (update paths accordingly).

### 1.4 Create 4fps Action Geodesic Means Calculation Scripts

**Create**: `/vision/anishn/GTCC_CVPR2024/calculate_action_geodesic_means_4fps.py`
- Copy from `calculate_action_geodesic_means.py`
- Update aligned features path to `/vision/anishn/GTCC_Data_Processed_4fps/egoprocel/aligned_features/`
- Output: `/vision/anishn/GTCC_CVPR2024/gtcc_action_means_4fps.json`

**Create**: `/vision/anishn/GTCC_CVPR2024/calculate_tcc_action_geodesic_means_4fps.py`
- Copy from `calculate_tcc_action_geodesic_means.py`
- Update aligned features path to `/vision/anishn/GTCC_Data_Processed_4fps/egoprocel/tcc_aligned_features/`
- Output: `/vision/anishn/GTCC_CVPR2024/tcc_action_means_4fps.json`

**Create**: `/vision/anishn/GTCC_CVPR2024/calculate_vava_action_geodesic_means_4fps.py`
- Copy from `calculate_vava_action_geodesic_means.py`
- Update aligned features path to `/vision/anishn/GTCC_Data_Processed_4fps/egoprocel/vava_aligned_features/`
- Output: `/vision/anishn/GTCC_CVPR2024/vava_action_means_4fps.json`

**Create**: `/vision/anishn/GTCC_CVPR2024/calculate_lav_action_geodesic_means_4fps.py`
- Copy from `calculate_lav_action_geodesic_means.py`
- Update aligned features path to `/vision/anishn/GTCC_Data_Processed_4fps/egoprocel/lav_aligned_features/`
- Output: `/vision/anishn/GTCC_CVPR2024/lav_action_means_4fps.json`

### 1.5 Modify eval.py for --level Argument (GTCC/TCC/VAVA/LAV Evaluation)

**Modify**: `/vision/anishn/GTCC_CVPR2024/eval.py`

Instead of creating separate eval scripts, we modified the existing eval.py to accept a `--level` argument:

**Changes made** (all marked with `[4FPS MODIFICATION]` comments):

1. **Lines 217-244**: Added early argument parsing to conditionally import evaluation module
   - `--level video`: imports from `utils.evaluation` (progress 0->1 across entire video)
   - `--level action`: imports from `utils.evaluation_action_level` (progress 0->1 per action segment)

2. **Line 271**: Modified output folder to include level suffix
   ```python
   out_folder = f'{folder_to_test}/EVAL_{eval_level}_level'
   ```
   - Video-level: outputs to `EVAL_video_level/`
   - Action-level: outputs to `EVAL_action_level/`

3. **Lines 332-335**: Added `--level` argument to argparser
   ```python
   parser.add_argument('--level', choices=['video', 'action'], default='action',
                       help='Evaluation level: video (0->1 per video) or action (0->1 per action segment)')
   ```

4. **Line 376**: Pass `eval_level` to `begin_eval_loop_over_tasks()`

---

## Part 2: New Output Directories

```
/vision/anishn/GTCC_CVPR2024/output_4fps/
  multi-task-setting/
    # Training outputs (each contains EVAL_video_level/ and EVAL_action_level/)
    V1___GTCC_egoprocel.4fps/
      ckpt/
      EVAL_video_level/    # Created by: python eval.py -f ... --level video
      EVAL_action_level/   # Created by: python eval.py -f ... --level action
    V1___tcc_egoprocel.4fps/
      ckpt/
      EVAL_video_level/
      EVAL_action_level/
    V1___VAVA_egoprocel.4fps/
      ckpt/
      EVAL_video_level/
      EVAL_action_level/
    V1___LAV_egoprocel.4fps/
      ckpt/
      EVAL_video_level/
      EVAL_action_level/

    # ProTAS evaluation (separate folders)
    protas_eval_egoprocel_video_level_4fps/
    protas_eval_egoprocel_act_level_4fps/

/vision/anishn/GTCC_Data_Processed_4fps/egoprocel/
  aligned_features/          # GTCC-aligned (to create)
  tcc_aligned_features/      # TCC-aligned (to create)
  vava_aligned_features/     # VAVA-aligned (to create)
  lav_aligned_features/      # LAV-aligned (to create)

# Action geodesic means JSON files
/vision/anishn/GTCC_CVPR2024/
  gtcc_action_means_4fps.json
  tcc_action_means_4fps.json
  vava_action_means_4fps.json
  lav_action_means_4fps.json
```

---

## Part 3: Training Commands

### Environment Setup
```bash
export CUDA_VISIBLE_DEVICES=0
export DATASET_PATH=/vision/anishn/GTCC_Data_Processed_4fps
export OUTPUT_PATH=/vision/anishn/GTCC_CVPR2024/output_4fps
export JSON_DPATH=/vision/anishn/GTCC_CVPR2024/dset_jsons
# Do NOT set LOCAL_RANK (ensures single GPU mode)
cd /vision/anishn/GTCC_CVPR2024
conda activate /vision/anishn/envs/vidal
```

### 3.1 Train GTCC (4fps)
```bash
python multitask_train.py 1 --gtcc --ego --resnet --mcn -bs 2 -ep 50 -m "4fps"
```
Output: `V1___GTCC_egoprocel.4fps`

### 3.2 Train TCC (4fps)
```bash
python multitask_train.py 1 --tcc --ego --resnet --mcn -bs 2 -ep 50 -m "4fps"
```

### 3.3 Train VAVA (4fps)
```bash
python multitask_train.py 1 --vava --ego --resnet --mcn -bs 2 -ep 50 -m "4fps"
```

### 3.4 Train LAV (4fps)
```bash
python multitask_train.py 1 --lav --ego --resnet --mcn -bs 2 -ep 50 -m "4fps"
```

**Note**: Using `--resnet` because this is the same architecture used for 1fps training. Both 1fps and 4fps data contain raw (T, 1024, 14, 14) spatial features in `frames/` directory.

---

## Part 4: Evaluation Commands

### 4.1 Generate Aligned Features (after training)
```bash
python generate_gtcc_aligned_features_4fps.py
python generate_tcc_aligned_features_4fps.py
python generate_vava_aligned_features_4fps.py
python generate_lav_aligned_features_4fps.py
```

### 4.2 Generate Action Geodesic Means JSON (after aligned features)
```bash
python calculate_action_geodesic_means_4fps.py
python calculate_tcc_action_geodesic_means_4fps.py
python calculate_vava_action_geodesic_means_4fps.py
python calculate_lav_action_geodesic_means_4fps.py
```

### 4.3 GTCC/TCC/VAVA/LAV Evaluation

Use the modified `eval.py` with `--level` argument to switch between video-level and action-level evaluation:

**Video-Level Evaluation** (progress 0->1 across entire video):
- Uses `utils.evaluation` module
- Outputs to `{model_folder}/EVAL_video_level/`
```bash
python eval.py -f /vision/anishn/GTCC_CVPR2024/output_4fps/multi-task-setting/V1___GTCC_egoprocel.4fps --level video
python eval.py -f /vision/anishn/GTCC_CVPR2024/output_4fps/multi-task-setting/V1___tcc_egoprocel.4fps --level video
python eval.py -f /vision/anishn/GTCC_CVPR2024/output_4fps/multi-task-setting/V1___VAVA_egoprocel.4fps --level video
python eval.py -f /vision/anishn/GTCC_CVPR2024/output_4fps/multi-task-setting/V1___LAV_egoprocel.4fps --level video
```

**Action-Level Evaluation** (progress 0->1 per action segment):
- Uses `utils.evaluation_action_level` module
- Outputs to `{model_folder}/EVAL_action_level/`
```bash
python eval.py -f /vision/anishn/GTCC_CVPR2024/output_4fps/multi-task-setting/V1___GTCC_egoprocel.4fps --level action
python eval.py -f /vision/anishn/GTCC_CVPR2024/output_4fps/multi-task-setting/V1___tcc_egoprocel.4fps --level action
python eval.py -f /vision/anishn/GTCC_CVPR2024/output_4fps/multi-task-setting/V1___VAVA_egoprocel.4fps --level action
python eval.py -f /vision/anishn/GTCC_CVPR2024/output_4fps/multi-task-setting/V1___LAV_egoprocel.4fps --level action
```

### 4.4 ProTAS 4fps Evaluation (Already Trained Models)

```bash
# Action-level (outputs to protas_eval_egoprocel_act_level_4fps/)
python eval_protas_action_level_4fps.py

# Video-level (outputs to protas_eval_egoprocel_video_level_4fps/)
python eval_protas_video_level_4fps.py
```

---

## Part 5: Order of Operations

1. **Modify pairwise skipping** in `alignment_training_loop.py` (lines 185-191)
2. **Create all 4fps scripts** (eval, aligned features, action means)
3. **Train models** (GTCC, TCC, VAVA, LAV) - run sequentially
4. **Generate aligned features** (4 scripts - gtcc, tcc, vava, lav)
5. **Generate action geodesic means JSON** (4 scripts - gtcc, tcc, vava, lav)
6. **Evaluate GTCC/TCC/VAVA/LAV** at video_level (separate output folders)
7. **Evaluate GTCC/TCC/VAVA/LAV** at action_level (separate output folders)
8. **Evaluate ProTAS 4fps models** at video_level and action_level

---

## Part 6: Files to Modify/Create

### Modify:
| File | Lines | Change |
|------|-------|--------|
| `/vision/anishn/GTCC_CVPR2024/models/alignment_training_loop.py` | 185-191 | Raise thresholds to 999_999_999_999 |
| `/vision/anishn/GTCC_CVPR2024/eval.py` | 217-244, 271, 332-335, 376 | Added `--level` argument for video/action evaluation (see section 1.5) |

### Create (by copying and modifying):
| New File | Template | Key Changes |
|----------|----------|-------------|
| **Aligned Feature Generation** | | |
| `generate_gtcc_aligned_features_4fps.py` | `generate_gtcc_aligned_features.py` | Lines 15,17,18: 4fps paths |
| `generate_tcc_aligned_features_4fps.py` | `generate_tcc_aligned_features.py` | 4fps paths |
| `generate_vava_aligned_features_4fps.py` | `generate_vava_aligned_features.py` | 4fps paths |
| `generate_lav_aligned_features_4fps.py` | `generate_lav_aligned_features.py` | 4fps paths |
| **Action Means JSON** | | |
| `calculate_action_geodesic_means_4fps.py` | `calculate_action_geodesic_means.py` | 4fps aligned features path, output json |
| `calculate_tcc_action_geodesic_means_4fps.py` | `calculate_tcc_action_geodesic_means.py` | 4fps paths |
| `calculate_vava_action_geodesic_means_4fps.py` | `calculate_vava_action_geodesic_means.py` | 4fps paths |
| `calculate_lav_action_geodesic_means_4fps.py` | `calculate_lav_action_geodesic_means.py` | 4fps paths |
| **ProTAS Evaluation Scripts** | | |
| `eval_protas_action_level_4fps.py` | `eval_protas_action_level.py` | Lines 33-34, 38-63: 4fps paths |
| `eval_protas_video_level_4fps.py` | `eval_protas_video_level.py` | Lines 33-34, 38-63: 4fps paths |

---

## Part 7: Verification

After each training run:
1. Check that checkpoint exists: `ls output_4fps/multi-task-setting/V1___*_egoprocel.4fps/ckpt/epoch-50.pt`
2. Check training log for batch statistics (skipped batches should be minimal)

After evaluation:
1. Check OGPE metrics in output CSV files
2. Compare with 1fps baselines

---

## Data Summary

| Dataset | Path | Status |
|---------|------|--------|
| GTCC 4fps frames | `/vision/anishn/GTCC_Data_Processed_4fps/egoprocel/frames/` | Exists (raw T,1024,14,14) |
| GTCC 1fps frames | `/vision/anishn/GTCC_Data_Processed_1fps/egoprocel/frames/` | Reference (same format) |
| ProTAS 4fps features | `/vision/anishn/ProTAS/data_4fps/` | Exists (2048-dim) |
| ProTAS 4fps models | `/u/anishn/models/egoprocel_*_4fps/` | Exists (5 subsets trained) |
| GTCC 4fps aligned | To be generated | After training |

## Reference: Existing 1fps Config
- **Architecture**: `resnet50`
- **Output path**: `/vision/anishn/GTCC_CVPR2024/output/multi-task-setting/V1___GTCC_egoprocel/`
- **Data folder**: `/vision/anishn/GTCC_Data_Processed_1fps/egoprocel`
