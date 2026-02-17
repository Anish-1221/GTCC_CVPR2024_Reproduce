# Plan: Train 4fps Models (GTCC, TCC, LAV, VAVA) with Validation

## Overview
Train all 4 alignment models (GTCC, TCC, LAV, VAVA) at 4fps on single GPUs with the new validation code and data splits. Output to `output_4fps_val`.

## Understanding
- **New train/test/val splits**: Pre-defined stratified splits (75% train, 10% val, 15% test) loaded from `/vision/anishn/GTCC_CVPR2024/data_splits.json`
- **Validation code**: Built into `alignment_training_loop.py` - runs validation after each epoch, saves best model based on validation loss (`best_model.pt`)
- **Available GPUs**: 3, 4, 5, 6, 7 (5 GPUs, 4 models - can run all in parallel)

---

## Part 0: Pre-Training Fixes (Required for Evaluation to Work)

Since training now saves only `best_model.pt` (instead of `epoch-50.pt`), we need to update checkpoint loading in two places so evaluation works after training.

### Fix 1: Update `ckpt_save.py` for eval.py

**File**: `/vision/anishn/GTCC_CVPR2024/utils/ckpt_save.py`

**Function**: `get_ckpt_MCN()` (lines 28-43)

**Issue**: The function parses `epoch-*.pt` filenames. When only `best_model.pt` exists, `int('best_model')` crashes.

**Fix**: Add logic to prefer `best_model.pt`:
```python
def get_ckpt_MCN(folder, num_heads, device, dropout=False):
    # First check for best_model.pt (from validation-based training)
    best_model_path = folder + '/ckpt/best_model.pt'
    if os.path.exists(best_model_path):
        best_ckpt_file = best_model_path
        ckpt_handle = "best_model"
    else:
        # Fallback: Sort checkpoints numerically by epoch number
        ckpts = glob.glob(folder + f'/ckpt/*')
        ckpts = sorted(ckpts, key=lambda x: int(x.split('epoch-')[-1].split('.')[0]))
        best_ckpt_file = ckpts[-1]
        ckpt_handle = ".".join(best_ckpt_file.split('/')[-1].split('.')[:-1])

    try:
        model, _, epoch, _, _ = ckpt_restore_mprong(
            best_ckpt_file,
            num_heads=num_heads,
            dropout=dropout,
            device=device
        )
        return model, epoch, ckpt_handle
    except Exception as e:
        return None, None, None
```

**Note**: Add `import os` at the top of the file if not already present.

### Fix 2: Update Aligned Feature Scripts

**Files**: All 4 `generate_*_aligned_features_4fps.py` scripts

**Change checkpoint loading logic** (lines ~29-38) to prefer `best_model.pt`:
```python
# First try best_model.pt (from validation-based training)
ckpt_path = os.path.join(EXP_FOLDER, 'ckpt', 'best_model.pt')
if not os.path.exists(ckpt_path):
    # Fallback to epoch-50.pt
    ckpt_path = os.path.join(EXP_FOLDER, 'ckpt', 'epoch-50.pt')
    if not os.path.exists(ckpt_path):
        # Final fallback: find any epoch checkpoint
        ckpt_files = sorted(glob.glob(os.path.join(EXP_FOLDER, 'ckpt', 'epoch-*.pt')))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoints found in {EXP_FOLDER}/ckpt/")
        ckpt_path = ckpt_files[-1]
```

---

## Part 1: Training Commands

### Environment Setup (for each terminal/GPU)
```bash
cd /vision/anishn/GTCC_CVPR2024
conda activate /vision/anishn/envs/vidal
export DATASET_PATH=/vision/anishn/GTCC_Data_Processed_4fps
export OUTPUT_PATH=/vision/anishn/GTCC_CVPR2024/output_4fps_val
export JSON_DPATH=/vision/anishn/GTCC_CVPR2024/dset_jsons
```

### Training Commands (run in parallel on separate GPUs)

**GPU 3 - GTCC:**
```bash
CUDA_VISIBLE_DEVICES=3 python multitask_train.py 1 --gtcc --ego --resnet --mcn -bs 2 -ep 50 -m "4fps"
```

**GPU 4 - TCC:**
```bash
CUDA_VISIBLE_DEVICES=4 python multitask_train.py 1 --tcc --ego --resnet --mcn -bs 2 -ep 50 -m "4fps"
```

**GPU 5 - LAV:**
```bash
CUDA_VISIBLE_DEVICES=5 python multitask_train.py 1 --lav --ego --resnet --mcn -bs 2 -ep 50 -m "4fps"
```

**GPU 6 - VAVA:**
```bash
CUDA_VISIBLE_DEVICES=6 python multitask_train.py 1 --vava --ego --resnet --mcn -bs 2 -ep 50 -m "4fps"
```

### Expected Output Structure
```
/vision/anishn/GTCC_CVPR2024/output_4fps_val/
└── multi-task-setting/
    ├── V1___GTCC_egoprocel.4fps/
    │   ├── config.json
    │   ├── ckpt/
    │   │   └── best_model.pt          # Saved when val loss improves
    │   ├── train_loss_epochlevel.png
    │   ├── val_loss_epochlevel.png
    │   └── train_val_loss.png
    ├── V1___tcc_egoprocel.4fps/
    ├── V1___LAV_egoprocel.4fps/
    └── V1___VAVA_egoprocel.4fps/
```

---

## Part 2: Post-Training Steps

### 2.1 Generate Aligned Features (4fps)
After training completes, update the EXP_FOLDER paths in aligned feature scripts (checkpoint loading fix was already done in Part 0):

**Files to update (line ~15):**
- `generate_gtcc_aligned_features_4fps.py` → `output_4fps_val/.../V1___GTCC_egoprocel.4fps`
- `generate_tcc_aligned_features_4fps.py` → `output_4fps_val/.../V1___tcc_egoprocel.4fps`
- `generate_vava_aligned_features_4fps.py` → `output_4fps_val/.../V1___VAVA_egoprocel.4fps`
- `generate_lav_aligned_features_4fps.py` → `output_4fps_val/.../V1___LAV_egoprocel.4fps`

**Run scripts:**
```bash
python generate_gtcc_aligned_features_4fps.py
python generate_tcc_aligned_features_4fps.py
python generate_vava_aligned_features_4fps.py
python generate_lav_aligned_features_4fps.py
```

### 2.2 Calculate Action Geodesic Means (4fps)
```bash
python calculate_action_geodesic_means_4fps.py
python calculate_tcc_action_geodesic_means_4fps.py
python calculate_vava_action_geodesic_means_4fps.py
python calculate_lav_action_geodesic_means_4fps.py
```

**Output files:**
- `gtcc_action_means_4fps.json`
- `tcc_action_means_4fps.json`
- `vava_action_means_4fps.json`
- `lav_action_means_4fps.json`

### 2.3 Evaluation Commands

**Video-Level Evaluation:**
```bash
python eval.py -f /vision/anishn/GTCC_CVPR2024/output_4fps_val/multi-task-setting/V1___GTCC_egoprocel.4fps --level video
python eval.py -f /vision/anishn/GTCC_CVPR2024/output_4fps_val/multi-task-setting/V1___tcc_egoprocel.4fps --level video
python eval.py -f /vision/anishn/GTCC_CVPR2024/output_4fps_val/multi-task-setting/V1___LAV_egoprocel.4fps --level video
python eval.py -f /vision/anishn/GTCC_CVPR2024/output_4fps_val/multi-task-setting/V1___VAVA_egoprocel.4fps --level video
```

**Action-Level Evaluation:**
```bash
python eval.py -f /vision/anishn/GTCC_CVPR2024/output_4fps_val/multi-task-setting/V1___GTCC_egoprocel.4fps --level action
python eval.py -f /vision/anishn/GTCC_CVPR2024/output_4fps_val/multi-task-setting/V1___tcc_egoprocel.4fps --level action
python eval.py -f /vision/anishn/GTCC_CVPR2024/output_4fps_val/multi-task-setting/V1___LAV_egoprocel.4fps --level action
python eval.py -f /vision/anishn/GTCC_CVPR2024/output_4fps_val/multi-task-setting/V1___VAVA_egoprocel.4fps --level action
```

---

## Part 3: Critical Files

### Files to Modify (EXP_FOLDER paths to output_4fps_val)
| File | Line | Change |
|------|------|--------|
| `generate_gtcc_aligned_features_4fps.py` | ~15 | `output_4fps` -> `output_4fps_val` |
| `generate_tcc_aligned_features_4fps.py` | ~15 | `output_4fps` -> `output_4fps_val` |
| `generate_vava_aligned_features_4fps.py` | ~15 | `output_4fps` -> `output_4fps_val` |
| `generate_lav_aligned_features_4fps.py` | ~15 | `output_4fps` -> `output_4fps_val` |

### Key Existing Files (no changes needed)
- `/vision/anishn/GTCC_CVPR2024/multitask_train.py` - Training entry point with validation
- `/vision/anishn/GTCC_CVPR2024/models/alignment_training_loop.py` - Validation loop
- `/vision/anishn/GTCC_CVPR2024/data_splits.json` - Pre-defined stratified splits
- `/vision/anishn/GTCC_CVPR2024/eval.py` - Evaluation with `--level` argument
- `/vision/anishn/GTCC_CVPR2024/utils/evaluation_action_level.py` - Action-level metrics

---

## Part 4: Verification

### During Training
- Check that `val_loss_epochlevel.png` is being generated
- Check that `best_model.pt` is being saved in the ckpt folder
- Monitor GPU usage with `nvidia-smi`

### After Training
1. Verify checkpoints exist:
```bash
ls /vision/anishn/GTCC_CVPR2024/output_4fps_val/multi-task-setting/V1___*_egoprocel.4fps/ckpt/best_model.pt
```

2. Check validation loss plots in each experiment folder

3. After evaluation, check OGPE metrics in output CSV files

---

## Summary

| Model | GPU | Command Flag | Expected Output Folder |
|-------|-----|--------------|----------------------|
| GTCC | 3 | `--gtcc --ego --resnet --mcn` | `V1___GTCC_egoprocel.4fps` |
| TCC | 4 | `--tcc --ego --resnet --mcn` | `V1___tcc_egoprocel.4fps` |
| LAV | 5 | `--lav --ego --resnet --mcn` | `V1___LAV_egoprocel.4fps` |
| VAVA | 6 | `--vava --ego --resnet --mcn` | `V1___VAVA_egoprocel.4fps` |
