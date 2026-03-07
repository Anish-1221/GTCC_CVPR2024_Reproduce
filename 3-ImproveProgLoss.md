# Plan: Use Pre-Alignment Features for Progress Head

## Context

Dense supervision (per-frame MSE) didn't fix the progress head. GRU with dense loss still produces identical predictions across all videos:
- Frame 1: ~0.157, Frame 2: ~0.45, Frame 3: ~0.71, then plateau at ~0.73
- Same curve for BaconAndEggs, Sandwich, every task/video

**Root cause**: The 128-d aligned embeddings are trained to be temporally consistent ACROSS videos of the same action. The alignment loss makes "frame 5 of cutting in video A" look like "frame 5 of cutting in video B". So the progress head sees the same temporal pattern everywhere and learns a single position-dependent curve.

---

## Phase 1: 512-d Intermediate Features — FAILED

Tried tapping the 512-d post-Conv3D/MaxPool3D features from inside Resnet50Encoder (before the final linear → 128-d). These features are still influenced by the alignment-trained Conv3D layers, so they exhibit the same temporal homogeneity. Predictions were identical across all videos.

**Code changes were implemented** (`--progress_features raw` flag, `_get_progress_features()` helper in loss_entry.py, raw_features pass-through in model). These remain in the codebase and are reused for Phase 2.

---

## Phase 2: 2048-d Pre-Computed Features from Disk

Use 2048-d features extracted from a pre-trained ResNet backbone, stored on disk in the ProTAS data pipeline. These features have **zero alignment training influence** and retain full video-specific information.

**Data verification (confirmed)**:
- 2048-d features at 1fps exist at `/vision/anishn/ProTAS/data_1fps/` across 5 subset folders
- All 325 GTCC videos have matching features with identical T dimensions
- Same naming convention (e.g., `S07_Brownie_7151062-1103.npy`)

### Step 0: Copy 2048-d Features (one-time data prep)

Copy 325 feature files from 5 ProTAS subset folders into a separate directory:

```
/vision/anishn/GTCC_Data_Processed_1fps_2048/egoprocel/features/  # (T, 2048) — NEW
```

Kept separate from the existing `GTCC_Data_Processed_1fps/` directory.

Source folders:
- `ProTAS/data_1fps/egoprocel_subset1_S/features/` — CMU videos (S##_*)
- `ProTAS/data_1fps/egoprocel_subset2_OP_P/features/` — EGTEA videos (OP*/P*)
- `ProTAS/data_1fps/egoprocel_subset3_tent/features/` — Tent videos
- `ProTAS/data_1fps/egoprocel_subset4_numbers/features/` — Meccano videos (####)
- `ProTAS/data_1fps/egoprocel_subset5_head/features/` — PC assembly videos (Head_##)

### Step 1: Update progress head input_dim (512 → 2048)

**File**: `models/model_multiprong.py` — change `progress_input_dim = 512` to `2048` when features == 'raw'

### Step 2: Inject 2048-d features in training loop

**File**: `models/alignment_training_loop.py`

After `output_dict = model(inputs)`, load 2048-d features from `{DATAFOLDER}/features/{handle}.npy` and inject into `output_dict['raw_features']`. Loss functions already pick these up via `_get_progress_features()`.

### Step 3: Inject 2048-d features in extract_progress.py

Same approach: load from disk using video handle instead of relying on model's 512-d raw_features.

### Files to Modify

| File | Change |
|------|--------|
| `models/model_multiprong.py` | `input_dim = 2048` when features == 'raw' (was 512) |
| `models/alignment_training_loop.py` | Load 2048-d features from disk, inject into output_dict |
| `extract_progress.py` | Load 2048-d features from disk for extraction |

### Verification

```bash
CUDA_VISIBLE_DEVICES=6 python multitask_train.py 1 --gtcc --ego --resnet --mcn \
  --progress_loss learnable --progress_lambda 500000.0 \
  --train_progress_only --reinit_progress_head \
  --alignment_checkpoint output_learnable_progress_v6/multi-task-setting_val/V1___GTCC_egoprocel/ckpt/best_model.pt \
  --progress_lr 0.001 --progress_epochs 50 \
  --progress_loss_mode dense --progress_features raw
```

Success criteria:
- **Different** progress curves across different videos (not the same plateau)
- Smooth 0→1 progression that scales with action length
- Values reaching close to 1.0 at action endpoints
