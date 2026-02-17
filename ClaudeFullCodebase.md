# Research Ecosystem Documentation: Egocentric Procedural Task Video Understanding

## 1. Executive Summary

This document provides comprehensive documentation for an integrated research ecosystem spanning three codebases focused on **egocentric procedural task video understanding**:

| Codebase | Purpose | Primary Paper |
|----------|---------|---------------|
| **ProTAS** | Progress-Aware Online Temporal Action Segmentation | CVPR 2024 |
| **GTCC_CVPR2024** | Self-Supervised Video Alignment for Progress Prediction | CVPR 2024 |
| **progress_visualization** | Comparative Visualization Pipeline | Internal Tool |

**Research Lab**: Elhamifar Lab, Northeastern University

**Environment Activation**:
```bash
conda activate /vision/anishn/envs/vidal
```

---

## 2. Research Overview & Goals

### 2.1 Problem Definition: Activity Progress Prediction

Given an egocentric video of a person performing a procedural task (e.g., making coffee, assembling furniture), predict:
- **How far along** they are in the overall task (0% to 100%)
- **Current action** being performed
- **Progress within current action** (0% to 100%)

### 2.2 Problem Definition: Temporal Action Segmentation

Given a video, produce **frame-level predictions** of which action is being performed at each timestep. Key challenges:
- Variable-length actions
- Similar-looking actions (e.g., "add milk" vs "add cream")
- Background/transition frames

### 2.3 Relationship Between Tasks

```
                     ┌──────────────────────┐
                     │   Raw Video Frames   │
                     └──────────┬───────────┘
                                │
                     ┌──────────▼───────────┐
                     │  Feature Extraction  │
                     │  (ResNet50, 2048-d)  │
                     └──────────┬───────────┘
                                │
           ┌────────────────────┼────────────────────┐
           │                    │                    │
   ┌───────▼───────┐   ┌───────▼───────┐   ┌───────▼───────┐
   │     GTCC      │   │     TCC       │   │     LAV       │
   │   (Alignment) │   │   (Baseline)  │   │   (Baseline)  │
   └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
           │                    │                    │
           └────────────────────┼────────────────────┘
                                │
                     ┌──────────▼───────────┐
                     │   Aligned Features   │
                     │     (128-dim)        │
                     └──────────┬───────────┘
                                │
                     ┌──────────▼───────────┐
                     │       ProTAS         │
                     │  (Action + Progress) │
                     └──────────────────────┘
```

---

## 3. ProTAS Codebase Documentation

**Location**: `/vision/anishn/ProTAS/`

### 3.1 Purpose

ProTAS (Progress-Aware Temporal Action Segmentation) performs **online temporal action segmentation** using progress prediction as an auxiliary task. It leverages task graphs to model action dependencies.

### 3.2 Architecture Overview

```
Input Features (2048-d) ──► Stage 1 ──► Stage 2 ──► Stage 3 ──► Stage 4
        │                     │          │          │          │
        │                     ▼          ▼          ▼          ▼
        │               [Action Logits + Progress Predictions]
        │                                                      │
        │                 ┌────────────────────────────────────┘
        │                 │
        │    ┌────────────▼─────────────┐
        └───►│   Task Graph Learner     │
             │  (Predecessor/Successor) │
             └──────────────────────────┘
```

**Key Components** (`model.py`):

| Class | Purpose |
|-------|---------|
| `MultiStageModel` | 4-stage temporal convolutional network |
| `SingleStageModel` | Dilated convolutions + GRU for progress |
| `TaskGraphLearner` | Learns action ordering constraints |
| `ProbabilityProgressFusionModel` | Fuses action probs with progress |
| `DilatedResidualLayer` | Causal/non-causal temporal convolutions |

### 3.3 Data Format

**Directory Structure** (`data_1fps/{dataset}/`):
```
{dataset}/
├── features/           # (T, 2048) numpy arrays per video
│   └── {video_name}.npy
├── groundTruth/        # Frame-level action labels
│   └── {video_name}.txt
├── progress/           # (num_classes, T) progress values
│   └── {video_name}.npy
├── graph/
│   └── graph.pkl       # Task graph (predecessor/successor matrices)
├── splits/
│   ├── train.split1.bundle
│   └── test.split1.bundle
└── mapping.txt         # Action name to index mapping
```

**Feature Format**:
- Shape: `(T, 2048)` or `(2048, T)` depending on dataset
- For EgoProceL: features stored as `(T, 2048)`, transposed during loading

**Ground Truth Format** (`groundTruth/{video}.txt`):
```
background
add_oil
add_oil
heat_pan
heat_pan
...
```

**Progress Format** (`progress/{video}.npy`):
- Shape: `(num_classes, T)`
- Values: 0.0 to 1.0 per action (resets for each action instance)

**Mapping Format** (`mapping.txt`):
```
0 background
1 add_honey_or_chocolate_syrup_to_the_cereal
2 add_hot_water_to_the_kettle_and_place_it_on_the_stove_and_boil_it
...
```

### 3.4 Key Files

| File | Purpose |
|------|---------|
| `main.py` | Training and inference entry point |
| `model.py` | Model architecture (MultiStageModel, TaskGraphLearner) |
| `batch_gen.py` | Data loading and batching |
| `eval.py` | Evaluation metrics (Acc, Edit, F1@{10,25,50}) |
| `extract_features.py` | ResNet50 feature extraction |
| `utils/write_graph.py` | Generate task graphs from annotations |
| `utils/write_progress_values.py` | Generate progress annotations |

### 3.5 Training Commands

```bash
# Standard training with task graph
python main.py \
    --dataset egoprocel_subset2_OP_P \
    --split 1 \
    --exp_id my_experiment \
    --num_epochs 50 \
    --graph \
    --causal

# Key arguments:
#   --dataset: Dataset name (matches folder in data_1fps/)
#   --graph: Enable task graph learning
#   --causal: Use causal convolutions (required for online inference)
#   --learnable_graph: Allow graph weights to be learned
#   --progress_lw: Progress loss weight (default: 1.0)
#   --graph_lw: Graph loss weight (default: 0.1)
```

### 3.6 Inference Commands

```bash
# Offline prediction (uses full video context)
python main.py \
    --action predict \
    --dataset egoprocel_subset2_OP_P \
    --exp_id my_experiment \
    --num_epochs 50

# Online prediction (causal, frame-by-frame)
python main.py \
    --action predict_online \
    --dataset egoprocel_subset2_OP_P \
    --exp_id my_experiment \
    --num_epochs 50 \
    --causal
```

### 3.7 Supported Datasets

| Dataset | Delimiter | Background Class | Feature Transpose |
|---------|-----------|------------------|-------------------|
| coffee, tea, pinwheels, oatmeal, quesadilla | `\|` | `BG` | Yes |
| 50salads | ` ` | `background` | No |
| GTEA, Breakfast | ` ` | `background` | No |
| egoprocel_subset* | `\|` | `BG` | Yes |

### 3.8 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Acc** | Frame-level accuracy (all frames) |
| **Acc-bg** | Frame-level accuracy (excluding background) |
| **Edit** | Levenshtein distance on action sequence |
| **F1@{10,25,50}** | Segment-level F1 at IoU thresholds |

---

## 4. GTCC_CVPR2024 Codebase Documentation

**Location**: `/vision/anishn/GTCC_CVPR2024/`

### 4.1 Purpose

GTCC (Graph-based Temporal Cycle Consistency) performs **self-supervised video alignment** for progress prediction. It learns to align videos of the same task without explicit action labels.

### 4.2 Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   Multi-Prong MCN Model                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Input (2048-d) ──► Base Encoder (StackingEncoder)     │
│                          │                              │
│              ┌───────────┼───────────┐                  │
│              ▼           ▼           ▼                  │
│          [Head 1]    [Head 2]    [Head N]              │
│              │           │           │                  │
│              └───────────┼───────────┘                  │
│                          ▼                              │
│                  Attention Layer                        │
│                          │                              │
│                          ▼                              │
│              Output Embeddings (128-d)                  │
│                                                         │
│  Optional: Dropout Network (for GTCC loss)             │
└─────────────────────────────────────────────────────────┘
```

**Key Components**:

| Component | Location | Purpose |
|-----------|----------|---------|
| `StackingEncoder` | `models/model_singleprong.py` | 1D conv temporal encoder |
| `MultiProngAttDropoutModel` | `models/model_multiprong.py` | Multi-head attention network (MCN) |
| `alignment_training_loop` | `models/alignment_training_loop.py` | Training loop for alignment losses |

### 4.3 Loss Functions

Defined in `utils/loss_functions.py`:

| Loss | Function | Description |
|------|----------|-------------|
| **GTCC** | `GTCC_loss()` | Graph-based TCC with GMM and dropout network |
| **TCC** | `TCC_loss()` | Temporal Cycle Consistency (Dwibedi et al.) |
| **LAV** | `LAV_loss()` | Learning by Aligning Videos (SoftDTW + contrastive) |
| **VAVA** | `VAVA_loss()` | Video Alignment with Variational Attention (OT-based) |

**GTCC Loss Key Parameters**:
```python
GTCC_loss(
    sequences,          # List of embedded video sequences
    n_components=1,     # GMM components (K in paper)
    gamma=1,            # Dropout decay rate
    delta=1,            # Margin ratio for neighbor window
    dropouts=None,      # Dropout network outputs
    softmax_temp=1,     # Temperature for attention
    alignment_variance=0,
    max_gmm_iters=8,
    epoch=0
)
```

### 4.4 Data Format

**JSON Metadata** (`dset_jsons/{dataset}.json`):
```json
{
    "BaconAndEggs.egtea": {
        "handles": [
            "OP01-R03-BaconAndEggs",
            "OP02-R03-BaconAndEggs",
            ...
        ],
        "hdl_actions": [
            ["0", "30", "0", "30", ...],  # Action IDs per frame
            ...
        ]
    },
    ...
}
```

**Data Directory Structure**:
```
GTCC_Data_Processed_1fps/
└── egoprocel/
    ├── frames/                    # Raw features (T, 2048) or (T, 1024, 14, 14)
    │   └── {video_name}.npy
    ├── aligned_features/          # GTCC-aligned (T, 128)
    │   └── {video_name}.npy
    ├── tcc_aligned_features/      # TCC-aligned
    ├── vava_aligned_features/     # VAVA-aligned
    └── lav_aligned_features/      # LAV-aligned
```

### 4.5 Key Files

| File | Purpose |
|------|---------|
| `singletask_train.py` | Single-task training (one task at a time) |
| `multitask_train.py` | Multi-task training (all tasks jointly) |
| `eval.py` | Evaluation entry point |
| `generate_gtcc_aligned_features.py` | Generate aligned features for ProTAS |
| `configs/entry_config.py` | Configuration management |
| `configs/generic_config.py` | Default hyperparameters |
| `utils/evaluation.py` | Evaluation metrics (OPE, KT, PC, etc.) |
| `utils/loss_functions.py` | GTCC, TCC, LAV, VAVA loss implementations |
| `utils/tensorops.py` | Tensor operations for progress computation |

### 4.6 Training Commands

```bash
# Set environment variables
export DATASET_PATH=/vision/anishn/GTCC_Data_Processed_1fps
export OUTPUT_PATH=/vision/anishn/GTCC_CVPR2024/output
export JSON_DPATH=/vision/anishn/GTCC_CVPR2024/dset_jsons

# Train GTCC model
python multitask_train.py \
    --version 1 \
    --dataset egoprocel \
    --loss_type GTCC \
    --base_arch temporal_stacking \
    --lr 0.001 \
    --batch_size 4 \
    --epochs 50 \
    --output_dimensions 128 \
    --MCN

# Loss type options: GTCC, tcc, LAV, VAVA
# Base architecture options: temporal_stacking, resnet50, naive
```

### 4.7 Evaluation Metrics

Defined in `utils/evaluation.py`:

| Class | Metric | Description |
|-------|--------|-------------|
| `OnlineGeoProgressError` | OPE/OGPE | Online progress estimation error |
| `PhaseProgression` | PC | Phase classification accuracy |
| `EnclosedAreaError` | EAE | Area between predicted and true progress curves |
| `KendallsTau` | KT | Rank correlation of nearest neighbors |
| `WildKendallsTau` | KTW | Kendall's tau for in-the-wild videos |
| `PhaseClassification` | PHP | Phase classification with SVM |

### 4.8 Generating Aligned Features

```bash
# Generate GTCC-aligned features for ProTAS consumption
python generate_gtcc_aligned_features.py \
    --checkpoint /path/to/model/ckpt/epoch-50.pt \
    --output_dir /path/to/aligned_features/
```

---

## 5. progress_visualization Codebase Documentation

**Location**: `/vision/anishn/progress_visualization/`

### 5.1 Purpose

Creates **comparative visualization videos** showing progress estimates from 11 different models side-by-side with the original video.

### 5.2 Pipeline Overview

```
┌───────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  Raw Video    │────►│ Feature Extract  │────►│  GTCC Inference   │
│  (MP4)        │     │ (24fps or 1fps)  │     │  (8 models)       │
└───────────────┘     └──────────────────┘     └─────────┬─────────┘
                                                         │
                      ┌──────────────────┐               │
                      │ ProTAS Inference │◄──────────────┘
                      │  (3 variants)    │
                      └─────────┬────────┘
                                │
                      ┌─────────▼────────┐
                      │  Visualization   │
                      │  (MP4 output)    │
                      └──────────────────┘
```

### 5.3 Models Visualized

**GTCC Family (8 models)**:
| Model | Description |
|-------|-------------|
| `gtcc_full_video` | GTCC trained on full videos (progress 0→1 across video) |
| `gtcc_act_level` | GTCC trained at action level (progress resets per action) |
| `tcc_full_video` | TCC baseline, full video |
| `tcc_act_level` | TCC baseline, action level |
| `vava_full_video` | VAVA baseline, full video |
| `vava_act_level` | VAVA baseline, action level |
| `lav_full_video` | LAV baseline, full video |
| `lav_act_level` | LAV baseline, action level |

**ProTAS Family (3 variants)**:
| Model | Description |
|-------|-------------|
| `protas_pred` | Progress for model's predicted action |
| `protas_gt` | Progress for ground truth action (model output) |
| `protas_true` | True ground truth progress (pre-computed) |

### 5.4 Key Files

| File | Purpose |
|------|---------|
| `scripts/extract_features_24fps.py` | Extract ResNet50 features at 24fps |
| `scripts/run_gtcc_inference.py` | Run all 8 GTCC models |
| `scripts/run_protas_inference.py` | Run ProTAS with 3 output variants |
| `scripts/visualize_progress.py` | Create MP4 with progress bars |

### 5.5 Usage

```bash
# Step 1: Extract features (if needed)
python scripts/extract_features_24fps.py

# Step 2: Run GTCC family inference
python scripts/run_gtcc_inference.py --fps both

# Step 3: Run ProTAS inference
python scripts/run_protas_inference.py --fps both

# Step 4: Create visualization
python scripts/visualize_progress.py --fps both
```

### 5.6 Output Format

- **Location**: `/vision/anishn/progress_visualization/outputs/`
- **Structure**:
```
outputs/
├── features/
│   └── {video}_gtcc_24fps.npy    # 24fps features for GTCC
│   └── {video}_protas_24fps.npy  # 24fps features for ProTAS
├── progress/
│   ├── gtcc_full_video_1fps.npy
│   ├── gtcc_act_level_24fps.npy
│   ├── protas_pred_1fps.npy
│   └── ...
└── visualizations/
    └── {video}_1fps.mp4          # Final visualization
```

---

## 6. Data Flow & Relationships

### 6.1 Complete Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAW VIDEO DATA                              │
│  /vision/anishn/Egocentric/Datasets/EGTEA-Gaze+/                   │
│  /vision/anishn/Egocentric/Datasets/CMU_Kitchens/                  │
│  /vision/anishn/Egocentric/Datasets/EpicTent/                      │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FEATURE EXTRACTION                             │
│  ResNet50 → 2048-d features (or 1024x14x14 conv maps)              │
│  Output: /vision/anishn/GTCC_Data_Processed_1fps/egoprocel/frames/ │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  GTCC Training  │    │  TCC Training   │    │  LAV/VAVA       │
│  (GTCC_CVPR24)  │    │  (GTCC_CVPR24)  │    │  Training       │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ALIGNED FEATURES                               │
│  /vision/anishn/GTCC_Data_Processed_1fps/egoprocel/                │
│    ├── aligned_features/      (GTCC)                               │
│    ├── tcc_aligned_features/  (TCC)                                │
│    ├── vava_aligned_features/ (VAVA)                               │
│    └── lav_aligned_features/  (LAV)                                │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ProTAS TRAINING                                │
│  Uses raw features (2048-d) + aligned features for comparison      │
│  Checkpoints: /u/anishn/models/{exp_id}/{dataset}/                 │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      VISUALIZATION                                  │
│  /vision/anishn/progress_visualization/outputs/                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Shared Data Paths

| Data Type | Path |
|-----------|------|
| Raw Videos | `/vision/anishn/Egocentric/Datasets/` |
| GTCC Features (1fps) | `/vision/anishn/GTCC_Data_Processed_1fps/` |
| GTCC JSON Metadata | `/vision/anishn/GTCC_CVPR2024/dset_jsons/` |
| ProTAS Data (1fps) | `/vision/anishn/ProTAS/data_1fps/` |
| ProTAS Models | `/u/anishn/models/` |
| GTCC Models | `/vision/anishn/GTCC_CVPR2024/output/` |

### 6.3 Model Dependencies

```
GTCC/TCC/LAV/VAVA Models (trained first)
         │
         ├──► Generate aligned features
         │         │
         │         ▼
         └──► ProTAS (can use raw OR aligned features)
                   │
                   ▼
              Visualization Pipeline
```

---

## 7. Input/Output Specifications

### 7.1 Input Formats

| Input Type | Format | Shape | Example |
|------------|--------|-------|---------|
| Raw video | MP4/AVI | HxWx3 | EGTEA videos at 24fps |
| ResNet features | .npy | (T, 2048) | Extracted at 1fps |
| Conv features | .npy | (T, 1024, 14, 14) | For MCN models |
| Ground truth | .txt | T lines | One action per line |
| Progress GT | .npy | (C, T) | Per-class progress |
| Task graph | .pkl | dict | predecessor/successor matrices |

### 7.2 Intermediate Formats

| Data | Producer | Consumer | Shape |
|------|----------|----------|-------|
| Aligned features | GTCC/TCC/etc | ProTAS, Viz | (T, 128) |
| Action predictions | ProTAS | Viz | (T,) int |
| Progress predictions | ProTAS | Viz | (C, T) float |
| Cumulative distance | GTCC eval | Progress | (T,) float |

### 7.3 Output Formats

| Output | Format | Description |
|--------|--------|-------------|
| Action segmentation | .txt | Frame-level action predictions |
| Progress estimates | .npy | (T,) or (C, T) progress values |
| Evaluation metrics | .json | Acc, Edit, F1, OPE, KT, etc. |
| Visualization | .mp4 | Video with progress bars |

---

## 8. Environment & Dependencies

### 8.1 Conda Environment

```bash
conda activate /vision/anishn/envs/vidal
```

### 8.2 Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 1.x+ | Deep learning framework |
| torchvision | 0.x+ | Vision models (ResNet) |
| numpy | 1.x+ | Array operations |
| scipy | 1.x+ | Scientific computing |
| opencv-python | 4.x+ | Video I/O |
| numba | 0.x+ | JIT compilation for GMM |
| imageio | 2.x+ | Image/video reading |
| tqdm | 4.x+ | Progress bars |
| pandas | 1.x+ | Data manipulation |
| scikit-learn | 1.x+ | SVM for evaluation |
| matplotlib | 3.x+ | Plotting |

### 8.3 Environment Variables (GTCC)

```bash
export DATASET_PATH=/vision/anishn/GTCC_Data_Processed_1fps
export OUTPUT_PATH=/vision/anishn/GTCC_CVPR2024/output
export JSON_DPATH=/vision/anishn/GTCC_CVPR2024/dset_jsons
```

---

## 9. Quick Reference

### 9.1 Key File Paths

```bash
# ProTAS
/vision/anishn/ProTAS/main.py                    # Training entry
/vision/anishn/ProTAS/model.py                   # Model architecture
/vision/anishn/ProTAS/data_1fps/                 # Data directory
/u/anishn/models/                                # Saved models

# GTCC
/vision/anishn/GTCC_CVPR2024/multitask_train.py  # Training entry
/vision/anishn/GTCC_CVPR2024/utils/loss_functions.py  # Loss functions
/vision/anishn/GTCC_CVPR2024/utils/evaluation.py # Metrics
/vision/anishn/GTCC_CVPR2024/output/             # Experiments

# Visualization
/vision/anishn/progress_visualization/scripts/   # Pipeline scripts
/vision/anishn/progress_visualization/outputs/   # Generated outputs
```

### 9.2 Common Commands

```bash
# Activate environment
conda activate /vision/anishn/envs/vidal

# Train ProTAS
cd /vision/anishn/ProTAS
python main.py --dataset egoprocel_subset2_OP_P --graph --causal

# Train GTCC
cd /vision/anishn/GTCC_CVPR2024
python multitask_train.py --version 1 --dataset egoprocel --loss_type GTCC --MCN

# Run visualization pipeline
cd /vision/anishn/progress_visualization
python scripts/run_gtcc_inference.py
python scripts/run_protas_inference.py
python scripts/visualize_progress.py
```

### 9.3 Dataset Quick Reference

| Subset | Videos | Actions | Source |
|--------|--------|---------|--------|
| egoprocel_subset1_S | CMU Kitchens | 50 | S01-S07 |
| egoprocel_subset2_OP_P | EGTEA-Gaze+ | 50 | OP01-OP06, P02-P24 |
| egoprocel_subset3_tent | EpicTent | ~40 | Tent assembly |
| egoprocel_subset4_numbers | MECCANO | ~30 | Numbered videos |
| egoprocel_subset5_head | PC Assembly | ~30 | Head_* videos |

### 9.4 Model Checkpoint Paths

```bash
# GTCC models
/vision/anishn/GTCC_CVPR2024/output/multi-task-setting/
├── V1___GTCC_egoprocel_full_video/ckpt/epoch-50.pt
├── V1___GTCC_egoprocel_act_level/ckpt/epoch-50.pt
├── V2___tcc_egoprocel_full_video/ckpt/epoch-50.pt
├── V2___VAVA_egoprocel_full_video/ckpt/epoch-50.pt
└── V6___LAV_egoprocel_full_video/ckpt/epoch-50.pt

# ProTAS models
/u/anishn/models/
├── egoprocel_subset2_OP_P_1fps/egoprocel_subset2_OP_P/split_1/epoch-50.model
└── ...
```

---

## 10. Online Progress Estimation (Detailed)

This section provides an in-depth look at how progress estimation works across all three codebases, covering the mathematical foundations, implementation details, and the two paradigms (video-level vs action-level).

### 10.1 Progress Estimation Paradigms

There are **two fundamentally different approaches** to progress estimation:

| Paradigm | Description | Progress Range | Use Case |
|----------|-------------|----------------|----------|
| **Video-Level** | Progress goes 0→1 across the entire video | 0% at start, 100% at end | Overall task completion |
| **Action-Level** | Progress resets to 0 at each action boundary | 0→1 within each action | Per-action completion |

```
VIDEO-LEVEL PROGRESS:
Frame:    [0]----[100]----[200]----[300]----[400]----[500]
Progress:  0.0    0.2      0.4      0.6      0.8      1.0
           |_______________________________________________|
                    Monotonically increasing

ACTION-LEVEL PROGRESS:
Frame:    [0]----[100]----[200]----[300]----[400]----[500]
Action:   |--Action A---|--Action B---|---Action C----|
Progress:  0.0 → 1.0     0.0 → 1.0     0.0 → 1.0
           |___________|  |___________|  |____________|
              Resets at each action boundary
```

### 10.2 ProTAS Progress Estimation

ProTAS performs **explicit progress prediction** as an auxiliary task alongside action segmentation.

#### 10.2.1 Architecture for Progress Prediction

```
                    ┌─────────────────────────────────────┐
                    │       SingleStageModel              │
                    ├─────────────────────────────────────┤
Input Features ────►│  conv_1x1 → Dilated Conv Layers    │
(2048-d)            │             │                       │
                    │             ▼                       │
                    │    ┌────────┴────────┐              │
                    │    │                 │              │
                    │    ▼                 ▼              │
                    │ conv_out          GRU (APP)         │
                    │    │                 │              │
                    │    ▼                 ▼              │
                    │ Action Logits   conv_app            │
                    │ (C classes)        │               │
                    │    │                ▼              │
                    │    │         Progress Output       │
                    │    │         (C classes × T)       │
                    │    │                │              │
                    │    └───────┬────────┘              │
                    │            ▼                       │
                    │  ProbabilityProgressFusionModel    │
                    │            │                       │
                    │            ▼                       │
                    │     Fused Output                   │
                    └─────────────────────────────────────┘
```

**Key Code** (`/vision/anishn/ProTAS/model.py:83-104`):
```python
class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, causal=False):
        # ... conv layers ...

        # Action Progress Prediction (APP) module
        self.gru_app = nn.GRU(num_f_maps, num_f_maps, num_layers=1,
                              batch_first=True, bidirectional=not causal)
        self.conv_app = nn.Conv1d(num_f_maps, num_classes, 1)
        self.prob_fusion = ProbabilityProgressFusionModel(num_classes)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)

        prob_out = self.conv_out(out) * mask[:, 0:1, :]

        # Progress prediction via GRU
        progress_out, _ = self.gru_app(out.permute(0, 2, 1))
        progress_out = progress_out.permute(0, 2, 1)
        progress_out = self.conv_app(progress_out) * mask[:, 0:1, :]

        # Fuse action probabilities with progress
        out = self.prob_fusion(prob_out, progress_out)
        return out, progress_out  # Returns both action and progress
```

#### 10.2.2 Ground Truth Progress Generation

Ground truth progress is pre-computed per action class using `write_progress_values.py`:

```python
def write_progress_values(dataset, bg_class=[0], map_delimiter=' '):
    """
    For each video:
    - Parse ground truth action labels
    - For each action segment, compute linear progress 0→1
    - Store as (num_classes, T) array
    """
    progress_values = np.zeros([len(actions_dict), len(content)])

    for k, v in groupby(classes):  # Group consecutive frames by action
        segment_length = len(list(v))
        if k not in bg_class:
            # Linear progress from 0 to 1 within segment
            cur_progress = (np.arange(segment_length) + 1) / segment_length
            progress_values[k, cur_frame:cur_frame+segment_length] = cur_progress
        cur_frame += segment_length
```

**Progress Format**: `(num_classes, T)` where:
- Each row is an action class
- Values are 0.0 outside that action's segments
- Values go 0→1 linearly within each segment of that action

#### 10.2.3 Progress Loss Function

ProTAS uses **MSE loss** between predicted and ground truth progress:

```python
# In Trainer.train() - model.py:159-166
for p, progress_p in zip(predictions, progress_predictions):
    # Action classification loss
    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes),
                    batch_target.view(-1))
    # Smoothness loss
    loss += 0.15 * torch.mean(torch.clamp(
        self.mse(F.log_softmax(p[:, :, 1:], dim=1),
                 F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16))
    # Progress loss (MSE)
    progress_loss += self.mse(progress_p, batch_progress_target).mean()

loss += self.progress_lw * progress_loss  # Default weight: 1.0
```

#### 10.2.4 Extracting Progress at Inference

At inference, progress is extracted for the **predicted** or **ground truth** action:

```python
# Run model
predictions, progress_predictions = model(input_x, mask)

# progress_predictions[-1] shape: (1, num_classes, T)
progress_all = progress_predictions[-1].squeeze(0).cpu().numpy()  # (num_classes, T)

# Option 1: Progress for PREDICTED action at each frame
predicted_actions = predictions[-1].argmax(dim=1).squeeze().numpy()  # (T,)
progress_pred = np.array([progress_all[predicted_actions[t], t] for t in range(T)])

# Option 2: Progress for GROUND TRUTH action at each frame
progress_gt = np.array([progress_all[gt_actions[t], t] for t in range(T)])
```

### 10.3 GTCC Progress Estimation

GTCC computes progress **implicitly** from learned video embeddings using cumulative geodesic distance.

#### 10.3.1 Core Principle: Geodesic Distance

The key insight is that in a well-aligned embedding space, progress can be measured by how far along the "trajectory" a frame is:

```
Embedding Space:

    ●────●────●────●────●────●
   t=0  t=1  t=2  t=3  t=4  t=5

Progress = Cumulative L2 distance from start
         = ||e₁-e₀|| + ||e₂-e₁|| + ... + ||eₜ-eₜ₋₁||
```

#### 10.3.2 Cumulative Distance Function

**Key Code** (`/vision/anishn/GTCC_CVPR2024/utils/tensorops.py:327-331`):
```python
def get_cum_matrix(video):
    """
    Compute cumulative L2 distance along the embedding trajectory.

    Args:
        video: Tensor of shape (T, embedding_dim)

    Returns:
        P: Tensor of shape (T,) with cumulative distances
    """
    P = torch.zeros(video.shape[0])
    for t in range(1, video.shape[0]):
        P[t] = P[t-1] + torch.linalg.norm(video[t] - video[t-1])
    return P
```

#### 10.3.3 Video-Level Progress Estimation

For video-level progress (0→1 across entire video):

```python
def compute_progress_from_aligned_features(aligned_features):
    """
    Compute progress from aligned features using cumulative L2 distance.
    Progress goes 0→1 across the entire video.
    """
    cum_dist = get_cum_matrix(aligned_features)  # Shape: (T,)
    progress = cum_dist / cum_dist.max()  # Normalize to 0-1
    return progress.numpy()
```

#### 10.3.4 Action-Level Progress Estimation

For action-level progress (0→1 within each action, resets at boundaries):

```python
def compute_progress_action_level(aligned_features, segments, action_means):
    """
    Compute progress with per-action reset.
    Each action independently goes 0→1, background stays at 0.
    """
    T = aligned_features.shape[0]
    progress = torch.zeros(T)

    for seg in segments:
        action_name = seg['name']
        start, end = seg['start'], seg['end']

        # Skip background/SIL
        if action_name.lower() in ['sil', 'background']:
            continue

        # Extract ONLY this segment's embeddings
        segment_outputs = aligned_features[start:end + 1]

        # Compute cumulative distance for THIS SEGMENT ONLY
        segment_cum = get_cum_matrix(segment_outputs)

        # Normalize by action-specific mean (from training data)
        if action_name in action_means:
            action_mean = action_means[action_name]['mean']
            progress[start:end + 1] = segment_cum / action_mean
        else:
            # Fallback: normalize by segment's own max
            progress[start:end + 1] = segment_cum / segment_cum.max()

    return torch.clamp(progress, 0, 1).numpy()
```

#### 10.3.5 Action-Specific Normalization

To handle varying action lengths and complexities, GTCC computes **per-action geodesic means** from training data:

**Calculation** (`/vision/anishn/GTCC_CVPR2024/calculate_action_geodesic_means.py`):
```python
def calculate_geodesic_distance(features):
    """
    Geodesic = sum of L2 distances between consecutive frames.
    """
    if features.shape[0] < 2:
        return 0.0
    diffs = torch.norm(features[1:] - features[:-1], p=2, dim=1)
    return torch.sum(diffs).item()

# For each action in training set:
for action_name, distances in action_distances.items():
    final_stats[action_name] = {
        "mean": float(np.mean(distances)),
        "std": float(np.std(distances)),
        "count": len(distances)
    }
```

**Stored at**: `/vision/anishn/GTCC_CVPR2024/{method}_action_means.json`
```json
{
    "add_oil": {"mean": 2.3456, "std": 0.5678, "count": 45},
    "chop_vegetables": {"mean": 5.1234, "std": 1.2345, "count": 32},
    ...
}
```

### 10.4 Ground Truth Progress Functions

The codebase provides two ground truth computation functions:

#### 10.4.1 Video-Level Ground Truth

```python
def get_trueprogress(time_dict):
    """
    Compute video-level ground truth progress (0→1 across entire video).
    SIL/background frames maintain the previous progress value.
    """
    N = time_dict['end_frame'][-1] + 1
    progress = torch.zeros(N)
    prev_prg = 0

    # Each non-SIL action contributes equally to total progress
    prg = 1 / sum([1 if step != 'SIL' else 0 for step in time_dict['step']])

    for step, start, end in zip(time_dict['step'],
                                 time_dict['start_frame'],
                                 time_dict['end_frame']):
        if step != 'SIL':
            # Linear interpolation from prev_prg to prev_prg + prg
            progress[start:end+1] = torch.linspace(prev_prg, prev_prg + prg,
                                                    round(end - start + 1))
            prev_prg = prev_prg + prg
        else:
            # SIL frames maintain previous progress
            progress[start:end+1] = progress[start-1]
    return progress
```

#### 10.4.2 Action-Level Ground Truth

```python
def get_trueprogress_per_action(time_dict):
    """
    Compute action-level ground truth progress (0→1 per action).
    Each action independently goes 0→1.
    """
    N = time_dict['end_frame'][-1] + 1
    progress = torch.zeros(N)

    for step, start, end in zip(time_dict['step'],
                                 time_dict['start_frame'],
                                 time_dict['end_frame']):
        if step not in ['SIL', 'background']:
            segment_length = end - start + 1
            progress[start:end+1] = torch.linspace(0, 1, segment_length)
        else:
            progress[start:end+1] = 0  # Background has no progress

    return progress
```

### 10.5 Online Progress Error (OPE/OGPE) Metric

The primary evaluation metric for progress estimation:

```python
class OnlineGeoProgressError:
    """
    OGPE = mean(|true_progress - pred_progress|)

    Lower is better. Perfect alignment = 0.0
    """
    def __call__(self, model, config_obj, epoch, test_dataloaders, ...):
        # 1. Get normalization factor from training data
        train_cum_means, train_cum_vars = get_average_train_cum_distance(
            model, testfolder, data_structure, targ_task=task
        )

        gpe_list = []
        for outputs_dict, tdict in embedded_dl:
            outputs = outputs_dict['outputs']

            # 2. Compute ground truth progress
            true_progress = get_trueprogress(tdict)

            # 3. Compute predicted progress via cumulative distance
            pred_progress = get_cum_matrix(outputs)

            # 4. Normalize by training mean
            pred_progress = pred_progress / train_cum_means[task]

            # 5. Compute mean absolute error
            gpe = torch.mean(torch.abs(true_progress - pred_progress))
            gpe_list.append(gpe.item())

        return {'ogpe': np.mean(gpe_list)}
```

### 10.6 Visualization Pipeline Progress Computation

The visualization pipeline computes progress for 11 models:

```python
# GTCC Family (8 models) - uses cumulative distance
MODELS = {
    'gtcc_full_video': ...,   # Video-level: cum_dist / cum_dist.max()
    'gtcc_act_level': ...,    # Action-level: cum_dist / action_mean, per segment
    'tcc_full_video': ...,
    'tcc_act_level': ...,
    # ... etc
}

# ProTAS Family (3 variants) - explicit predictions
PROTAS_VARIANTS = {
    'protas_pred':  # progress_all[predicted_action[t], t]
    'protas_gt':    # progress_all[gt_action[t], t]  (model output)
    'protas_true':  # Pre-computed ground truth (linear 0→1 per action)
}
```

### 10.7 Progress Estimation Comparison

| Aspect | ProTAS | GTCC/TCC/LAV/VAVA |
|--------|--------|-------------------|
| **Method** | Explicit prediction (learned) | Implicit (cumulative distance) |
| **Output** | Per-class progress (C × T) | Single trajectory (T,) |
| **Training** | Supervised (MSE loss) | Self-supervised (alignment loss) |
| **Action Info** | Requires action labels | Does not require action labels |
| **Normalization** | None (direct regression) | By training mean or max |
| **Backbone** | TCN + GRU | CNN encoder |

### 10.8 Key Files for Progress Estimation

| File | Location | Purpose |
|------|----------|---------|
| `utils/write_progress_values.py` | ProTAS | Generate ground truth progress |
| `model.py` (SingleStageModel) | ProTAS | Progress prediction head (GRU) |
| `utils/tensorops.py` | GTCC | `get_cum_matrix`, `get_trueprogress` |
| `utils/evaluation.py` | GTCC | `OnlineGeoProgressError` metric |
| `calculate_action_geodesic_means.py` | GTCC | Compute per-action normalization |
| `eval_protas_video_level.py` | GTCC | Evaluate ProTAS at video level |
| `eval_protas_action_level.py` | GTCC | Evaluate ProTAS at action level |
| `run_gtcc_inference.py` | Viz | Compute GTCC progress for visualization |
| `run_protas_inference.py` | Viz | Compute ProTAS progress for visualization |

### 10.9 Code Examples

#### Computing Progress from Aligned Features
```python
import torch
import numpy as np

# Load aligned features
aligned_features = torch.from_numpy(
    np.load('/vision/anishn/GTCC_Data_Processed_1fps/egoprocel/aligned_features/OP01-R07-Pizza.npy')
).float()

# Video-level progress
cum_dist = torch.zeros(aligned_features.shape[0])
for t in range(1, aligned_features.shape[0]):
    cum_dist[t] = cum_dist[t-1] + torch.linalg.norm(
        aligned_features[t] - aligned_features[t-1]
    )
video_progress = cum_dist / cum_dist.max()
print(f"Video progress: min={video_progress.min():.3f}, max={video_progress.max():.3f}")
```

#### Extracting ProTAS Progress
```python
import torch
import numpy as np
from models.protas_model import MultiStageModel

# Load model
model = MultiStageModel(num_stages=4, num_layers=10, num_f_maps=64,
                        dim=2048, num_classes=50, causal=True,
                        use_graph=True, init_graph_path='graph.pkl')
model.load_state_dict(torch.load('epoch-50.model'))
model.eval()

# Load features (T, 2048)
features = np.load('video_features.npy')
input_x = torch.tensor(features.T, dtype=torch.float).unsqueeze(0)  # (1, 2048, T)
mask = torch.ones(input_x.size())

# Run inference
with torch.no_grad():
    predictions, progress_predictions = model(input_x, mask)

# Extract progress for predicted action
progress_all = progress_predictions[-1].squeeze(0).numpy()  # (num_classes, T)
predicted_actions = predictions[-1].argmax(dim=1).squeeze().numpy()  # (T,)
progress = np.array([progress_all[predicted_actions[t], t] for t in range(len(predicted_actions))])
progress = np.clip(progress, 0, 1)
```

---

## 11. Troubleshooting

### 11.1 Common Issues

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce batch size, use gradient checkpointing |
| Feature dimension mismatch | Check if features need transpose (EgoProceL: yes) |
| Missing graph.pkl | Run `utils/write_graph.py` to generate |
| Action mapping errors | Check delimiter (`\|` vs ` `) |

### 11.2 Data Validation

```python
# Check feature shapes
import numpy as np
features = np.load('/vision/anishn/ProTAS/data_1fps/egoprocel_subset2_OP_P/features/OP01-R07-Pizza.npy')
print(f"Features shape: {features.shape}")  # Expected: (T, 2048)

# Check progress values
progress = np.load('/vision/anishn/ProTAS/data_1fps/egoprocel_subset2_OP_P/progress/OP01-R07-Pizza.npy')
print(f"Progress shape: {progress.shape}")  # Expected: (num_classes, T)
print(f"Progress range: [{progress.min()}, {progress.max()}]")  # Expected: [0, 1]
```

---

*Document generated for the Elhamifar Lab research ecosystem. Last updated: February 2025.*
