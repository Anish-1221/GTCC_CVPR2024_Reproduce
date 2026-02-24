# GTCC Progress Loss: Training Pipeline and Architecture Explanation

## Overview

This document explains how the GTCC (Gaussian Temporal Cycle-Consistency) framework implements progress loss training, curriculum learning, and joint training. Based on verified code from `/vision/anishn/GTCC_CVPR2024/`.

---

## Part 1: Architecture Diagrams

### Training Type 1: GTCC with Cumulative L2 Progress Loss

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE: cumulative_l2 METHOD                      │
└─────────────────────────────────────────────────────────────────────────────────┘

INPUT: Batch of Videos + Action Annotations (times_dict)
              │
              ▼
┌─────────────────────────────────────────┐
│         BASE MODEL (ResNet/Stacking)    │
│   Input: (T, 2048) features per video   │
│   Output: (T, 128) embeddings           │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│      MULTI-PRONG ATTENTION MODEL        │
│  ┌────────┐ ┌────────┐     ┌────────┐   │
│  │Prong 1 │ │Prong 2 │ ... │Prong N │   │
│  └───┬────┘ └───┬────┘     └───┬────┘   │
│      └──────────┼──────────────┘        │
│                 ▼                       │
│     ┌─────────────────────┐             │
│     │  Attention Layer    │             │
│     │  (Softmax weights)  │             │
│     └─────────┬───────────┘             │
│               ▼                         │
│     Weighted Combination → (T, 128)     │
└───────────────────┬─────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌───────────────────┐   ┌───────────────────────────────────────┐
│  ALIGNMENT LOSS   │   │        PROGRESS LOSS (cumulative_l2)  │
│                   │   │                                       │
│ GTCC/TCC/LAV/VAVA │   │  1. Pick random video from batch      │
│                   │   │  2. Pick random frame index           │
│ Compares pairs    │   │  3. For each action segment:          │
│ of video          │   │     - Compute cumulative L2 distance  │
│ embeddings        │   │     - Normalize by segment max        │
│                   │   │  4. GT = linear progress per action   │
│                   │   │  5. Loss = |predicted - GT|           │
└─────────┬─────────┘   └───────────────────┬───────────────────┘
          │                                 │
          └─────────────┬───────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              TOTAL LOSS COMPUTATION                             │
│                                                                 │
│   total_loss = alignment_loss + lambda_fixed * progress_loss    │
│                                                                 │
│   Default: lambda_fixed = 0.1                                   │
└─────────────────────────────────────────────────────────────────┘


CUMULATIVE L2 PROGRESS COMPUTATION (per action):
═══════════════════════════════════════════════

Action Segment: [frame 20 ─────────────────────────── frame 80]
Embeddings:      e[20]    e[21]    e[22]   ...    e[79]   e[80]

Cumulative Distance:
  P[20] = 0
  P[21] = ||e[21] - e[20]||
  P[22] = P[21] + ||e[22] - e[21]||
  ...
  P[80] = P[79] + ||e[80] - e[79]||

Normalized Predicted Progress:
  pred[t] = P[t] / P[80]    (values in [0, 1])

Ground Truth Progress:
  gt[t] = (t - 20 + 1) / 61  (linear from 0 to 1)

Background/SIL segments: progress = 0
```

---

### Training Type 2: GTCC with Learnable Progress Head

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE: learnable METHOD                          │
└─────────────────────────────────────────────────────────────────────────────────┘

INPUT: Batch of Videos + Action Annotations (times_dict)
              │
              ▼
┌─────────────────────────────────────────┐
│         BASE MODEL (ResNet/Stacking)    │
│   Input: (T, 2048) features per video   │
│   Output: (T, 128) embeddings           │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│      MULTI-PRONG ATTENTION MODEL        │
│  ┌────────┐ ┌────────┐     ┌────────┐   │
│  │Prong 1 │ │Prong 2 │ ... │Prong N │   │
│  └───┬────┘ └───┬────┘     └───┬────┘   │
│      └──────────┼──────────────┘        │
│                 ▼                       │
│     ┌─────────────────────┐             │
│     │  Attention Layer    │             │
│     └─────────┬───────────┘             │
│               ▼                         │
│     Combined Embedding → (T, 128)       │
└───────────────────┬─────────────────────┘
                    │
        ┌───────────┴───────────────────────────────────┐
        │                                               │
        ▼                                               ▼
┌───────────────────┐                   ┌───────────────────────────────────────┐
│  ALIGNMENT LOSS   │                   │     PROGRESS LOSS (learnable head)    │
│                   │                   │                                       │
│ GTCC/TCC/LAV/VAVA │                   │  1. Pick random video from batch      │
│                   │                   │  2. Pick random non-background action │
│                   │                   │  3. Sample random segment within it   │
│                   │                   │  4. Sample random target frame index  │
│                   │                   │  5. Extract embeddings up to target   │
│                   │                   │  6. Pass through ProgressHead →       │
└─────────┬─────────┘                   │                                       │
          │                             │  ┌─────────────────────────────────┐  │
          │                             │  │        PROGRESS HEAD            │  │
          │                             │  │  ┌───────────────────────────┐  │  │
          │                             │  │  │ GRU (input→hidden_dim)    │  │  │
          │                             │  │  └─────────────┬─────────────┘  │  │
          │                             │  │                ▼                │  │
          │                             │  │  ┌───────────────────────────┐  │  │
          │                             │  │  │ Linear(hidden→32) + ReLU  │  │  │
          │                             │  │  └─────────────┬─────────────┘  │  │
          │                             │  │                ▼                │  │
          │                             │  │  ┌───────────────────────────┐  │  │
          │                             │  │  │ Linear(32→1) + Sigmoid    │  │  │
          │                             │  │  └─────────────┬─────────────┘  │  │
          │                             │  │                ▼                │  │
          │                             │  │      pred_progress ∈ [0,1]      │  │
          │                             │  └─────────────────────────────────┘  │
          │                             │                                       │
          │                             │  7. GT = (target - action_start + 1)  │
          │                             │         / action_length               │
          │                             │  8. Loss = |pred - GT|                │
          │                             └───────────────────┬───────────────────┘
          │                                                 │
          └─────────────────────────┬───────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│              TOTAL LOSS COMPUTATION                             │
│                                                                 │
│   total_loss = alignment_loss + lambda_fixed * progress_loss    │
│                                                                 │
│   (ProgressHead is learnable - gradients flow through GRU)      │
└─────────────────────────────────────────────────────────────────┘


SEGMENT SAMPLING VISUALIZATION:
═══════════════════════════════

Video: [───SIL───][──────Action "cut"──────][───SIL───][─Action "mix"─]
Frames:  0......19  20..................99   100....119  120........180

Step 1: Select valid action (length >= 3, not background)
        → Choose "cut" (frames 20-99, length=80)

Step 2: Sample segment bounds within action
        → seg_start = 35, seg_end = 70

Step 3: Sample random target index within segment
        → target_idx = 55

Step 4: Extract embeddings[35:56] (21 frames)
        → Pass to ProgressHead

Step 5: Compute GT progress at target_idx
        → GT = (55 - 20 + 1) / 80 = 36/80 = 0.45

Step 6: ProgressHead predicts scalar ∈ [0, 1]
        → Loss = |pred - 0.45|
```

---

## Part 2: Curriculum Learning in GTCC

GTCC implements curriculum learning through the **dropout mechanism** in the alignment loss:

```
CURRICULUM LEARNING IN GTCC:
═══════════════════════════

Formula: drop_min = gamma^(epoch + 1)

Where gamma is typically < 1 (e.g., 0.7)

Epoch 0:  drop_min = 0.7^1  = 0.70
Epoch 1:  drop_min = 0.7^2  = 0.49
Epoch 2:  drop_min = 0.7^3  = 0.34
Epoch 5:  drop_min = 0.7^6  = 0.12
Epoch 10: drop_min = 0.7^11 = 0.02

INTERPRETATION:
───────────────
Early epochs (high drop_min):
  → Most frames treated as "easy" (BX close to 1)
  → Standard alignment loss applied

Late epochs (low drop_min):
  → Dropout network learns which frames are "hard"
  → Hard frames get inverted loss (1/align_loss)
  → Model learns to handle difficult alignments

LOSS FORMULA:
─────────────
For each frame:
  BX = dropout_network(embedding)  # normalized to [drop_min, 1]

  if gamma < 1:
      loss = BX * align_loss + (1-BX) * (1/align_loss)
  else:
      loss = align_loss  # no curriculum (gamma=1)

Effect:
  - BX ≈ 1 (easy frame): loss ≈ align_loss
  - BX ≈ drop_min (hard frame): loss ≈ (1/align_loss)
    → Encourages divergence for hard-to-align frames
```

---

## Part 3: Joint Training Pipeline

```
JOINT TRAINING: Alignment + Progress Loss
═════════════════════════════════════════

┌────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING LOOP                                    │
│                                                                            │
│  for epoch in range(num_epochs):                                           │
│      for batch in dataloader:                                              │
│          videos, times = batch                                             │
│                                                                            │
│          # Forward pass                                                    │
│          output_dict = model(videos)                                       │
│          # output_dict contains:                                           │
│          #   'outputs': List of (T, 128) embeddings                        │
│          #   'dropouts': Dropout scalars (if GTCC)                         │
│          #   'progress_head': ProgressHead module (if learnable)           │
│                                                                            │
│          # Loss computation                                                │
│          loss_dict = loss_fn(output_dict, epoch, times=times)              │
│                                                                            │
│          # loss_dict contains:                                             │
│          #   'alignment_loss': Sum of GTCC/TCC/LAV/VAVA losses             │
│          #   'progress_loss': Progress supervision loss                    │
│          #   'total_loss': alignment_loss + lambda * progress_loss         │
│                                                                            │
│          # Backward pass                                                   │
│          loss_dict['total_loss'].backward()                                │
│          optimizer.step()                                                  │
│                                                                            │
│      # Checkpoint saving (dual system)                                     │
│      if val_loss_combined < best_combined:                                 │
│          save('best_model_combined.pt')                                    │
│      if val_alignment_loss < best_alignment:                               │
│          save('best_model_alignment.pt')                                   │
└────────────────────────────────────────────────────────────────────────────┘


LOSS COMPOSITION:
─────────────────

┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│                    ┌─────────────────────┐                               │
│                    │  ALIGNMENT LOSS     │                               │
│                    │  ─────────────────  │                               │
│    ┌───────────┐   │  GTCC_loss() or     │                               │
│    │ Video     │   │  TCC_loss() or      │   ────┐                       │
│    │ Embeddings│──▶│  LAV_loss() or      │       │                       │
│    │ (pairs)   │   │  VAVA_loss()        │       │                       │
│    └───────────┘   └─────────────────────┘       │                       │
│                                                  │                       │
│                                                  ▼                       │
│                                          ┌─────────────┐                 │
│                                          │ total_loss  │                 │
│                                          │     =       │                 │
│                                          │ alignment   │                 │
│                    ┌─────────────────────┐│  + λ *     │                 │
│                    │  PROGRESS LOSS      ││ progress   │                 │
│    ┌───────────┐   │  ─────────────────  │└─────────────┘                 │
│    │ Video     │   │  cumulative_l2:     │       ▲                       │
│    │ Embeddings│──▶│    |cum_dist - gt|  │       │                       │
│    │ + times   │   │  or                 │  ────┘                        │
│    └───────────┘   │  learnable:         │                               │
│                    │    |head(seg) - gt| │                               │
│                    └─────────────────────┘                               │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘


WHAT EACH LOSS LEARNS:
──────────────────────

ALIGNMENT LOSS (GTCC):
  - Cross-video temporal correspondence
  - Frame-to-frame soft matching across videos
  - GMM-based attention for flexible alignment
  - With dropout: learns which frames are hard to align

PROGRESS LOSS:
  - Per-action temporal ordering
  - Embeddings should reflect "how far" into an action
  - cumulative_l2: Implicit - embeddings move in feature space
  - learnable: Explicit - ProgressHead learns to predict progress
```

---

## Part 4: Key Function Details

### Ground Truth Progress (`get_trueprogress_per_action`)
```
Location: utils/tensorops.py:312-326

Input: times_dict with action annotations
Output: (N,) tensor of progress values

For each action segment:
  - Non-background: progress[t] = (t - start + 1) / length
  - Background ('0', 'SIL', 'background'): progress[t] = 0

Example:
  Action at frames 20-79 (60 frames):
    progress[20] = 1/60 = 0.017
    progress[39] = 20/60 = 0.333
    progress[79] = 60/60 = 1.000
```

### Predicted Progress (`get_normalized_predicted_progress_action`)
```
Location: utils/tensorops.py:336-368

Input: features (T, D), times_dict
Output: (T,) tensor of predicted progress

For each action segment:
  1. Extract segment features
  2. Compute cumulative L2 distance: P[t] = Σ ||e[i] - e[i-1]||
  3. Normalize: pred[t] = P[t] / P[last]

Edge cases handled:
  - Segments < 2 frames: return 0.5
  - Zero movement (identical frames): return linear interpolation
  - Background segments: return 0
```

### Segment Sampling (`sample_action_segment_with_random_index`)
```
Location: utils/tensorops.py:371-408

Purpose: Sample training data for learnable progress head

Steps:
  1. Filter valid actions (non-background, length >= min_segment_len)
  2. Random select one action
  3. Random segment bounds within action
  4. Random target frame within segment
  5. Return embeddings[seg_start:target+1], gt_progress, action_name

Returns None if no valid actions found
```

---

## Part 5: Configuration and Usage

### Config Settings (configs/generic_config.py:67-76)
```python
CONFIG.PROGRESS_LOSS = {
    'enabled': False,               # Enable progress loss
    'method': 'cumulative_l2',      # 'cumulative_l2' or 'learnable'
    'lambda_fixed': 0.1,            # Weight coefficient
    'learnable': {
        'hidden_dim': 64,           # GRU hidden dimension
        'use_gru': True,            # Use GRU (True) or mean-pooling (False)
        'min_segment_len': 3,       # Minimum frames in sampled segment
    },
}
```

### Training Commands
```bash
# GTCC with cumulative L2 progress loss
python multitask_train.py 1 --gtcc --ego --resnet --mcn \
    --progress_loss cumulative_l2 --progress_lambda 0.1

# GTCC with learnable progress head
python multitask_train.py 1 --gtcc --ego --resnet --mcn \
    --progress_loss learnable --progress_lambda 0.1

# Standard GTCC (no progress loss)
python multitask_train.py 1 --gtcc --ego --resnet --mcn
```

---

## Part 6: Summary Comparison

| Aspect | cumulative_l2 | learnable |
|--------|---------------|-----------|
| **Learnable parameters** | None (uses embedding distances) | Yes (GRU + MLP) |
| **Progress prediction** | Normalized cumulative L2 distance | Neural network output |
| **Sampling** | Random frame from random video | Random segment within random action |
| **Ground truth** | Linear progress per action segment | Position-based progress in action |
| **Gradient flow** | Through encoder only | Through encoder + ProgressHead |
| **Complexity** | Simpler, no extra parameters | More complex, additional module |
| **Use case** | Implicit progress supervision | Explicit progress prediction |

---

## Part 7: Code File Locations

| File | Purpose |
|------|---------|
| `utils/loss_entry.py:1-129` | Loss function factory, combines alignment + progress |
| `utils/tensorops.py:298-408` | Progress computation helpers |
| `models/model_multiprong.py:144-255` | ProgressHead and MultiProngAttDropoutModel |
| `configs/generic_config.py:67-76` | Progress loss configuration |
| `utils/loss_functions.py:345-454` | GTCC loss with curriculum learning |

---

## Part 8: Inference Pipeline Diagrams

### Inference Type 1: cumulative_l2 Method

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE: cumulative_l2 METHOD                     │
└─────────────────────────────────────────────────────────────────────────────────┘

PREREQUISITES:
  1. Trained model checkpoint (best_model_combined.pt or best_model_alignment.pt)
  2. action_means.json (pre-computed from training set using calculate_action_means.py)

┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│  INPUT: Test Video Features (T, 2048) + Ground Truth Action Segments            │
│                      │                                                          │
│                      ▼                                                          │
│  ┌─────────────────────────────────────────┐                                    │
│  │      TRAINED MODEL (no grad)            │                                    │
│  │  Base Model → Multi-Prong → Attention   │                                    │
│  └───────────────────┬─────────────────────┘                                    │
│                      │                                                          │
│                      ▼                                                          │
│            Video Embeddings (T, 128)                                            │
│                      │                                                          │
│                      ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  FOR EACH ACTION SEGMENT (from ground truth):                           │    │
│  │                                                                         │    │
│  │    Action: "cut" (frames 20-99)                                         │    │
│  │                      │                                                  │    │
│  │                      ▼                                                  │    │
│  │    ┌─────────────────────────────────────────┐                          │    │
│  │    │  1. Extract segment embeddings[20:100]  │                          │    │
│  │    └─────────────────────┬───────────────────┘                          │    │
│  │                          ▼                                              │    │
│  │    ┌─────────────────────────────────────────┐                          │    │
│  │    │  2. Compute cumulative L2 distance      │                          │    │
│  │    │     P[t] = Σ ||e[i] - e[i-1]||          │                          │    │
│  │    └─────────────────────┬───────────────────┘                          │    │
│  │                          ▼                                              │    │
│  │    ┌─────────────────────────────────────────┐                          │    │
│  │    │  3. Normalize by ACTION MEAN            │                          │    │
│  │    │     (from action_means.json)            │                          │    │
│  │    │                                         │                          │    │
│  │    │     pred[t] = P[t] / action_means["cut"]│                          │    │
│  │    └─────────────────────┬───────────────────┘                          │    │
│  │                          ▼                                              │    │
│  │              Predicted Progress (0 → ~1.0)                              │    │
│  │                                                                         │    │
│  │    Background/SIL segments: pred[t] = 0                                 │    │
│  │                                                                         │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                      │                                                          │
│                      ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  COMPUTE EVALUATION METRICS                                             │    │
│  │                                                                         │    │
│  │  Ground Truth: gt[t] = (t - action_start + 1) / action_length           │    │
│  │                (linear 0→1 per action, 0 for background)                │    │
│  │                                                                         │    │
│  │  OGPE = mean(|pred[t] - gt[t]|) over all action frames                  │    │
│  │         (background frames excluded from metric)                        │    │
│  │                                                                         │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                      │                                                          │
│                      ▼                                                          │
│                  OUTPUT: OGPE score (lower is better)                           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘


WHY ACTION MEANS ARE NEEDED:
════════════════════════════

During TRAINING:
  - Progress normalized by segment's own max: pred[t] = P[t] / P[last]
  - Always produces values in [0, 1]

During INFERENCE:
  - We don't have the "expected" final distance
  - Need to normalize by typical distance for this action type
  - action_means.json stores: {"cut": {"mean": 45.2, "std": 8.1}, ...}
  - Computed from training set embeddings

Generation commands:
  python generate_aligned_features.py --exp_folder <checkpoint_folder>
  python calculate_action_means.py --exp_folder <checkpoint_folder>
```

---

### Inference Type 2: learnable Method (ProgressHead) - Online Per-Frame Prediction

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE: learnable METHOD                         │
│                    (Online Per-Frame Prediction)                                │
└─────────────────────────────────────────────────────────────────────────────────┘

PREREQUISITES:
  1. Trained model checkpoint with ProgressHead weights
  2. NO action_means.json needed (ProgressHead predicts directly)

┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│  INPUT: Test Video Features (T, 2048) + Ground Truth Action Segments            │
│                      │                                                          │
│                      ▼                                                          │
│  ┌─────────────────────────────────────────┐                                    │
│  │      TRAINED MODEL (no grad)            │                                    │
│  │  Base Model → Multi-Prong → Attention   │                                    │
│  │            + ProgressHead               │                                    │
│  └───────────────────┬─────────────────────┘                                    │
│                      │                                                          │
│                      ▼                                                          │
│            Video Embeddings (T, 128)                                            │
│                      │                                                          │
│                      ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  FOR EACH ACTION SEGMENT (from ground truth):                           │    │
│  │                                                                         │    │
│  │    Action: "cut" (frames 20-99, length=80)                              │    │
│  │                      │                                                  │    │
│  │                      ▼                                                  │    │
│  │    ┌─────────────────────────────────────────────────────────────────┐  │    │
│  │    │  ONLINE PER-FRAME PREDICTION (similar to cumulative_l2)         │  │    │
│  │    │                                                                 │  │    │
│  │    │  FOR t = 20, 21, 22, ..., 99:                                   │  │    │
│  │    │                                                                 │  │    │
│  │    │    1. Extract partial segment: embeddings[20:t+1]               │  │    │
│  │    │       (only frames from action start UP TO current frame)       │  │    │
│  │    │                                                                 │  │    │
│  │    │    2. Pass through ProgressHead:                                │  │    │
│  │    │       ┌───────────────────────────────┐                         │  │    │
│  │    │       │ GRU(partial segment)          │                         │  │    │
│  │    │       │         ↓                     │                         │  │    │
│  │    │       │ Linear(64→32) + ReLU          │                         │  │    │
│  │    │       │         ↓                     │                         │  │    │
│  │    │       │ Linear(32→1) + Sigmoid        │                         │  │    │
│  │    │       │         ↓                     │                         │  │    │
│  │    │       │ pred_progress[t] ∈ [0, 1]     │                         │  │    │
│  │    │       └───────────────────────────────┘                         │  │    │
│  │    │                                                                 │  │    │
│  │    │  Example for action "cut" (frames 20-99):                       │  │    │
│  │    │    t=20: ProgressHead(emb[20:21])     → pred[20] = 0.02         │  │    │
│  │    │    t=21: ProgressHead(emb[20:22])     → pred[21] = 0.04         │  │    │
│  │    │    t=30: ProgressHead(emb[20:31])     → pred[30] = 0.15         │  │    │
│  │    │    t=50: ProgressHead(emb[20:51])     → pred[50] = 0.42         │  │    │
│  │    │    t=99: ProgressHead(emb[20:100])    → pred[99] = 0.98         │  │    │
│  │    │                                                                 │  │    │
│  │    │  Each frame gets its OWN predicted progress value!              │  │    │
│  │    │  (Just like cumulative_l2 computes cumulative distance          │  │    │
│  │    │   at each frame)                                                │  │    │
│  │    │                                                                 │  │    │
│  │    └─────────────────────────────────────────────────────────────────┘  │    │
│  │                                                                         │    │
│  │    Background/SIL segments: pred[t] = 0 (skipped)                       │    │
│  │                                                                         │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                      │                                                          │
│                      ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │  COMPUTE EVALUATION METRICS                                             │    │
│  │                                                                         │    │
│  │  Ground Truth: gt[t] = (t - action_start + 1) / action_length           │    │
│  │                                                                         │    │
│  │  OGPE = mean(|pred[t] - gt[t]|) over all action frames                  │    │
│  │                                                                         │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                      │                                                          │
│                      ▼                                                          │
│                  OUTPUT: OGPE score (lower is better)                           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘


COMPARISON: cumulative_l2 vs learnable (both do per-frame prediction)
═════════════════════════════════════════════════════════════════════

cumulative_l2 at inference:
  - For each frame t: pred[t] = cumulative_L2_distance[0:t] / action_mean
  - Needs pre-computed action_means.json
  - May exceed 1.0 if video has more movement than average

learnable at inference:
  - For each frame t: pred[t] = ProgressHead(embeddings[start:t+1])
  - No external files needed (ProgressHead predicts directly)
  - Always outputs values in [0, 1] due to Sigmoid
  - TRUE online prediction at each timestep
```

---

### Evaluation Flow Summary

```
EVALUATION PIPELINE (eval.py):
══════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│  1. LOAD MODEL                                                                  │
│     ├─ Load checkpoint (best_model_combined.pt)                                 │
│     ├─ Detect model type: use_progress_head = True/False                        │
│     └─ Set model to eval() mode                                                 │
│                                                                                 │
│  2. LOAD TEST DATA                                                              │
│     ├─ Load from data_splits.json (fixed train/val/test splits)                 │
│     └─ Create test dataloader                                                   │
│                                                                                 │
│  3. FOR EACH TEST VIDEO:                                                        │
│     │                                                                           │
│     ├─ Forward pass → embeddings (T, 128)                                       │
│     │                                                                           │
│     ├─ Load ground truth action segments from ProTAS groundTruth/*.txt          │
│     │                                                                           │
│     ├─ Compute predicted progress:                                              │
│     │   ├─ [cumulative_l2] Use action_means.json normalization                  │
│     │   └─ [learnable] Use ProgressHead + linear interpolation                  │
│     │                                                                           │
│     ├─ Compute ground truth progress:                                           │
│     │   └─ get_trueprogress_per_action(tdict)                                   │
│     │                                                                           │
│     └─ Compute error on action frames only (exclude background)                 │
│                                                                                 │
│  4. AGGREGATE RESULTS                                                           │
│     ├─ OGPE = mean of all per-video errors                                      │
│     └─ Save to EVAL_action_level/<checkpoint>/ogpe.csv                          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘


EVALUATION METRICS:
═══════════════════

┌────────────────────────────────────────────────────────────────────────────────┐
│                                                                                │
│  OGPE (Online Geo Progress Error):                                             │
│  ─────────────────────────────────                                             │
│                                                                                │
│    OGPE = (1/N_action) * Σ |pred_progress[t] - gt_progress[t]|                 │
│                                                                                │
│    Where N_action = number of frames in non-background actions                 │
│                                                                                │
│    Range: [0, 1+] (can exceed 1 if predictions overshoot)                      │
│    Perfect: 0.0                                                                │
│    Random: ~0.33                                                               │
│                                                                                │
│  ──────────────────────────────────────────────────────────────────────────    │
│                                                                                │
│  Visualization (plotting=True):                                                │
│                                                                                │
│    1.0 ┤                    ╭──── GT (green)                                   │
│        │                 ╭──╯                                                  │
│        │              ╭──╯                                                     │
│        │           ╭──╯  ╭──── Pred (blue)                                     │
│    0.5 ┤        ╭──╯  ╭──╯                                                     │
│        │     ╭──╯  ╭──╯                                                        │
│        │  ╭──╯  ╭──╯     ◄── Error area (red shading)                          │
│        │╭─╯  ╭──╯                                                              │
│    0.0 ┼─────────────────────────────────────────                              │
│         0        Frame        T                                                │
│                                                                                │
│    Saved to: <exp_folder>/plotting_progress/<task>/<video>.pdf                 │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

---

## Part 9: Code Fix for Online Per-Frame Prediction (Learnable Method)

The current evaluation code for learnable method uses interpolation instead of true per-frame prediction. Here's the fix:

### Current Code (evaluation_action_level.py:370-399) - WRONG:
```python
if use_learnable_progress:
    progress_head = outputs_dict.get('progress_head')

    for seg in segments:
        action_name = seg['name']
        start = max(0, min(seg['start'], current_len-1))
        end = max(0, min(seg['end'], current_len-1))

        if start >= current_len or action_name in ['SIL', 'background']:
            continue

        # PROBLEM: Takes FULL segment and interpolates
        segment_outputs = outputs[start:end+1]
        with torch.no_grad():
            segment_progress = progress_head(segment_outputs.to(device))

            # WRONG: Interpolates instead of per-frame prediction
            if segment_progress.dim() == 0 or segment_progress.numel() == 1:
                final_prog = segment_progress.item()
                seg_len = end - start + 1
                pred2_progress[start:end+1] = torch.linspace(0, final_prog, seg_len)
```

### Fixed Code - Online Per-Frame Prediction:
```python
if use_learnable_progress:
    # [LEARNABLE PROGRESS] Use ProgressHead for ONLINE per-frame prediction
    # Similar to how cumulative_l2 computes cumulative distance at each frame
    progress_head = outputs_dict.get('progress_head')
    if progress_head is None:
        print(f"[ERROR] Learnable model missing progress_head in output")
        continue

    for seg in segments:
        action_name = seg['name']
        start = max(0, min(seg['start'], current_len-1))
        end = max(0, min(seg['end'], current_len-1))

        if start >= current_len or action_name in ['SIL', 'background']:
            continue

        # ONLINE PER-FRAME PREDICTION:
        # For each frame t, predict progress using only frames [start:t+1]
        # This mirrors how cumulative_l2 computes cumulative distance at each frame
        with torch.no_grad():
            for t in range(start, end + 1):
                # Extract partial segment from action start up to current frame
                partial_segment = outputs[start:t+1].to(device)

                # Predict progress at frame t using only past frames (within this action)
                pred_progress_t = progress_head(partial_segment)
                pred2_progress[t] = pred_progress_t.item()
```

### Visual Comparison:

```
ACTION: frames 20-99 (80 frames total)

OLD (Interpolation):
────────────────────
  1. ProgressHead(embeddings[20:100]) → 0.95 (single scalar)
  2. pred[20:100] = linspace(0, 0.95, 80)

  Result: [0.00, 0.012, 0.024, ..., 0.95]  (linear, not learned!)

NEW (Online Per-Frame):
───────────────────────
  t=20: ProgressHead(emb[20:21])   → pred[20] = 0.03
  t=21: ProgressHead(emb[20:22])   → pred[21] = 0.05
  t=22: ProgressHead(emb[20:23])   → pred[22] = 0.07
  ...
  t=50: ProgressHead(emb[20:51])   → pred[50] = 0.41
  ...
  t=99: ProgressHead(emb[20:100])  → pred[99] = 0.97

  Result: Each frame has its OWN predicted value based on frames seen so far!

  This is analogous to cumulative_l2:
  t=20: cum_dist[20:21] / mean   → pred[20]
  t=21: cum_dist[20:22] / mean   → pred[21]
  ...
```

### Performance Note:

The new online prediction is slower because it calls ProgressHead once per frame instead of once per segment. For an action with 80 frames, this is 80x more forward passes through ProgressHead.

If performance is a concern, you could batch the predictions:
```python
# Batch version (more efficient)
with torch.no_grad():
    seg_len = end - start + 1
    # Pre-allocate predictions
    segment_preds = torch.zeros(seg_len)

    for idx, t in enumerate(range(start, end + 1)):
        partial_segment = outputs[start:t+1].to(device)
        segment_preds[idx] = progress_head(partial_segment).item()

    pred2_progress[start:end+1] = segment_preds
```

---

### Inference Commands

```bash
# Evaluate cumulative_l2 model (action-level metrics)
python eval.py -f /path/to/experiment_folder --level action

# Evaluate cumulative_l2 model (video-level metrics)
python eval.py -f /path/to/experiment_folder --level video

# Before running eval for cumulative_l2, generate action means:
python generate_aligned_features.py --exp_folder /path/to/experiment_folder
python calculate_action_means.py --exp_folder /path/to/experiment_folder

# For learnable models, no pre-processing needed - just run eval
python eval.py -f /path/to/learnable_experiment_folder --level action
```

---

## Verification

To test the implementation:

1. **Check loss computation**:
   - Run training and verify `progress_loss` appears in logs
   - Confirm `total_loss = alignment_loss + lambda * progress_loss`

2. **Check checkpoints**:
   - `best_model_combined.pt` (tracks total_loss)
   - `best_model_alignment.pt` (tracks alignment_loss only)

3. **Validate progress functions**:
   ```python
   from utils.tensorops import get_trueprogress_per_action
   times = {'step': ['0', '30', '0'], 'start_frame': [0, 10, 50], 'end_frame': [9, 49, 99]}
   gt = get_trueprogress_per_action(times)
   # Check: gt[0:10] = 0, gt[10:50] = linear 0→1, gt[50:100] = 0
   ```
