# Training Pipeline Documentation

This document explains the complete training pipeline for alignment models (LAV, VAVA, GTCC, TCC) with learnable progress prediction.

---

## Overall Training Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING LOOP                                       │
│                        (alignment_training_loop.py)                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌─────────────┐      ┌──────────────────────────────────────────────────┐     │
│   │   Batch     │      │           MultiProngAttDropoutModel              │     │
│   │  (videos,   │─────▶│                                                  │     │
│   │   times)    │      │  ┌─────────────┐   ┌─────────────┐              │     │
│   └─────────────┘      │  │ Base Model  │──▶│ Head Models │──▶ Attention │     │
│                        │  │ (ResNet50)  │   │ (16 heads)  │    Fusion    │     │
│                        │  └─────────────┘   └─────────────┘       │      │     │
│                        │                                          ▼      │     │
│                        │                              ┌───────────────┐  │     │
│                        │                              │ Aligned Emb.  │  │     │
│                        │                              │ (T × 128)     │  │     │
│                        │                              └───────────────┘  │     │
│                        │                                      │          │     │
│                        │                    ┌─────────────────┴──────┐   │     │
│                        │                    ▼                        ▼   │     │
│                        │            ┌─────────────┐         ┌──────────┐ │     │
│                        │            │ output_dict │         │ progress │ │     │
│                        │            │ ['outputs'] │         │  _head   │ │     │
│                        │            └─────────────┘         └──────────┘ │     │
│                        └──────────────────────────────────────────────────┘     │
│                                           │                       │             │
│                                           ▼                       ▼             │
│                        ┌──────────────────────────────────────────────────┐     │
│                        │              LOSS COMPUTATION                     │     │
│                        │                (loss_entry.py)                    │     │
│                        │                                                   │     │
│                        │   total_loss = alignment_loss + λ × progress_loss │     │
│                        │                                                   │     │
│                        │   ┌─────────────────┐    ┌──────────────────┐    │     │
│                        │   │ Alignment Loss  │    │  Progress Loss   │    │     │
│                        │   │ (LAV/VAVA/GTCC) │    │   (Learnable)    │    │     │
│                        │   └─────────────────┘    └──────────────────┘    │     │
│                        └──────────────────────────────────────────────────┘     │
│                                           │                                      │
│                                           ▼                                      │
│                        ┌──────────────────────────────────────────────────┐     │
│                        │              BACKWARD + OPTIMIZE                  │     │
│                        │                                                   │     │
│                        │  1. loss.backward()                               │     │
│                        │  2. Rescale progress_head gradients (÷ λ)         │     │
│                        │  3. Gradient clipping (max_norm=0.00001)          │     │
│                        │  4. optimizer.step()                              │     │
│                        └──────────────────────────────────────────────────┘     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Progress Head Training (Detailed)

### How Progress Head Training Works (Step-by-Step)

The progress head learns to predict "how far into an action" we are, outputting a value from 0 (action just started) to 1 (action complete). Here's exactly how it's trained:

#### Step 1: Segment Sampling
For each video in the batch, we sample multiple training examples. Each example consists of a **partial action segment** and its **ground truth progress value**.

- We randomly pick a non-background action from the video (e.g., "chop vegetables" spanning frames 50-100)
- We then pick a **target frame** within that action (e.g., frame 75)
- The **segment** is all frames from the action start up to the target: frames 50-75
- The **ground truth progress** is how far the target frame is into the action: (75-50+1)/(100-50+1) = 26/51 ≈ 0.51

We repeat this 10 times per video (samples_per_video=10), and for each sample we pick 5 different target frames (frames_per_segment=5). This gives us ~50 training examples per video.

#### Step 2: Stratified Sampling (Why It Matters)
Without stratified sampling, random target selection tends to pick frames from the middle of actions (progress ~0.3-0.7). The model rarely sees examples with progress near 0 or 1.

With stratified sampling enabled, we split possible target frames into three bins:
- **Early bin** (progress 0.0-0.33): Frames near the start of the action
- **Mid bin** (progress 0.33-0.67): Frames in the middle
- **Late bin** (progress 0.67-1.0): Frames near the end

We sample from each bin, ensuring the model learns what "just started" (progress ~0.1) and "almost done" (progress ~0.9) look like.

#### Step 3: Forward Pass Through Progress Head
For each (segment, gt_progress) pair:

1. **Input**: The segment embeddings, shape (T, 128) where T is the number of frames in the segment
2. **Position Encoding**: We concatenate each frame's normalized position to its embedding:
   - Frame 0 gets position 0/T = 0
   - Frame 1 gets position 1/T
   - Frame T-1 gets position (T-1)/T ≈ 1
   - This tells the GRU "you've seen 1 frame" vs "you've seen 50 frames"
3. **GRU Processing**: The GRU processes the sequence with a **learnable initial hidden state** (h0). This learned h0 helps the model output sensible values even for very short segments.
4. **Output**: The final hidden state goes through FC layers and Sigmoid to produce a single value in [0, 1]

#### Step 4: Weighted Loss Computation
We compute L1 loss between prediction and ground truth, but with a twist:

- **Early frames get higher weight**: If gt_progress=0.1, weight=10 (capped). If gt_progress=0.5, weight=2.
- **Why?** Early frame predictions are qualitatively more important. A user watching a progress bar wants to see it start near 0, not jump to 0.6 immediately.
- **Formula**: `loss = weight × |pred_progress - gt_progress|`

#### Step 5: Aggregation
All weighted losses across all samples are summed and divided by total weight (weighted average). This becomes `progress_loss`.

#### Step 6: Combined with Alignment Loss
The final loss is:
```
total_loss = alignment_loss + λ × progress_loss
```

Where λ (lambda) is typically 1,000,000 to make progress_loss contribute meaningfully (since alignment_loss for LAV can be 10M-100M while progress_loss is ~0.2).

#### Step 7: Gradient Rescaling (Critical!)
After `loss.backward()`, the progress_head has gradients scaled by λ (way too large). Before `optimizer.step()`, we divide progress_head gradients by λ:

```python
for param in progress_head.parameters():
    param.grad = param.grad / λ
```

This ensures:
- **Encoder** gets: `grad(alignment) + λ × grad(progress)` — the λ scaling helps progress signal compete with alignment
- **Progress Head** gets: `grad(progress)` — natural scale, not blown up by λ

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PROGRESS HEAD TRAINING FLOW                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  For each video in batch:                                                        │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                            │ │
│  │  1. SAMPLE SEGMENTS (tensorops.py)                                         │ │
│  │     ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │     │  samples_per_video = 10                                          │   │ │
│  │     │  frames_per_segment = 5                                          │   │ │
│  │     │  → ~50 training samples per video                                │   │ │
│  │     │                                                                   │   │ │
│  │     │  Stratified Sampling (if enabled):                               │   │ │
│  │     │  ┌─────────┬─────────┬─────────┐                                 │   │ │
│  │     │  │  EARLY  │   MID   │  LATE   │                                 │   │ │
│  │     │  │ 0-0.33  │0.33-0.67│0.67-1.0 │  ← Progress bins                │   │ │
│  │     │  │ ~2 samp │ ~2 samp │ ~1 samp │  ← Per segment                  │   │ │
│  │     │  └─────────┴─────────┴─────────┘                                 │   │ │
│  │     └─────────────────────────────────────────────────────────────────┘   │ │
│  │                              │                                             │ │
│  │                              ▼                                             │ │
│  │  2. FOR EACH SAMPLE: (segment_embeddings, gt_progress)                    │ │
│  │     ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │     │                                                                  │   │ │
│  │     │  Action: [frame 0] [frame 1] [frame 2] ... [frame N-1]          │   │ │
│  │     │              │         │         │              │                │   │ │
│  │     │              └─────────┴─────────┘              │                │   │ │
│  │     │                    │                            │                │   │ │
│  │     │           segment_embeddings              target_frame           │   │ │
│  │     │           (action_start to t)                   │                │   │ │
│  │     │                    │                            │                │   │ │
│  │     │                    ▼                            ▼                │   │ │
│  │     │            gt_progress = (t - action_start + 1) / action_length │   │ │
│  │     │                                                                  │   │ │
│  │     └─────────────────────────────────────────────────────────────────┘   │ │
│  │                              │                                             │ │
│  │                              ▼                                             │ │
│  │  3. PROGRESS HEAD FORWARD PASS                                            │ │
│  │     ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │     │                                                                  │   │ │
│  │     │  segment_embeddings (T × 128)                                   │   │ │
│  │     │         │                                                        │   │ │
│  │     │         ▼                                                        │   │ │
│  │     │  ┌─────────────────┐                                            │   │ │
│  │     │  │ Position Encode │  positions = [0, 1/T, 2/T, ..., (T-1)/T]   │   │ │
│  │     │  └─────────────────┘                                            │   │ │
│  │     │         │                                                        │   │ │
│  │     │         ▼                                                        │   │ │
│  │     │  ┌─────────────────┐                                            │   │ │
│  │     │  │   Concatenate   │  x = [embeddings, positions] → (T × 129)   │   │ │
│  │     │  └─────────────────┘                                            │   │ │
│  │     │         │                                                        │   │ │
│  │     │         ▼                                                        │   │ │
│  │     │  ┌─────────────────┐                                            │   │ │
│  │     │  │   GRU Layer     │  Input: (1, T, 129)                        │   │ │
│  │     │  │   (with h0)     │  h0: learnable (1, 1, 64)                  │   │ │
│  │     │  └─────────────────┘                                            │   │ │
│  │     │         │                                                        │   │ │
│  │     │         ▼  h_n (final hidden state)                             │   │ │
│  │     │  ┌─────────────────┐                                            │   │ │
│  │     │  │   FC Layers     │  64 → 32 → 1                               │   │ │
│  │     │  │   + Sigmoid     │                                            │   │ │
│  │     │  └─────────────────┘                                            │   │ │
│  │     │         │                                                        │   │ │
│  │     │         ▼                                                        │   │ │
│  │     │    pred_progress ∈ [0, 1]                                       │   │ │
│  │     │                                                                  │   │ │
│  │     └─────────────────────────────────────────────────────────────────┘   │ │
│  │                              │                                             │ │
│  │                              ▼                                             │ │
│  │  4. COMPUTE WEIGHTED LOSS                                                 │ │
│  │     ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │     │                                                                  │   │ │
│  │     │  weight = min(1 / gt_progress, weight_cap)                      │   │ │
│  │     │                                                                  │   │ │
│  │     │  Examples:                                                       │   │ │
│  │     │    gt_progress = 0.1  →  weight = 10.0 (capped)                 │   │ │
│  │     │    gt_progress = 0.2  →  weight = 5.0                           │   │ │
│  │     │    gt_progress = 0.5  →  weight = 2.0                           │   │ │
│  │     │    gt_progress = 1.0  →  weight = 1.0                           │   │ │
│  │     │                                                                  │   │ │
│  │     │  p_loss = weight × |pred_progress - gt_progress|                │   │ │
│  │     │                                                                  │   │ │
│  │     └─────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                            │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  5. AGGREGATE PROGRESS LOSS                                                      │
│     ┌─────────────────────────────────────────────────────────────────────────┐ │
│     │  avg_progress_loss = Σ(weighted_losses) / Σ(weights)                    │ │
│     │                                                                          │ │
│     │  (Weighted average ensures early frame errors dominate)                  │ │
│     └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Loss Computation Details

### Total Loss Formula
```
total_loss = alignment_loss + λ × progress_loss
```

Where:
- `alignment_loss`: LAV, VAVA, GTCC, or TCC loss (depends on method)
- `progress_loss`: Weighted average of |pred - gt| for all sampled frames
- `λ` (lambda): Scaling factor for progress loss (default: configurable via `--progress_lambda`)

### Loss Scales (Typical Values)

| Method | Alignment Loss Scale | Progress Loss Scale | Recommended λ |
|--------|---------------------|--------------------|--------------|
| GTCC   | 40,000 - 250,000    | 0.1 - 0.5          | 100,000 - 1,000,000 |
| LAV    | 1,000,000 - 100,000,000 | 0.1 - 0.5      | 1,000,000 |
| VAVA   | 1,000 - 100,000     | 0.1 - 0.5          | 10,000 - 100,000 |
| TCC    | 1,000 - 50,000      | 0.1 - 0.5          | 10,000 - 100,000 |

### Why Large Lambda?
Progress loss is typically 0.1-0.5 (L1 error between predictions and GT in [0,1] range). Alignment losses are much larger. To ensure progress learning contributes meaningfully to gradients, we scale it up with a large λ.

---

## Gradient Scaling for Progress Head

### The Problem
When we compute:
```
total_loss = alignment_loss + λ × progress_loss
```

During `loss.backward()`, the progress_head receives gradients scaled by λ:
```
grad(progress_head) = λ × ∂(progress_loss)/∂(progress_head)
```

If λ = 1,000,000, the progress_head would receive gradients 1M× larger than intended!

### The Solution (alignment_training_loop.py, lines 421-432)

After `loss.backward()` but before `optimizer.step()`, we rescale:

```python
# Rescale progress_head gradients to undo the lambda scaling
if progress_config.get('enabled', False) and progress_config.get('method') == 'learnable':
    progress_lambda = progress_config.get('lambda_fixed', 1.0)
    model_to_check = model.module if hasattr(model, 'module') else model
    if hasattr(model_to_check, 'progress_head') and progress_lambda != 1.0 and progress_lambda > 0:
        for param in model_to_check.progress_head.parameters():
            if param.grad is not None:
                param.grad = param.grad / progress_lambda
```

### Gradient Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GRADIENT FLOW                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  total_loss = alignment_loss + λ × progress_loss                            │
│                                                                              │
│  After loss.backward():                                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                        │ │
│  │  Encoder (base_model, heads, attention):                               │ │
│  │    grad = ∂(alignment)/∂(encoder) + λ × ∂(progress)/∂(encoder)        │ │
│  │    [KEEPS λ scaling - alignment loss dominates, progress provides      │ │
│  │     additional gradient signal to learn progress-aware embeddings]     │ │
│  │                                                                        │ │
│  │  Progress Head:                                                        │ │
│  │    grad = λ × ∂(progress)/∂(progress_head)                            │ │
│  │          ↓                                                             │ │
│  │    grad = grad / λ    ← RESCALING APPLIED                             │ │
│  │          ↓                                                             │ │
│  │    grad = ∂(progress)/∂(progress_head)    [Natural scale]             │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Then: gradient clipping (max_norm=0.00001) + optimizer.step()              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Configuration Reference

### Config File: `configs/generic_config.py`

```python
CONFIG.PROGRESS_LOSS = edict({
    'enabled': False,                    # Enable progress loss
    'method': 'cumulative_l2',           # 'cumulative_l2' or 'learnable'
    'lambda_fixed': 0.1,                 # λ scaling (override with --progress_lambda)
    'learnable': {
        'hidden_dim': 64,                # GRU hidden dimension
        'use_gru': True,                 # Use GRU (True) or mean pooling (False)
        'min_segment_len': 3,            # Minimum frames for a valid segment
        'samples_per_video': 10,         # Segments to sample per video
        'frames_per_segment': 5,         # Target frames per segment
        'stratified_sampling': True,     # Ensure early/mid/late coverage
        'weighted_loss': True,           # Weight early frame errors more
        'weight_cap': 10.0,              # Maximum weight (prevents instability)
    },
})
```

### Training Command Example

```bash
CUDA_VISIBLE_DEVICES=0 python multitask_train.py 1 \
    --lav \
    --egoprocel \
    --resnet \
    --mcn \
    --progress_loss learnable \
    --progress_lambda 1000000.0
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `models/alignment_training_loop.py` | Main training loop, gradient rescaling |
| `models/model_multiprong.py` | Model architecture, ProgressHead class |
| `utils/loss_entry.py` | Loss computation, sampling integration |
| `utils/tensorops.py` | Stratified sampling function |
| `configs/generic_config.py` | Default configuration values |

---

## Key Training Features

### 1. Position Encoding
- Each frame embedding is concatenated with its normalized position (0 to ~1)
- Helps GRU understand "where" it is in the sequence
- Implementation: `ProgressHead.forward()` in `model_multiprong.py`

### 2. Learnable Initial Hidden State (h0)
- GRU starts with learned h0 instead of zeros
- Reduces Sigmoid bias toward 0.5 at early frames
- Implementation: `self.h0 = nn.Parameter(torch.zeros(1, 1, hidden_dim))`

### 3. Stratified Sampling
- Ensures coverage of early (0-0.33), mid (0.33-0.67), late (0.67-1.0) progress values
- Prevents bias toward middle-range progress values
- Implementation: `sample_action_segment_with_multiple_frames()` in `tensorops.py`

### 4. Weighted Loss
- Early frames (low GT progress) get higher weight: `weight = 1/gt_progress`
- Capped at `weight_cap` (default: 10) to prevent instability
- Implementation: `loss_entry.py` lines 180-191

### 5. Gradient Rescaling
- Progress head gradients are divided by λ after backward pass
- Ensures progress head trains at natural scale while encoder gets boosted gradient
- Implementation: `alignment_training_loop.py` lines 421-432
