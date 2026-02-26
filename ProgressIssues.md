# Progress Head Learning Issue - Diagnosis and Solutions

## Problem Statement
VAVA (and likely GTCC, TCC, LAV) with learnable progress head:
- Progress loss is low (~0.25)
- But predictions start at ~0.64 and stay around 0.54-0.64
- Expected: start at ~0 and increase to ~1

## Root Cause Analysis

### Issue 1: GRU Has No Context at Early Frames
```
Frame 1: GRU sees [emb_1] → 1 frame only → random output biased to ~0.5
Frame 2: GRU sees [emb_1, emb_2] → 2 frames → still minimal context
Frame 3: GRU sees [emb_1, emb_2, emb_3] → barely enough
```

The GRU hidden state with 1-3 frames has almost no temporal information. The Sigmoid output layer naturally biases toward 0.5 when inputs are uncertain.

### Issue 2: Loss is Deceptively Low
```python
# At frame 1 of 80-frame action:
GT_progress = 1/80 = 0.0125
Predicted = 0.54
Error = |0.54 - 0.0125| = 0.53

# At frame 40:
GT_progress = 40/80 = 0.5
Predicted = 0.52
Error = |0.52 - 0.5| = 0.02  # Accidentally good!

# Average loss appears low because middle frames accidentally align
```

### Issue 3: GRU Doesn't Know "Where" It Is
The GRU only sees embeddings - it has no explicit signal about:
- How many frames it has seen
- What fraction of the action has elapsed
- The expected duration of actions

---

## Proposed Solutions

### Solution A: Add Positional Encoding to Input (Recommended)
**Idea**: Concatenate frame position information to each embedding

```python
class ProgressHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, use_gru=True):
        # Add +1 for position feature
        self.gru = nn.GRU(input_dim + 1, hidden_dim, ...)  # 129 → 64

    def forward(self, segment_embeddings):
        T = segment_embeddings.shape[0]
        # Normalized position: 0, 1/T, 2/T, ..., (T-1)/T
        positions = torch.arange(T, device=segment_embeddings.device).float() / max(T, 1)
        positions = positions.unsqueeze(1)  # (T, 1)
        x = torch.cat([segment_embeddings, positions], dim=1)  # (T, D+1)
        ...
```

**Why it helps**: GRU now knows "I've seen 1 frame" vs "I've seen 50 frames"

---

### Solution B: Learnable Initial Hidden State
**Idea**: Instead of zero-initializing GRU hidden state, learn it

```python
class ProgressHead(nn.Module):
    def __init__(self, ...):
        self.h0 = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(self, segment_embeddings):
        x = segment_embeddings.unsqueeze(0)
        _, h_n = self.gru(x, self.h0)  # Use learned h0
        ...
```

**Why it helps**: Better initialization reduces random bias at early frames

---

### Solution C: Increase Minimum Segment Length
**Idea**: Don't train on segments < 10 frames

```python
# In config:
'min_segment_len': 10  # Was 3
```

**Why it helps**: Ensures GRU always has enough context
**Downside**: May hurt generalization to short actions

---

### Solution D: Use Relative Progress Loss
**Idea**: Instead of predicting absolute progress, predict relative change

```python
# Current: pred = absolute_progress(0 to 1)
# New: pred = how much progress since last frame

# This naturally starts at small values
```

**Downside**: Accumulation errors; more complex inference

---

### Solution E: Weighted Loss (Penalize Early Frame Errors More)
**Idea**: Early frames matter more for qualitative results

```python
# Weight by inverse of GT progress (capped)
weight = 1.0 / max(gt_progress, 0.1)
loss = weight * |pred - gt|
```

**Why it helps**: Forces model to get early frames right

---

## Affected Models
This fix applies to ALL alignment methods with learnable progress:
- VAVA + learnable
- GTCC + learnable
- TCC + learnable
- LAV + learnable

The `ProgressHead` is shared across all of them.

---

## Files to Modify

| File | Change |
|------|--------|
| `models/model_multiprong.py` | Modify `ProgressHead` class with chosen solution(s) |
| `configs/generic_config.py` | (Solution C only) Change min_segment_len |

---

## Implementation Details for All Solutions

### Solution A: Position Encoding

**File**: `models/model_multiprong.py`

```python
class ProgressHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, use_gru=True):
        super(ProgressHead, self).__init__()
        self.use_gru = use_gru
        self.input_dim = input_dim

        if use_gru:
            # +1 for position encoding
            self.gru = nn.GRU(input_dim + 1, hidden_dim, num_layers=1,
                              batch_first=True, bidirectional=False)
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(input_dim + 1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

    def forward(self, segment_embeddings):
        T = segment_embeddings.shape[0]
        device = segment_embeddings.device

        # Position encoding: normalized frame index (0 to 1)
        positions = torch.arange(T, device=device, dtype=torch.float32) / max(T, 1)
        positions = positions.unsqueeze(1)  # (T, 1)
        x = torch.cat([segment_embeddings, positions], dim=1)  # (T, D+1)

        if self.use_gru:
            x = x.unsqueeze(0)  # (1, T, D+1)
            _, h_n = self.gru(x)
            progress = self.fc(h_n.squeeze())
        else:
            x = x.mean(dim=0)
            progress = self.fc(x)

        return progress.squeeze()
```

---

### Solution B: Learnable Initial Hidden State

**File**: `models/model_multiprong.py`

```python
class ProgressHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, use_gru=True):
        super(ProgressHead, self).__init__()
        self.use_gru = use_gru

        if use_gru:
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1,
                              batch_first=True, bidirectional=False)
            # Learnable initial hidden state
            self.h0 = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

    def forward(self, segment_embeddings):
        if self.use_gru:
            x = segment_embeddings.unsqueeze(0)  # (1, T, D)
            _, h_n = self.gru(x, self.h0)  # Use learnable h0 instead of zeros
            progress = self.fc(h_n.squeeze())
        else:
            x = segment_embeddings.mean(dim=0)
            progress = self.fc(x)
        return progress.squeeze()
```

---

### Solution A+B Combined (Recommended)

**File**: `models/model_multiprong.py`

```python
class ProgressHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, use_gru=True):
        super(ProgressHead, self).__init__()
        self.use_gru = use_gru
        self.input_dim = input_dim

        if use_gru:
            # +1 for position encoding
            self.gru = nn.GRU(input_dim + 1, hidden_dim, num_layers=1,
                              batch_first=True, bidirectional=False)
            # Learnable initial hidden state
            self.h0 = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(input_dim + 1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

    def forward(self, segment_embeddings):
        T = segment_embeddings.shape[0]
        device = segment_embeddings.device

        # Position encoding: normalized frame index
        positions = torch.arange(T, device=device, dtype=torch.float32) / max(T, 1)
        positions = positions.unsqueeze(1)  # (T, 1)
        x = torch.cat([segment_embeddings, positions], dim=1)  # (T, D+1)

        if self.use_gru:
            x = x.unsqueeze(0)  # (1, T, D+1)
            _, h_n = self.gru(x, self.h0)  # Use learnable h0
            progress = self.fc(h_n.squeeze())
        else:
            x = x.mean(dim=0)
            progress = self.fc(x)

        return progress.squeeze()
```

---

### Solution C: Increase Minimum Segment Length

**File**: `configs/generic_config.py` (or via command line / config.json)

```python
CONFIG.PROGRESS_LOSS = edict({
    ...
    'learnable': {
        'min_segment_len': 10,  # Was 3, now require at least 10 frames
        ...
    },
})
```

**Pros**: Simple config change, no code modification
**Cons**: May hurt generalization to short actions (< 10 frames)

---

### Solution D: Relative Progress Prediction

**File**: `models/model_multiprong.py` + `utils/loss_entry.py`

This requires more significant changes:
1. Predict delta_progress per frame instead of absolute progress
2. Accumulate predictions during inference
3. Change loss computation to use delta GT

```python
# In ProgressHead.forward():
# Instead of outputting progress directly, output delta
delta = self.fc(h_n.squeeze())  # Small value representing progress increment

# In loss computation:
# GT becomes: delta_gt = 1 / action_length (constant per action)
# Loss = |pred_delta - delta_gt|

# In inference:
# Accumulate: progress[t] = sum(delta[0:t])
```

**Pros**: Naturally produces small values at start
**Cons**: Accumulation errors; more complex implementation

---

### Solution E: Weighted Loss

**File**: `utils/loss_entry.py`

```python
# In progress loss computation:
for seg_emb, gt_prog in frame_samples:
    pred_prog = progress_head(seg_emb)

    # Weight inversely by GT (penalize early frame errors more)
    weight = 1.0 / max(gt_prog, 0.05)  # Cap at 20x
    weight = min(weight, 20.0)  # Safety cap

    p_loss = weight * torch.abs(pred_prog - gt_tensor)
    total_progress_loss += p_loss
```

**Pros**: Forces model to get early frames right
**Cons**: May destabilize training if weights too extreme

---

## Verification Plan

1. Apply chosen solution(s) to `models/model_multiprong.py`
2. Train model with learnable progress (example with VAVA):
   ```bash
   CUDA_VISIBLE_DEVICES=X python multitask_train.py 1 --vava --egoprocel --resnet --mcn --progress_loss learnable --progress_lambda 1000000.0
   ```
3. Update visualization script if needed (for position encoding, input dim changes)
4. Run visualization to check qualitative results:
   ```bash
   cd /vision/anishn/progress_visualization/scripts
   python quick_vava_viz.py --num_segments 5
   ```
5. **Expected behavior after fix**:
   - Frame 1: Progress ≈ 0.01-0.05 (near zero)
   - Middle frames: Progress increases smoothly
   - Final frame: Progress ≈ 0.95-1.0

## Summary of Solutions

| Solution | Complexity | Risk | Expected Impact |
|----------|------------|------|-----------------|
| A: Position encoding | Low | Low | High - gives GRU explicit frame count |
| B: Learnable h0 | Very Low | Very Low | Medium - better initialization |
| A+B: Combined | Low | Low | Highest - recommended |
| C: Min segment len | Config only | Medium | Medium - quick fix but may hurt generalization |
| D: Relative progress | High | Medium | High - but requires more changes |
| E: Weighted loss | Low | Medium | Medium - may destabilize training |

**Recommendation**: Start with A+B combined. If still not working, add Solution E (weighted loss).
