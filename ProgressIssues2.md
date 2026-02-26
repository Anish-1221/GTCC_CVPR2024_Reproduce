# Fix ProgressHead with Position Encoding + Learnable h0

## Problem
LAV learnable progress model outputs range 0.65 → 0.8 instead of 0 → 1.
The model learns temporal dynamics (progress increases) but has compressed output range.

## Root Cause (from ProgressIssues.md)
1. GRU has no explicit context about frame position
2. Sigmoid output naturally biases toward 0.5 with uncertain inputs
3. GRU doesn't know "where" it is in the action

## Solution: A+B Combined
Modify `ProgressHead` class with:
1. **Position Encoding**: Concatenate normalized frame index to embeddings
2. **Learnable h0**: Learn initial hidden state instead of zero-init

## File to Modify
`/vision/anishn/GTCC_CVPR2024/models/model_multiprong.py`

## Implementation

### Current ProgressHead (lines ~144-180):
```python
class ProgressHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, use_gru=True):
        self.gru = nn.GRU(input_dim, hidden_dim, ...)  # No position
        # h0 defaults to zeros
```

### New ProgressHead:
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

        # Position encoding: normalized frame index (0 to ~1)
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

## Key Changes
| Aspect | Before | After |
|--------|--------|-------|
| GRU input dim | `input_dim` (128) | `input_dim + 1` (129) |
| Hidden state init | Zeros (default) | Learnable `self.h0` parameter |
| Forward pass | Raw embeddings | Embeddings + position encoding |

## Retraining Required
Yes - the model architecture changes, so existing checkpoints won't load.

**Training command** (example for LAV):
```bash
CUDA_VISIBLE_DEVICES=X python multitask_train.py 1 --lav --egoprocel --resnet --mcn --progress_loss learnable --progress_lambda 1000000.0
```

## Verification
1. Modify `ProgressHead` in `models/model_multiprong.py`
2. Retrain LAV with learnable progress
3. Run `quick_lav_viz.py` to visualize
4. **Expected**: Progress starts near 0, increases to near 1

## Why This Should Work
- Position encoding tells GRU "I'm at frame 1/T" vs "I'm at frame T/T"
- Learnable h0 reduces initial Sigmoid bias
- Your LAV already learns increasing dynamics (0.65→0.8)
- This fix gives the model the context needed to map to correct 0-1 range

---

## Compatibility with Existing Training Loop

**CONFIRMED: Solution A+B works with the existing training code without modifications.**

### Training Flow (unchanged)
1. `alignment_training_loop.py`: Calls `loss_fn(output_dict, epoch, times=times)`
2. `loss_entry.py` (lines 141-182): For learnable method:
   ```python
   progress_head = output_dict['progress_head']
   frame_samples = sample_action_segment_with_multiple_frames(vid_emb, times[vid_idx], ...)
   for seg_emb, gt_prog in frame_samples:
       pred_prog = progress_head(seg_emb)  # <-- Interface unchanged
       loss = |pred_prog - gt_progress|
   ```
3. `tensorops.py` (lines 411-464): Sampling returns `(segment_embeddings, gt_progress)` pairs

### What Changes vs What Stays Same
| Aspect | Status | Notes |
|--------|--------|-------|
| Training loop | **No change** | `alignment_training_loop.py` unchanged |
| Loss function | **No change** | `loss_entry.py` unchanged |
| Sampling | **No change** | `tensorops.py` unchanged |
| ProgressHead interface | **Same** | `forward(seg_emb)` still works |
| ProgressHead internals | **Changed** | Position encoding + learnable h0 |
| Existing checkpoints | **Incompatible** | GRU dim changed, must retrain |

### Gradient Scaling (already handled)
The training loop already has special handling for progress_head gradients (lines 419-430):
```python
if hasattr(model_to_check, 'progress_head') and progress_lambda != 1.0:
    for param in model_to_check.progress_head.parameters():
        if param.grad is not None:
            param.grad = param.grad / progress_lambda
```

This continues to work with the modified ProgressHead.

## Causal Inference Consideration
In causal/online mode, at frame t we feed frames [1..t] and position encoding becomes:
```
positions = [0, 1/t, 2/t, ..., (t-1)/t]
```

The model learns that:
- "position=1 with small t" + "certain embedding patterns" = low progress (early in action)
- "position=1 with large t" + "certain embedding patterns" = high progress (late in action)

The GRU hidden state accumulates information about the sequence length and embedding evolution.

---

## Training Improvements

In addition to the ProgressHead architecture changes, we improved the training setup to provide better learning signal.

### Problem: Sampling Bias
The original sampling (tensorops.py line 442) had a critical issue:
```python
min_target = action_start + min_segment_len - 1  # Earliest target = frame 2
```

This means:
- For a 10-frame action: earliest GT progress = 3/10 = **0.30**
- For a 20-frame action: earliest GT progress = 3/20 = **0.15**
- The model rarely sees GT progress values < 0.15!

This explains why the model outputs start high - it never learned what "near-zero progress" looks like.

### Solution: Three Training Improvements

#### 1. More Training Samples
**File**: `configs/generic_config.py`

| Parameter | Before | After |
|-----------|--------|-------|
| `samples_per_video` | 5 | **10** |
| `frames_per_segment` | 3 | **5** |
| Samples per video | ~15 | **~50** |

#### 2. Stratified Sampling
**File**: `utils/tensorops.py` - `sample_action_segment_with_multiple_frames()`

New parameter `stratified=True` ensures samples from:
- **Early** (progress 0.0 - 0.33)
- **Mid** (progress 0.33 - 0.67)
- **Late** (progress 0.67 - 1.0)

```python
# Split possible targets into three bins based on progress value
early_targets = []  # progress 0.0 - 0.33
mid_targets = []    # progress 0.33 - 0.67
late_targets = []   # progress 0.67 - 1.0

# Sample from each bin proportionally
samples_per_bin = max(1, frames_per_segment // 3)
```

#### 3. Weighted Loss for Early Frames
**File**: `utils/loss_entry.py`

Early frames (low GT progress) get higher weight to emphasize their importance:
```python
if use_weighted_loss:
    # Weight = 1 / gt_prog, capped to prevent instability
    weight = 1.0 / max(gt_prog, 1.0 / weight_cap)
    weight = min(weight, weight_cap)  # Default cap: 10x
else:
    weight = 1.0

p_loss = weight * torch.abs(pred_prog - gt_tensor)
```

Example weights:
- GT progress = 0.1 → weight = 10.0 (capped)
- GT progress = 0.2 → weight = 5.0
- GT progress = 0.5 → weight = 2.0
- GT progress = 1.0 → weight = 1.0

### New Config Options
```python
CONFIG.PROGRESS_LOSS = edict({
    'enabled': False,
    'method': 'cumulative_l2',
    'lambda_fixed': 0.1,
    'learnable': {
        'hidden_dim': 64,
        'use_gru': True,
        'min_segment_len': 3,
        'samples_per_video': 10,      # Increased from 5
        'frames_per_segment': 5,      # Increased from 3
        'stratified_sampling': True,  # NEW: ensure early/mid/late coverage
        'weighted_loss': True,        # NEW: weight early frame errors more
        'weight_cap': 10.0,           # NEW: max weight (prevents instability)
    },
})
```

### Files Modified
| File | Change |
|------|--------|
| `models/model_multiprong.py` | ProgressHead with position encoding + learnable h0 |
| `configs/generic_config.py` | New training parameters |
| `utils/tensorops.py` | Stratified sampling support |
| `utils/loss_entry.py` | Weighted loss + stratified sampling integration |

### Expected Impact
1. **Position encoding + learnable h0**: Model knows where it is in the sequence
2. **More samples**: 3x more training signal per batch
3. **Stratified sampling**: Guarantees early frame coverage
4. **Weighted loss**: Penalizes early frame errors 10x more

Combined, these changes should produce progress predictions that:
- Start near 0 at action start
- Increase smoothly through the action
- End near 1 at action end
