# Plan: Improve Progress Head Loss Formulation

## Context

V6 (GRU) and V7 (Transformer) progress-head-only training is currently running (train8/train9). Both show decreasing train loss but val loss is plateauing around ~0.34 after 5-6 epochs. The gradient clipping fix and progress-head-only training mode are already implemented and working. Now the question is whether the **loss formulation itself** is holding back learning.

**Timing**: Wait for train8/train9 to finish all 50 epochs as baseline, then re-run with new loss for comparison.

## Problem: Current Loss Creates a "Predict Low" Bias

The current loss (`utils/loss_entry.py:260-384`) has three components that interact poorly:

### 1. Weighted L1 Loss (`weight = 1/gt_prog`, capped at 20)
- At `gt_prog=0.05` (early frame) → weight = **20x**
- At `gt_prog=0.5` (mid frame) → weight = **2x**
- At `gt_prog=1.0` (end frame) → weight = **1x**

### 2. Boundary Loss — First Frame (`boundary_weight=5.0`)
- Target: `1/action_len` (e.g., 0.033 for a 30-frame action) → pushes first-frame toward ~0

### 3. Boundary Loss — Last Frame (`boundary_weight=5.0`)
- Target: `1.0` → pushes last-frame toward 1.0

### Why This is a Problem

The weighted loss makes early errors **20x costlier** than late errors. The model's optimal strategy is to **predict conservatively low values everywhere** — being wrong at `gt=0.9` (weight=1.1x) costs far less than being wrong at `gt=0.1` (weight=10x). This explains V6 saturating at ~0.42.

The weighted average (`total_progress_loss / total_weight`) further skews toward minimizing early-frame error.

---

## Three Configurable Loss Formulations

All three options will be implemented and selectable via a new `--progress_loss_mode` CLI flag.

### Option A: `uniform_mono` — Uniform L1 + Monotonicity + Endpoint (Default)

**Component 1: Uniform L1 Loss**
```python
p_loss = |pred - gt_progress|  # weight=1.0 for ALL samples
# Average by num_samples (not total_weight)
```

**Component 2: Monotonicity Penalty (NEW)**
For multiple target frames from the **same action**, penalize violations of monotonic increase:
```python
# frame_samples from same action, sorted by gt_prog
for consecutive (pred_early, pred_late):
    mono_loss += max(0, pred_early - pred_late + margin)  # margin=0.01
```
- Weight: `monotonicity_weight = 2.0`
- Directly encodes the structural prior that progress must increase

**Component 3: Endpoint Regularization (replaces boundary loss)**
Only supervise the **last frame** of each action (no first-frame supervision):
```python
full_segment = embeddings[action_start : action_end+1]
pred_end = progress_head(full_segment)
endpoint_loss += |pred_end - 1.0|
```
- Weight: `endpoint_weight = 1.0`
- No first-frame supervision → removes the double penalty on early frames

**Combined**: `total = l1_avg + mono_weight * mono_avg + endpoint_weight * endpoint_avg`

### Option B: `sqrt_weighted` — Sqrt Weighting (Softer Version)

Keep same structure as current loss but with gentler weighting:
```python
weight = 1.0 / sqrt(max(gt_prog, 1.0 / weight_cap))
weight = min(weight, weight_cap)  # weight_cap=5.0 (was 20)
```
- At `gt_prog=0.05` → weight = **4.5x** (was 20x)
- At `gt_prog=0.5` → weight = **1.4x** (was 2x)
- At `gt_prog=1.0` → weight = **1.0x** (same)

Keep boundary loss but reduce `boundary_weight` from 5.0 to 1.0. Average by `num_samples` (not `total_weight`).

### Option C: `mse` — MSE Loss

Replace L1 with MSE:
```python
p_loss = (pred - gt_progress)^2
```
- MSE naturally penalizes large errors more — if model predicts 0.3 for gt=0.9, the squared error is 0.36 (vs L1 = 0.6)
- No weighting needed, no boundary loss needed
- Add endpoint regularization (same as Option A) for extra signal
- Simplest formulation

### Comparison Table

| Aspect | A: uniform_mono | B: sqrt_weighted | C: mse |
|--------|-----------------|------------------|--------|
| Early-frame bias | None | Mild (4.5x max) | None |
| Monotonicity | Explicit penalty | None | None |
| Endpoint supervision | Yes (1.0 weight) | Yes (boundary 1.0) | Yes (1.0 weight) |
| First-frame supervision | No | Yes (boundary 1.0) | No |
| Complexity | Medium | Low (minimal change) | Low |
| Structural prior | Strong | Weak | None |

---

## Files to Modify

### 1. `utils/loss_entry.py`

**In `get_progress_only_loss_function()` (~line 260-384) and `get_loss_function()` learnable branch (~line 141-254):**

Replace the current progress loss computation with a dispatch based on `progress_loss_mode`:

```python
progress_loss_mode = learnable_config.get('progress_loss_mode', 'uniform_mono')

if progress_loss_mode == 'uniform_mono':
    # Option A: Uniform L1 + monotonicity + endpoint
    ...
elif progress_loss_mode == 'sqrt_weighted':
    # Option B: Sqrt weighting + reduced boundary
    ...
elif progress_loss_mode == 'mse':
    # Option C: MSE + endpoint
    ...
else:
    # Fallback: legacy weighted L1 + boundary (current behavior)
    ...
```

The legacy behavior remains accessible via `progress_loss_mode='legacy'` or if `weighted_loss=True` is explicitly set.

### 2. `configs/generic_config.py`

Add to `CONFIG.PROGRESS_LOSS.learnable`:
```python
'progress_loss_mode': 'uniform_mono',  # NEW: 'uniform_mono', 'sqrt_weighted', 'mse', 'legacy'
'monotonicity_loss': True,             # NEW (Option A)
'monotonicity_weight': 2.0,            # NEW (Option A)
'monotonicity_margin': 0.01,           # NEW (Option A)
'endpoint_loss': True,                 # NEW (Options A & C)
'endpoint_weight': 1.0,               # NEW (Options A & C)
```

Change defaults:
```python
'weighted_loss': False,    # was True — now off by default
'boundary_loss': False,    # was True — now off by default
```

### 3. `utils/parser_util.py`

Add new CLI argument:
```python
parser.add_argument('--progress_loss_mode',
    type=str, default='uniform_mono',
    choices=['uniform_mono', 'sqrt_weighted', 'mse', 'legacy'],
    help='Progress loss formulation. Options: uniform_mono (default), sqrt_weighted, mse, legacy')
```

Add to return dict: `'progress_loss_mode': args.progress_loss_mode`

### 4. `configs/entry_config.py`

Propagate to config:
```python
progress_loss_mode = args_given.get('progress_loss_mode', 'uniform_mono')
CONFIG.PROGRESS_LOSS['learnable']['progress_loss_mode'] = progress_loss_mode
```

### 5. `utils/tensorops.py`

In `sample_action_segment_with_multiple_frames` (~line 516): Sort `target_indices` before the results loop so results are ordered by time (needed for monotonicity computation):
```python
target_indices.sort()  # Add this line before the results loop
```

---

## Implementation Detail: Monotonicity Penalty

The key change is processing `frame_samples` per-action rather than independently:

```python
# For each video, for each sampled action:
action_preds = []
for seg_emb, gt_prog in frame_samples:  # all from same action, sorted by target
    if seg_emb is not None and seg_emb.shape[0] >= 2:
        pred_prog = progress_head(seg_emb)
        gt_tensor = torch.tensor(gt_prog, device=pred_prog.device, dtype=pred_prog.dtype)

        # Uniform L1
        total_l1_loss += torch.abs(pred_prog - gt_tensor)
        num_samples += 1
        action_preds.append((gt_prog, pred_prog))

# Monotonicity: consecutive predictions should increase
action_preds.sort(key=lambda x: x[0])
for j in range(1, len(action_preds)):
    pred_earlier = action_preds[j-1][1]
    pred_later = action_preds[j][1]
    violation = pred_earlier - pred_later + margin
    mono_loss += torch.clamp(violation, min=0)
    num_mono_pairs += 1
```

---

## Usage

### After train8/train9 finish, re-run with new loss:

**Option A (recommended, default):**
```bash
CUDA_VISIBLE_DEVICES=5 python multitask_train.py 1 --gtcc --egoprocel --resnet --mcn \
  --progress_loss learnable --progress_arch transformer --progress_lambda 500000.0 \
  --train_progress_only \
  --alignment_checkpoint <path_to_v7_alignment_ckpt> \
  --progress_lr 0.001 --progress_epochs 50 \
  --progress_loss_mode uniform_mono
```

**Option B:**
```bash
... --progress_loss_mode sqrt_weighted
```

**Option C:**
```bash
... --progress_loss_mode mse
```

**Legacy (current behavior):**
```bash
... --progress_loss_mode legacy
```

---

## Verification

1. **Baseline**: train8/train9 finish with old loss → run `extract_progress.py` → record progress curves
2. **Option A**: Re-run V7 with `--progress_loss_mode uniform_mono` → compare progress curves
3. **Quick A/B/C**: Run all three options for 10 epochs each, compare val loss trajectories
4. **Full run**: Best option for 50 epochs → `extract_progress.py` → expect progress reaching 0.8-1.0
5. **Monotonicity check**: Progress values should strictly increase within each action

## Expected Outcomes

- **Option A (uniform_mono)**: Best for learning full 0→1 curve — monotonicity penalty directly encodes what we want
- **Option B (sqrt_weighted)**: Moderate improvement — less extreme than current but still some early-frame bias
- **Option C (mse)**: Good for penalizing large errors at high progress — simplest change
- All should outperform legacy loss (current train8/train9 baseline)
