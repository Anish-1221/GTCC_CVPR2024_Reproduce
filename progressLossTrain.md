# Learnable Progress Head with Action-Level Segment Sampling

## Overview

Add a **learnable neural network progress head** that predicts progress values for randomly sampled action segments during training. This differs fundamentally from the previous `progressLoss.md` plan.

---

## Key Differences from Previous Plan (progressLoss.md)

| Aspect | Previous Plan (Cumulative L2) | This Plan (Learnable Head) |
|--------|------------------------------|---------------------------|
| **Progress prediction** | Cumulative L2 distance between embeddings (no neural network) | Learnable GRU/MLP head that outputs progress |
| **Learnable params** | None for progress computation | Yes - new `ProgressHead` module |
| **What learns** | Encoder learns to produce "progress-like" embeddings indirectly | Progress head explicitly learns to predict progress |
| **Sampling strategy** | Random video, random frame | Random action, then random segment within that action |
| **Ground truth** | Uses `get_trueprogress()` (video-level) or `get_trueprogress_per_action()` | Uses segment-specific progress computed from action boundaries |
| **Gradient flow** | Through cumulative distance computation | Through progress head forward pass |

### Visual Comparison

```
PREVIOUS PLAN (Implicit - Cumulative L2 Distance):
┌─────────┐     ┌─────────┐     ┌──────────────────────┐
│ Features│ ──▶ │ Encoder │ ──▶ │ Embeddings (T, 128)  │
└─────────┘     └─────────┘     └──────────┬───────────┘
                                           │
                           ┌───────────────▼───────────────┐
                           │  Cumulative L2 Distance       │
                           │  (NO learnable params)        │
                           │  Σ ||e[t] - e[t-1]||          │
                           └───────────────┬───────────────┘
                                           ▼
                                    Predicted Progress

THIS PLAN (Explicit - Learnable Head):
┌─────────┐     ┌─────────┐     ┌──────────────────────┐
│ Features│ ──▶ │ Encoder │ ──▶ │ Embeddings (T, 128)  │
└─────────┘     └─────────┘     └──────────┬───────────┘
                                           │
                           ┌───────────────▼───────────────┐
                           │  Sample Action Segment        │
                           │  (Random action, random span) │
                           └───────────────┬───────────────┘
                                           │
                           ┌───────────────▼───────────────┐
                           │  ProgressHead (LEARNABLE)     │
                           │  GRU → Linear → Sigmoid       │
                           └───────────────┬───────────────┘
                                           ▼
                                    Predicted Progress
```

---

## Supervisor's Requirements (Clarified)

1. **Learnable Progress Head**: A neural network that takes segment features and predicts progress
2. **Action-Level Segment Sampling with Random Frame Index**:
   - Sample a random **action** within the video (skip SIL/background)
   - Sample random **start/end frames** within that action (defines the segment)
   - **Sample a random frame INDEX within that segment** (the target frame)
   - Extract the corresponding **segment of embeddings up to that index**
   - Feed to progress head → predict progress at that index
   - **GT progress** = position of that index within the action

### Sampling Visualization:
```
Video:     [---SIL---][-------Action: "cut"-------][---SIL---]
Frames:    0........19  20....................99    100.....150

Step 1: Sample action "cut" (frames 20-99, length=80)
Step 2: Sample segment start=30, end=70
Step 3: Sample random index within [30,70], e.g., idx=55
Step 4: Extract embeddings[30:55+1] (segment up to idx)
Step 5: Progress head predicts progress at frame 55
Step 6: GT progress = (55 - 20 + 1) / 80 = 0.45
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `models/model_multiprong.py` | Add `ProgressHead` class, modify `MultiProngAttDropoutModel` |
| `models/alignment_training_loop.py` | Pass `times` to loss function, dual checkpoint tracking, track alignment_loss separately |
| `utils/loss_entry.py` | Add progress loss computation with segment sampling, track `alignment_loss` key, curriculum schedule |
| `utils/tensorops.py` | Add `sample_action_segment_with_random_index()` helper function |
| `configs/generic_config.py` | Add `CONFIG.PROGRESS_LOSS` section with curriculum options |
| `utils/ckpt_save.py` | Support `checkpoint_type` parameter for loading combined vs alignment-only checkpoints |

---

## Implementation Steps

### Step 1: Add ProgressHead Class

**File:** `models/model_multiprong.py`

Add after line 140 (after imports, before `MultiProngAttDropoutModel`):

```python
class ProgressHead(nn.Module):
    """
    Learnable progress prediction head.
    Takes a segment of embeddings (from start to random target index) and predicts progress.

    Two architecture options (controlled by use_gru config):
    - GRU: Captures temporal ordering (later frames = higher progress)
    - Mean pooling: Simpler, faster, but loses temporal info
    """
    def __init__(self, input_dim=128, hidden_dim=64, use_gru=True):
        super(ProgressHead, self).__init__()
        self.use_gru = use_gru

        if use_gru:
            #############################################
            # OPTION 1: GRU-based (captures temporal order)
            #############################################
            # GRU processes variable-length sequence, final hidden state
            # captures "how far along" the segment is
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1,
                              batch_first=True, bidirectional=False)
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()  # Output in [0, 1]
            )
        else:
            #############################################
            # OPTION 2: Mean Pooling (simpler, faster)
            #############################################
            # Average all frame embeddings, then MLP
            # Loses temporal ordering but still captures segment content
            self.fc = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

    def forward(self, segment_embeddings):
        """
        Args:
            segment_embeddings: (T_segment, D) tensor - embeddings from start to target index

        Returns:
            progress: scalar in [0, 1] - predicted progress at the target index
        """
        if self.use_gru:
            x = segment_embeddings.unsqueeze(0)  # (1, T, D)
            _, h_n = self.gru(x)                  # h_n: (1, 1, hidden_dim)
            progress = self.fc(h_n.squeeze())
        else:
            x = segment_embeddings.mean(dim=0)    # Mean pool: (D,)
            progress = self.fc(x)
        return progress.squeeze()
```

**Architecture Comparison:**
| Option | Pros | Cons |
|--------|------|------|
| **GRU** | Captures temporal order (knows later frames = higher progress) | More parameters, slower |
| **Mean Pooling** | Simpler, faster, fewer params | Loses frame ordering info |

### Step 2: Modify MultiProngAttDropoutModel

**File:** `models/model_multiprong.py` (lines 143-202)

Modify `__init__` to add progress head parameter:

```python
class MultiProngAttDropoutModel(nn.Module):
    def __init__(
        self,
        base_model_class,
        base_model_params,
        output_dimensionality,
        num_heads,
        dropping=False,
        attn_layers=[512, 256],
        drop_layers=[512, 128, 256],
        use_progress_head=False,        # NEW
        progress_head_hidden_dim=64,    # NEW
        progress_head_use_gru=True,     # NEW
    ):
        super(MultiProngAttDropoutModel, self).__init__()
        # ... existing code (lines 154-177) ...

        # NEW: Add progress head
        self.use_progress_head = use_progress_head
        if use_progress_head:
            self.progress_head = ProgressHead(
                input_dim=output_dimensionality,
                hidden_dim=progress_head_hidden_dim,
                use_gru=progress_head_use_gru
            )
```

Modify `forward` return to include progress_head reference:

```python
    def forward(self, videos):
        # ... existing forward code (lines 179-197) ...

        result = {'outputs': outputs, 'attentions': attentions}
        if self.dropping:
            result['dropouts'] = dropouts
        # NEW: Include progress head for use in loss function
        if self.use_progress_head:
            result['progress_head'] = self.progress_head
        return result
```

### Step 3: Add Segment Sampling Function

**File:** `utils/tensorops.py`

Add after `get_trueprogress_per_action` (~line 325):

```python
def sample_action_segment_with_random_index(embeddings, times_dict, min_segment_len=3):
    """
    Sample a random segment within a random non-SIL action, with random target index.

    This implements the supervisor's sampling strategy:
    1. Pick a random non-SIL action
    2. Pick random start/end within that action (defines segment bounds)
    3. Pick a random target INDEX within [start, end]
    4. Return segment embeddings UP TO that index, and GT progress at that index

    Args:
        embeddings: (T, D) tensor - video frame embeddings
        times_dict: dict with 'step', 'start_frame', 'end_frame' keys
        min_segment_len: minimum frames in sampled segment

    Returns:
        segment_embeddings: (T_seg, D) tensor - embeddings from seg_start to target_idx
        gt_progress: float in [0, 1] - progress at target_idx within the action
        action_name: str - sampled action name
        OR (None, None, None) if no valid action found
    """
    import random

    # Find non-SIL actions with sufficient length
    valid_actions = []
    for idx, (step, start, end) in enumerate(zip(
        times_dict['step'], times_dict['start_frame'], times_dict['end_frame']
    )):
        action_length = end - start + 1
        if step not in ['SIL', 'background'] and action_length >= min_segment_len:
            valid_actions.append((idx, step, start, end))

    if len(valid_actions) == 0:
        return None, None, None

    # Step 1: Sample random action
    _, action_name, action_start, action_end = random.choice(valid_actions)
    action_length = action_end - action_start + 1

    # Step 2: Sample random segment bounds within the action
    max_start = action_end - min_segment_len + 1
    if max_start < action_start:
        max_start = action_start

    seg_start = random.randint(action_start, max_start)
    seg_end = random.randint(seg_start + min_segment_len - 1, action_end)

    # Step 3: Sample random TARGET INDEX within [seg_start, seg_end]
    target_idx = random.randint(seg_start, seg_end)

    # Clamp to valid embedding range
    T = embeddings.shape[0]
    seg_start = max(0, min(seg_start, T - 1))
    target_idx = max(seg_start, min(target_idx, T - 1))

    # Step 4: Extract segment embeddings FROM seg_start TO target_idx (inclusive)
    segment_embeddings = embeddings[seg_start:target_idx + 1]

    # Step 5: Compute GT progress at target_idx within this action
    # Progress = (position within action) / (action length)
    # Position at target_idx = target_idx - action_start + 1
    progress_at_target = (target_idx - action_start + 1) / action_length
    gt_progress = min(1.0, max(0.0, progress_at_target))

    return segment_embeddings, gt_progress, action_name
```

### Step 4: Modify Training Loop to Pass Times

**File:** `models/alignment_training_loop.py`

Find the loss function call (~line 341-346) and modify:

```python
# BEFORE:
loss_dict = loss_fn(output_dict, epoch)

# AFTER:
loss_dict = loss_fn(output_dict, epoch, times=times)
```

### Step 5: Modify Loss Function

**File:** `utils/loss_entry.py`

Add imports at top:

```python
from utils.tensorops import sample_action_segment_with_random_index
```

Modify `get_loss_function` to handle progress loss:

```python
def get_loss_function(config_obj):
    loss_booldict = config_obj.LOSS_TYPE
    TCC_ORIGINAL_PARAMS = config_obj.TCC_ORIGINAL_PARAMS
    GTCC_PARAMS = config_obj.GTCC_PARAMS
    LAV_PARAMS = config_obj.LAV_PARAMS
    VAVA_PARAMS = config_obj.VAVA_PARAMS

    # NEW: Progress loss config
    PROGRESS_CONFIG = getattr(config_obj, 'PROGRESS_LOSS', {'enabled': False})
    use_progress_loss = PROGRESS_CONFIG.get('enabled', False)
    progress_coef = PROGRESS_CONFIG.get('coefficient', 0.1)
    progress_loss_type = PROGRESS_CONFIG.get('loss_type', 'l1')
    min_seg_len = PROGRESS_CONFIG.get('min_segment_len', 3)

    def _alignment_loss_fn(output_dict_list, epoch, times=None):  # ADD times param
        if type(output_dict_list) != list:
            output_dict_list = [output_dict_list]

        loss_return_dict = {}
        loss_return_dict['total_loss'] = torch.tensor(0).float().to(device)

        # NEW: Track progress loss separately
        if use_progress_loss:
            loss_return_dict['progress_loss'] = torch.tensor(0).float().to(device)
            progress_count = 0

        for loss_term, verdict in loss_booldict.items():
            if verdict:
                loss_return_dict[loss_term + '_loss'] = torch.tensor(0).float().to(device)

        for idx, output_dict in enumerate(output_dict_list):
            if len(output_dict['outputs']) < 2:
                continue

            # Existing alignment loss computation
            for loss_term, verdict in loss_booldict.items():
                if verdict:
                    coefficient = 1
                    if loss_term == 'GTCC':
                        specific_loss = GTCC_loss(
                            output_dict['outputs'],
                            dropouts=output_dict['dropouts'],
                            epoch=epoch,
                            **GTCC_PARAMS
                        )
                    elif loss_term == 'tcc':
                        specific_loss = TCC_loss(output_dict['outputs'], **TCC_ORIGINAL_PARAMS)
                    elif loss_term == 'LAV':
                        specific_loss = LAV_loss(output_dict['outputs'], **LAV_PARAMS)
                    elif loss_term == 'VAVA':
                        specific_loss = VAVA_loss(output_dict['outputs'], global_step=epoch, **VAVA_PARAMS)
                    else:
                        print(f"BAD LOSS TERM: {loss_term}, {verdict}")
                        exit(1)

                    loss_return_dict[loss_term + '_loss'] += specific_loss
                    loss_return_dict['total_loss'] += coefficient * specific_loss

            # NEW: Compute progress loss
            if use_progress_loss and times is not None and 'progress_head' in output_dict:
                progress_head = output_dict['progress_head']
                outputs = output_dict['outputs']

                for vid_idx, vid_emb in enumerate(outputs):
                    if vid_idx >= len(times):
                        continue
                    vid_times = times[vid_idx]

                    seg_emb, gt_prog, _ = sample_action_segment_with_random_index(
                        vid_emb, vid_times, min_segment_len=min_seg_len
                    )

                    if seg_emb is not None and seg_emb.shape[0] >= 2:
                        pred_prog = progress_head(seg_emb)
                        gt_tensor = torch.tensor(gt_prog, device=pred_prog.device, dtype=pred_prog.dtype)

                        if progress_loss_type == 'l1':
                            p_loss = torch.abs(pred_prog - gt_tensor)
                        else:  # l2
                            p_loss = (pred_prog - gt_tensor) ** 2

                        loss_return_dict['progress_loss'] += p_loss
                        loss_return_dict['total_loss'] += progress_coef * p_loss
                        progress_count += 1

        return loss_return_dict

    return _alignment_loss_fn
```

### Step 6: Add Configuration

**File:** `configs/generic_config.py`

Add after VAVA_PARAMS (~line 63):

```python
############################
## Progress Loss Configuration
CONFIG.PROGRESS_LOSS = edict({
    'enabled': False,           # Enable learnable progress head
    'coefficient': 0.1,         # Fixed weight for progress loss (if curriculum disabled)
    'loss_type': 'l1',          # 'l1' or 'l2'
    'min_segment_len': 3,       # Minimum frames in sampled segment
    'hidden_dim': 64,           # Progress head GRU hidden dimension
    'use_gru': True,            # Use GRU (True) or mean pooling (False)
    'lambda_curriculum': {
        'enabled': False,       # Enable curriculum schedule for lambda
        'start': 0.01,          # Starting weight (epoch 0)
        'end': 0.5,             # Final weight (last epoch)
    },
})
```

### Step 7: Wire Up Config to Model

**File:** Where model is instantiated (e.g., `multitask_train.py` or training script)

When creating the model, pass progress head params:

```python
model = MultiProngAttDropoutModel(
    base_model_class=base_model_class,
    base_model_params=base_model_params,
    output_dimensionality=CONFIG.OUTPUT_DIMENSIONALITY,
    num_heads=num_heads,
    dropping=CONFIG.LOSS_TYPE['GTCC'],
    attn_layers=CONFIG.ARCHITECTURE['attn_layers'],
    drop_layers=CONFIG.ARCHITECTURE['drop_layers'],
    # NEW: Progress head params
    use_progress_head=CONFIG.PROGRESS_LOSS.get('enabled', False),
    progress_head_hidden_dim=CONFIG.PROGRESS_LOSS.get('hidden_dim', 64),
    progress_head_use_gru=CONFIG.PROGRESS_LOSS.get('use_gru', True),
)
```

---

## Infrastructure Steps (From Previous Plan)

### Step 8: Add Curriculum Schedule for Lambda

**File:** `utils/loss_entry.py`

Add function to compute lambda based on epoch (supports both fixed and curriculum):

```python
def get_progress_lambda(epoch, num_epochs, progress_config):
    """
    Get progress loss weight based on config (fixed or curriculum schedule).

    Curriculum: linearly interpolates from 'start' to 'end' over training.
    This allows alignment to stabilize first, then gradually add progress signal.
    """
    if progress_config.get('lambda_curriculum', {}).get('enabled', False):
        start = progress_config['lambda_curriculum']['start']
        end = progress_config['lambda_curriculum']['end']
        # Linear interpolation
        progress = epoch / max(num_epochs - 1, 1)
        return start + (end - start) * progress
    else:
        return progress_config.get('coefficient', 0.1)
```

Update `get_loss_function` to accept `num_epochs`:

```python
def get_loss_function(config_obj, num_epochs=None):
    # ... existing setup ...

    # In _alignment_loss_fn, replace fixed coefficient:
    # progress_coef = PROGRESS_CONFIG.get('coefficient', 0.1)
    # With:
    progress_lambda = get_progress_lambda(epoch, num_epochs or 50, PROGRESS_CONFIG)
```

### Step 9: Track alignment_loss Separately

**File:** `utils/loss_entry.py`

Modify loss function to track alignment-only loss (without progress):

```python
def _alignment_loss_fn(output_dict_list, epoch, times=None):
    # ... existing setup ...

    loss_return_dict = {}
    loss_return_dict['total_loss'] = torch.tensor(0).float().to(device)
    loss_return_dict['alignment_loss'] = torch.tensor(0).float().to(device)  # NEW: Track separately

    # ... in the alignment loss loop ...
    for loss_term, verdict in loss_booldict.items():
        if verdict:
            # ... compute specific_loss ...
            loss_return_dict[loss_term + '_loss'] += specific_loss
            loss_return_dict['alignment_loss'] += specific_loss  # NEW: Add to alignment_loss
            loss_return_dict['total_loss'] += specific_loss

    # Progress loss is added to total_loss but NOT to alignment_loss
    # This allows us to track both metrics separately
```

### Step 10: Dual Checkpoint Tracking

**File:** `models/alignment_training_loop.py`

**10a. Initialize dual tracking variables** (~line 220):

```python
# Before:
best_val_loss = float('inf')

# After:
best_val_loss_combined = float('inf')    # total_loss (alignment + progress)
best_val_loss_alignment = float('inf')   # alignment_loss only (original metric)
```

**10b. Modify validation checkpoint saving** (~line 450-480):

```python
# Run validation
val_loss_dict = run_validation_epoch(...)  # Now returns full dict

# Extract loss components
val_loss_combined = val_loss_dict['total_loss'].item()
val_loss_alignment = val_loss_dict.get('alignment_loss', val_loss_dict['total_loss']).item()

# Save best COMBINED model (alignment + progress)
if val_loss_combined < best_val_loss_combined:
    best_val_loss_combined = val_loss_combined
    logger.info(f"New best COMBINED loss: {val_loss_combined:.4f}")
    ckpt_save(
        model_t=model_to_save,
        optimizer_t=optimizer,
        epoch_t=epoch,
        loss_t=val_loss_combined,
        filename=ckpt_folder + '/best_model_combined.pt',
        config=CONFIG
    )

# Save best ALIGNMENT-ONLY model (for comparison with original method)
if val_loss_alignment < best_val_loss_alignment:
    best_val_loss_alignment = val_loss_alignment
    logger.info(f"New best ALIGNMENT loss: {val_loss_alignment:.4f}")
    ckpt_save(
        model_t=model_to_save,
        optimizer_t=optimizer,
        epoch_t=epoch,
        loss_t=val_loss_alignment,
        filename=ckpt_folder + '/best_model_alignment.pt',
        config=CONFIG
    )
```

**10c. Modify `run_validation_epoch` to return full dict**:

```python
# In run_validation_epoch, change:
loss_dict = loss_fn(output_dict, epoch, times=times)
# Return the full dict instead of just the scalar
return loss_dict  # Contains 'total_loss', 'alignment_loss', 'progress_loss', etc.
```

### Step 11: Update Checkpoint Loading

**File:** `utils/ckpt_save.py`

Modify checkpoint loading to support selecting between combined and alignment-only models:

```python
def get_ckpt_MCN(ckpt_folder, num_heads, device, dropout=False, checkpoint_type='combined'):
    """
    Load checkpoint with support for dual checkpoint system.

    Args:
        checkpoint_type: 'combined' (alignment + progress) or 'alignment' (original)
    """
    # First try the requested checkpoint type
    if checkpoint_type == 'combined':
        ckpt_file = f'{ckpt_folder}/best_model_combined.pt'
    else:
        ckpt_file = f'{ckpt_folder}/best_model_alignment.pt'

    # Fallback chain for backward compatibility
    if not os.path.exists(ckpt_file):
        ckpt_file = f'{ckpt_folder}/best_model.pt'

    if not os.path.exists(ckpt_file):
        # Final fallback: find any epoch checkpoint
        ckpt_files = sorted(glob.glob(f'{ckpt_folder}/epoch-*.pt'))
        if ckpt_files:
            ckpt_file = ckpt_files[-1]
        else:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_folder}")

    # ... existing loading logic ...
```

### Step 12: Loss Plotting

**File:** `models/alignment_training_loop.py`

Add plotting for both loss curves:

```python
import matplotlib.pyplot as plt

def plot_dual_losses(train_losses, val_losses, output_folder):
    """
    Plot both alignment and combined losses for comparison.

    Args:
        train_losses: dict with keys 'total_loss', 'alignment_loss', 'progress_loss'
        val_losses: dict with same keys
        output_folder: where to save plots
    """
    epochs = range(1, len(train_losses['total_loss']) + 1)

    # Plot 1: Combined vs Alignment Loss
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training losses
    axes[0].plot(epochs, train_losses['total_loss'], label='Total (combined)', color='blue')
    axes[0].plot(epochs, train_losses['alignment_loss'], label='Alignment only', color='green', linestyle='--')
    if 'progress_loss' in train_losses:
        axes[0].plot(epochs, train_losses['progress_loss'], label='Progress only', color='red', linestyle=':')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Losses')
    axes[0].legend()
    axes[0].grid(True)

    # Validation losses
    axes[1].plot(epochs, val_losses['total_loss'], label='Total (combined)', color='blue')
    axes[1].plot(epochs, val_losses['alignment_loss'], label='Alignment only', color='green', linestyle='--')
    if 'progress_loss' in val_losses:
        axes[1].plot(epochs, val_losses['progress_loss'], label='Progress only', color='red', linestyle=':')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Validation Losses')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(f'{output_folder}/dual_loss_comparison.png', dpi=150)
    plt.close()
```

In the training loop, accumulate losses per epoch and call plotting:

```python
# Initialize loss tracking lists
train_loss_history = {'total_loss': [], 'alignment_loss': [], 'progress_loss': []}
val_loss_history = {'total_loss': [], 'alignment_loss': [], 'progress_loss': []}

# After each epoch, append losses
train_loss_history['total_loss'].append(epoch_train_loss['total_loss'])
train_loss_history['alignment_loss'].append(epoch_train_loss['alignment_loss'])
if 'progress_loss' in epoch_train_loss:
    train_loss_history['progress_loss'].append(epoch_train_loss['progress_loss'])

# Same for validation...

# At end of training or periodically:
plot_dual_losses(train_loss_history, val_loss_history, ckpt_folder)
```

---

## Configuration Example

```python
CONFIG.LOSS_TYPE = {
    'GTCC': True,
    'tcc': False,
    'LAV': False,
    'VAVA': False,
}

CONFIG.PROGRESS_LOSS = {
    'enabled': True,
    'coefficient': 0.1,          # Fixed weight (used if curriculum disabled)
    'loss_type': 'l1',
    'min_segment_len': 3,
    'hidden_dim': 64,
    'use_gru': True,
    'lambda_curriculum': {
        'enabled': True,         # Use curriculum schedule
        'start': 0.01,           # Start with low progress weight
        'end': 0.5,              # End with higher progress weight
    },
}
```

### Expected Output Structure

```
/vision/anishn/GTCC_CVPR2024/output_4fps_val/multi-task-setting/V1___GTCC_egoprocel.progress/
├── ckpt/
│   ├── best_model_combined.pt    # Best total_loss (alignment + progress)
│   ├── best_model_alignment.pt   # Best alignment_loss only (for comparison)
│   └── epoch-*.pt                # Regular epoch checkpoints
├── dual_loss_comparison.png      # Plot comparing both loss curves
├── train_loss_epochlevel.png
├── val_loss_epochlevel.png
└── config.json
```

---

## Edge Cases Handled

1. **No valid non-SIL actions**: `sample_action_segment` returns `None`, skipped in loss
2. **Very short actions** (< min_segment_len): Filtered out during sampling
3. **Frame indices out of bounds**: Clamped to valid range
4. **Empty batch**: Progress loss term just stays 0

---

## Verification Steps

1. **Unit test sampling function**:
   ```python
   embeddings = torch.randn(200, 128)
   times = {'step': ['SIL', 'cut', 'mix'], 'start_frame': [0, 20, 100], 'end_frame': [19, 99, 199], 'name': 'test'}
   seg, gt, name = sample_action_segment_with_random_index(embeddings, times)
   assert seg is not None
   assert 0 <= gt <= 1
   print(f"Sampled {name}: segment shape {seg.shape}, GT progress {gt:.3f}")
   # The segment should be variable length (from random start to random target index)
   ```

2. **Verify progress head output**:
   ```python
   head = ProgressHead(input_dim=128, hidden_dim=64, use_gru=True)
   seg = torch.randn(15, 128)
   pred = head(seg)
   assert 0 <= pred <= 1
   ```

3. **Training run**:
   ```bash
   python multitask_train.py 1 --gtcc --ego --resnet --mcn -bs 2 -ep 5
   # Check that progress_loss is logged and total_loss includes it
   ```

4. **Gradient flow**: Verify gradients flow through progress head during training

5. **Verify dual checkpoint saving**:
   ```bash
   ls output_folder/ckpt/
   # Should see:
   #   best_model_combined.pt   (best total_loss)
   #   best_model_alignment.pt  (best alignment_loss)
   # They may be saved at DIFFERENT epochs!
   ```

6. **Verify loss logging**:
   - Check that `progress_loss`, `alignment_loss`, and `total_loss` are all logged
   - Confirm curriculum schedule increases lambda over epochs (if enabled)
   - Check training logs for messages like "New best COMBINED loss" and "New best ALIGNMENT loss"

7. **Verify loss plots**:
   ```bash
   ls output_folder/
   # Should see:
   #   dual_loss_comparison.png  (alignment vs combined loss curves)
   ```
   - Open `dual_loss_comparison.png` and verify:
     - Blue line (total_loss) = Green line (alignment_loss) + Red line (progress_loss)
     - Both training and validation curves should be shown

8. **Evaluate both checkpoints**:
   ```bash
   # Evaluate combined model
   python eval.py -f output_folder --checkpoint_type combined

   # Evaluate alignment-only model
   python eval.py -f output_folder --checkpoint_type alignment
   ```
   - Compare OGPE metrics between both checkpoints to see which performs better

---

## Summary

This plan adds a **learnable progress head** following the supervisor's requirements:
- **Explicit learning**: Neural network head predicts progress directly
- **Action-level sampling**: Random segments within actions are sampled (with random target index)
- **GT supervision**: Progress labels from segment position within action

### Infrastructure Features (from progressLoss.md):
- **Dual checkpoint system**: Saves both `best_model_combined.pt` and `best_model_alignment.pt`
- **Separate loss tracking**: `alignment_loss` tracked separately from `total_loss`
- **Curriculum schedule**: Optional gradual increase of progress loss weight
- **Loss plotting**: Visualizes alignment vs combined loss curves
- **Backward compatibility**: Falls back to old checkpoint names if new ones don't exist

### Key Implementation Files

| File | Step | Description |
|------|------|-------------|
| `models/model_multiprong.py` | 1-2 | ProgressHead class + model integration |
| `utils/tensorops.py` | 3 | Segment sampling function |
| `models/alignment_training_loop.py` | 4, 10, 12 | Pass times, dual checkpoints, plotting |
| `utils/loss_entry.py` | 5, 8, 9 | Progress loss, curriculum, alignment tracking |
| `configs/generic_config.py` | 6 | Configuration section |
| `multitask_train.py` | 7 | Wire up config to model |
| `utils/ckpt_save.py` | 11 | Checkpoint loading with type selection |
