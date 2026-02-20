# Progress-Weighted Alignment Loss Implementation Plan

## Overview

Add a weighted progress loss term to GTCC, TCC, LAV, and VAVA losses. The progress loss measures how well the learned embeddings predict temporal progress within videos.

**Key Design Decisions:**
- Single shared encoder (always pick one random video S or T for progress loss)
- Use cumulative L2 distance (`get_cum_matrix`) for predicted progress
- **Video-level**: Normalize predicted progress to [0,1] per entire video
- **Action-level**: Normalize predicted progress to [0,1] per action segment (resets at each action boundary)
- Configurable GT progress (video-level or action-level)
- Single random frame error per batch
- Dual checkpoint system (combined loss + alignment-only loss)
- Support both fixed lambda and curriculum schedule

---

## Files to Modify

| File | Changes |
|------|---------|
| `configs/generic_config.py` | Add progress loss configuration options |
| `utils/loss_entry.py` | Add progress loss computation to loss function |
| `utils/tensorops.py` | Add helpers for normalized predicted progress (video & action level) |
| `models/alignment_training_loop.py` | Pass `times` to loss fn, dual checkpoint tracking |
| `utils/ckpt_save.py` | Support dual checkpoint naming |

---

## Implementation Steps

### Step 1: Add Configuration Options

**File:** `configs/generic_config.py`

Add new config section after existing loss parameters (~line 50):

```python
# Progress Loss Configuration
CONFIG.PROGRESS_LOSS = {
    'enabled': False,              # Enable/disable progress loss
    'level': 'video',              # 'video' or 'action' (which GT progress to use)
    'lambda_fixed': 0.1,           # Fixed weight coefficient
    'lambda_curriculum': {
        'enabled': False,          # Enable curriculum schedule
        'start': 0.0,              # Starting weight (epoch 0)
        'end': 1.0,                # Final weight (last epoch)
    },
}
```

---

### Step 2: Add Normalized Progress Helpers

**File:** `utils/tensorops.py`

Add functions after `get_cum_matrix` (~line 332):

```python
def get_normalized_predicted_progress_video(features):
    """
    Compute predicted progress normalized to [0, 1] for VIDEO-LEVEL.

    The cumulative distance is computed across the entire video and
    normalized by the final (max) cumulative distance.

    Args:
        features: Tensor of shape (T, D) - frame embeddings

    Returns:
        Tensor of shape (T,) - progress values in [0, 1]
    """
    cum_dist = get_cum_matrix(features)
    max_val = cum_dist[-1]  # Final cumulative distance
    if max_val > 0:
        return cum_dist / max_val
    else:
        # Edge case: all frames identical, return linear progress
        return torch.linspace(0, 1, features.shape[0], device=features.device)


def get_normalized_predicted_progress_action(features, time_dict):
    """
    Compute predicted progress normalized to [0, 1] for ACTION-LEVEL.

    Each action segment gets its own cumulative distance that resets
    at action boundaries. Progress is normalized per-segment.
    Background/SIL segments get progress = 0.

    Args:
        features: Tensor of shape (T, D) - frame embeddings
        time_dict: dict with 'step', 'start_frame', 'end_frame' keys

    Returns:
        Tensor of shape (T,) - progress values in [0, 1] per action
    """
    T = features.shape[0]
    pred_progress = torch.zeros(T, device=features.device)

    for step, start, end in zip(time_dict['step'],
                                 time_dict['start_frame'],
                                 time_dict['end_frame']):
        # Clamp indices to valid range
        start = max(0, min(start, T - 1))
        end = max(0, min(end, T - 1))

        if step in ['SIL', 'background']:
            # Background segments get 0 progress
            pred_progress[start:end+1] = 0
        else:
            # Extract segment features
            segment_features = features[start:end+1]

            if segment_features.shape[0] < 2:
                # Single frame segment - set to midpoint
                pred_progress[start:end+1] = 0.5
            else:
                # Compute cumulative distance within segment
                segment_cum = get_cum_matrix(segment_features)

                # Normalize by segment's max (so it goes 0->1 within segment)
                max_val = segment_cum[-1]
                if max_val > 0:
                    pred_progress[start:end+1] = segment_cum / max_val
                else:
                    # All frames identical - linear interpolation
                    seg_len = end - start + 1
                    pred_progress[start:end+1] = torch.linspace(0, 1, seg_len, device=features.device)

    return pred_progress
```

---

### Step 3: Add Progress Loss Function

**File:** `utils/loss_entry.py`

Add new imports and functions:

```python
from utils.tensorops import (
    get_trueprogress,
    get_trueprogress_per_action,
    get_normalized_predicted_progress_video,
    get_normalized_predicted_progress_action
)
import random
import copy

def compute_progress_loss(outputs, times, level='video'):
    """
    Compute progress loss for a randomly selected video at a random frame.

    Args:
        outputs: List of tensors [(T_i, D), ...] - video embeddings
        times: List of time_dicts with action annotations
        level: 'video' (0->1 across video) or 'action' (0->1 per action)

    Returns:
        progress_loss: Scalar tensor
    """
    if len(outputs) == 0 or len(times) == 0:
        return torch.tensor(0.0, device=device)

    # Randomly select one video (S or T - doesn't matter, same encoder)
    video_idx = random.randint(0, len(outputs) - 1)
    video_features = outputs[video_idx]
    time_dict = times[video_idx]

    T = video_features.shape[0]
    if T < 2:
        return torch.tensor(0.0, device=device)

    # Randomly select one frame
    frame_idx = random.randint(0, T - 1)

    # Deep copy time_dict to avoid modifying original
    time_dict_copy = copy.deepcopy(time_dict)
    # Ensure end_frame matches actual video length
    time_dict_copy['end_frame'][-1] = T - 1

    if level == 'video':
        # Video-level: normalize cumulative distance by max across entire video
        pred_progress = get_normalized_predicted_progress_video(video_features)
        gt_progress = get_trueprogress(time_dict_copy)
    else:
        # Action-level: normalize cumulative distance per action segment
        pred_progress = get_normalized_predicted_progress_action(video_features, time_dict_copy)
        gt_progress = get_trueprogress_per_action(time_dict_copy)

    gt_progress = gt_progress.to(video_features.device)

    # Compute L1 error at the randomly selected frame
    progress_loss = torch.abs(pred_progress[frame_idx] - gt_progress[frame_idx])

    return progress_loss


def get_progress_lambda(epoch, num_epochs, progress_config):
    """
    Get progress loss weight based on config (fixed or curriculum).
    """
    if progress_config['lambda_curriculum']['enabled']:
        start = progress_config['lambda_curriculum']['start']
        end = progress_config['lambda_curriculum']['end']
        # Linear interpolation
        progress = epoch / max(num_epochs - 1, 1)
        return start + (end - start) * progress
    else:
        return progress_config['lambda_fixed']
```

Modify `get_loss_function` signature and implementation:

```python
def get_loss_function(config_obj, num_epochs=None):
    loss_booldict = config_obj.LOSS_TYPE
    TCC_ORIGINAL_PARAMS = config_obj.TCC_ORIGINAL_PARAMS
    GTCC_PARAMS = config_obj.GTCC_PARAMS
    LAV_PARAMS = config_obj.LAV_PARAMS
    VAVA_PARAMS = config_obj.VAVA_PARAMS
    PROGRESS_CONFIG = getattr(config_obj, 'PROGRESS_LOSS', {'enabled': False})

    def _alignment_loss_fn(output_dict_list, epoch, times_list=None):
        if type(output_dict_list) != list:
            output_dict_list = [output_dict_list]

        # Wrap times_list if needed
        if times_list is not None and not isinstance(times_list[0], list):
            times_list = [times_list]

        loss_return_dict = {}
        loss_return_dict['total_loss'] = torch.tensor(0).float().to(device)
        loss_return_dict['alignment_loss'] = torch.tensor(0).float().to(device)  # Track alignment-only

        if PROGRESS_CONFIG.get('enabled', False):
            loss_return_dict['progress_loss'] = torch.tensor(0).float().to(device)

        for loss_term, verdict in loss_booldict.items():
            if verdict:
                loss_return_dict[loss_term + '_loss'] = torch.tensor(0).float().to(device)

        for idx, output_dict in enumerate(output_dict_list):
            if len(output_dict['outputs']) < 2:
                continue

            # Compute alignment losses
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
                        specific_loss = TCC_loss(
                            output_dict['outputs'], **TCC_ORIGINAL_PARAMS
                        )
                    elif loss_term == 'LAV':
                        specific_loss = LAV_loss(
                            output_dict['outputs'], **LAV_PARAMS
                        )
                    elif loss_term == 'VAVA':
                        specific_loss = VAVA_loss(
                            output_dict['outputs'], global_step=epoch, **VAVA_PARAMS
                        )
                    else:
                        print(f"BAD LOSS TERM: {loss_term}, {verdict}")
                        exit(1)

                    loss_return_dict[loss_term + '_loss'] += specific_loss
                    loss_return_dict['alignment_loss'] += coefficient * specific_loss
                    loss_return_dict['total_loss'] += coefficient * specific_loss

            # Add progress loss (per batch, same level as alignment losses)
            if PROGRESS_CONFIG.get('enabled', False) and times_list is not None:
                times = times_list[idx] if len(times_list) > idx else times_list[0]
                if times is not None and len(output_dict['outputs']) > 0:
                    progress_loss = compute_progress_loss(
                        output_dict['outputs'],
                        times,
                        level=PROGRESS_CONFIG.get('level', 'video')
                    )
                    progress_lambda = get_progress_lambda(epoch, num_epochs or 50, PROGRESS_CONFIG)
                    loss_return_dict['progress_loss'] += progress_loss
                    loss_return_dict['total_loss'] += progress_lambda * progress_loss

        return loss_return_dict

    return _alignment_loss_fn
```

---

### Step 4: Update Training Loop

**File:** `models/alignment_training_loop.py`

**4a. Update get_loss_function call** (~line 200-210):

```python
# Before:
loss_fn = get_loss_function(CONFIG)

# After:
loss_fn = get_loss_function(CONFIG, num_epochs=num_epochs)
```

**4b. Modify loss function call to pass times** (~line 342):

```python
# Before:
loss_dict = loss_fn(output_dict, epoch)

# After:
loss_dict = loss_fn(output_dict, epoch, times_list=times)
```

**4c. Add dual checkpoint tracking** (modify initialization ~line 220):

```python
# Before:
best_val_loss = float('inf')

# After:
best_val_loss_combined = float('inf')    # New combined loss (alignment + progress)
best_val_loss_alignment = float('inf')   # Original alignment-only loss
```

**4d. Modify validation checkpoint saving** (~line 450-480):

```python
# Run validation
val_loss_dict = run_validation_epoch(...)  # Modify to return full dict

# Extract loss components
val_loss_combined = val_loss_dict['total_loss'].item()
val_loss_alignment = val_loss_dict.get('alignment_loss', val_loss_dict['total_loss']).item()

# Save best combined model
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

# Save best alignment-only model
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

**4e. Modify `run_validation_epoch` to return full loss dict** (~line 485-530):

Change from returning scalar to returning full dict with alignment_loss tracked.
Also pass `times` to the loss function in validation:

```python
# In run_validation_epoch, change:
loss_dict = loss_fn(output_dict, epoch, times_list=times)
# And return the full dict instead of just the scalar
```

---

### Step 5: Update Checkpoint Loading

**File:** `utils/ckpt_save.py`

Modify `get_ckpt_MCN` and `get_ckpt_basic` to support checkpoint selection:

```python
def get_ckpt_MCN(ckpt_folder, checkpoint_type='combined'):
    """
    Args:
        checkpoint_type: 'combined' (new loss) or 'alignment' (original loss)
    """
    if checkpoint_type == 'combined':
        ckpt_file = f'{ckpt_folder}/best_model_combined.pt'
    else:
        ckpt_file = f'{ckpt_folder}/best_model_alignment.pt'

    # Fallback to old naming for backward compatibility
    if not os.path.exists(ckpt_file):
        ckpt_file = f'{ckpt_folder}/best_model.pt'

    if not os.path.exists(ckpt_file):
        # ... existing fallback logic ...
```

---

## Key Difference: Video-Level vs Action-Level Progress

### Video-Level (`level='video'`)

```
Video: [Action1 frames] [SIL frames] [Action2 frames]
GT:    [0.0 → 0.5]      [0.5 → 0.5]  [0.5 → 1.0]

Pred:  Cumulative L2 distance across ENTIRE video
       Normalized by max(cumulative_distance) of video
       Result: [0 → 1] monotonically increasing
```

### Action-Level (`level='action'`)

```
Video: [Action1 frames] [SIL frames] [Action2 frames]
GT:    [0.0 → 1.0]      [0 → 0]      [0.0 → 1.0]

Pred:  Cumulative L2 distance RESETS at each action boundary
       For Action1: cum_dist[0:T1] / max(cum_dist[0:T1])
       For SIL: 0
       For Action2: cum_dist[0:T2] / max(cum_dist[0:T2])
       Result: Each action goes [0 → 1] independently
```

---

## Configuration Example

**Training config with progress loss enabled:**

```python
CONFIG.LOSS_TYPE = {
    'GTCC': True,
    'tcc': False,
    'LAV': False,
    'VAVA': False,
}

CONFIG.PROGRESS_LOSS = {
    'enabled': True,
    'level': 'video',           # or 'action'
    'lambda_fixed': 0.1,
    'lambda_curriculum': {
        'enabled': True,        # Use curriculum instead of fixed
        'start': 0.01,
        'end': 0.5,
    },
}
```

---

## Verification Steps

1. **Unit test progress loss computation:**
   ```python
   # Test video-level
   features = torch.randn(100, 128)
   time_dict = {'step': ['action1', 'SIL', 'action2'],
                'start_frame': [0, 40, 60],
                'end_frame': [39, 59, 99]}
   loss = compute_progress_loss([features], [time_dict], level='video')
   assert 0 <= loss <= 1

   # Test action-level
   loss_action = compute_progress_loss([features], [time_dict], level='action')
   assert 0 <= loss_action <= 1
   ```

2. **Verify action-level normalization:**
   ```python
   pred = get_normalized_predicted_progress_action(features, time_dict)
   # Check that each action segment independently goes toward 1
   # Check that SIL segments are 0
   ```

3. **Verify dual checkpoint saving:**
   - Train for a few epochs
   - Confirm both `best_model_combined.pt` and `best_model_alignment.pt` exist
   - Verify they may be saved at different epochs

4. **Verify loss logging:**
   - Check that `progress_loss`, `alignment_loss`, and `total_loss` are all logged
   - Confirm curriculum schedule increases lambda over epochs

5. **Run full training:**
   ```bash
   python multitask_train.py --config <config_with_progress>
   ```

6. **Evaluate both checkpoints:**
   ```bash
   python eval.py --checkpoint best_model_combined.pt
   python eval.py --checkpoint best_model_alignment.pt
   ```

---

## Summary of Changes

| Component | Change Type | Description |
|-----------|-------------|-------------|
| Config | Addition | `PROGRESS_LOSS` config dict |
| tensorops.py | Addition | `get_normalized_predicted_progress_video()` and `get_normalized_predicted_progress_action()` |
| loss_entry.py | Addition + Modification | `compute_progress_loss()`, `get_progress_lambda()`, updated `get_loss_function()` |
| alignment_training_loop.py | Modification | Pass `times` to loss fn, dual checkpoint tracking |
| ckpt_save.py | Modification | Support `checkpoint_type` parameter |

---

## Key Implementation Notes

1. **Random selection is per-batch:** Each batch randomly selects one video and one frame for progress loss.

2. **Gradient flow:** Progress loss gradients flow back through the encoder since cumulative L2 distance uses embeddings.

3. **Action-level resets cumulative distance:** Unlike video-level, action-level computes `get_cum_matrix` separately for each action segment, making each segment's progress independent.

4. **Backward compatibility:** Old checkpoints (`best_model.pt`) still work via fallback.

5. **Edge cases handled:**
   - Videos with < 2 frames return 0 loss
   - Single-frame action segments get progress = 0.5
   - All-identical frames return linear progress
   - SIL/background segments get progress = 0 for action-level
