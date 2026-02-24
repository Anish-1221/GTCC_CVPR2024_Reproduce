# Progress Loss Implementation Plan for GTCC

## Summary

Add two progress loss methods to GTCC, both at **action-level only**:
1. **Cumulative L2** (`cumulative_l2`): Non-learnable, uses embedding distance
2. **Learnable Head** (`learnable`): Neural network (GRU) predicts progress

Both are optional, configurable via CLI argument.

---

## Training Data Organization

**Source:** `egoprocel.json`

```json
{
  "BaconAndEggs.egtea": {
    "handles": ["OP01-R03-BaconAndEggs", ...],
    "hdl_actions": [["0", "30", "0", "6", ...], ...],
    "hdl_start_times": [[0, 1075, 1148, ...], ...],
    "hdl_end_times": [[1075, 1148, 1182, ...], ...]
  }
}
```

**Structure:** Each video is a sequence of segments:
```
Segment 0: frames 0-1075    → action '0' (background)
Segment 1: frames 1075-1148 → action '30' (break_eggs)
Segment 2: frames 1148-1182 → action '0' (background)
...
```

**Important:** Action labels in JSON are **IDs** (strings like `'0'`, `'30'`), NOT names.
- `'0'` = background (from `egoprocel-id2action.txt`)
- `'30'` = break_eggs
- etc.

**Times dict passed to loss function:**
```python
times_dict = {
    'step': ['0', '30', '0', '6', ...],      # Action IDs (NOT names!)
    'start_frame': [0, 44, 47, 49, ...],     # Scaled to 1 FPS
    'end_frame': [44, 47, 49, 53, ...],      # Scaled to 1 FPS
    'name': 'OP01-R03-BaconAndEggs'
}
```

---

## Step 0: Create Git Branch

```bash
cd /vision/anishn/GTCC_CVPR2024
git checkout -b progress_loss
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `utils/parser_util.py` | Add `--progress_loss` argument |
| `configs/generic_config.py` | Add `CONFIG.PROGRESS_LOSS` section |
| `configs/entry_config.py` | Parse progress loss args into config |
| `utils/tensorops.py` | Add 2 helper functions + modify `get_trueprogress_per_action` |
| `utils/loss_entry.py` | Add progress loss computation, modify signature |
| `models/model_multiprong.py` | Add `ProgressHead` class, modify model |
| `models/alignment_training_loop.py` | Pass `times` to loss fn, dual checkpoints |
| `multitask_train.py` | Wire progress config to model |

---

## Step 1: Add CLI Arguments

**File:** `utils/parser_util.py`

Add after line 17 (after `--message`):

```python
# Progress loss configuration
parser.add_argument('--progress_loss',
    type=str,
    default=None,
    choices=['cumulative_l2', 'learnable'],
    help='Progress loss method (action-level). Options: cumulative_l2, learnable')
parser.add_argument('--progress_lambda', type=float, default=0.1,
    help='Progress loss coefficient')
```

Update return dict (line 42-54) to include:
```python
'progress_loss': args.progress_loss,
'progress_lambda': args.progress_lambda,
```

---

## Step 2: Add Config Section

**File:** `configs/generic_config.py`

Add after line 63 (after `CONFIG.VAVA_PARAMS`):

```python
######################
## Progress Loss Configuration
CONFIG.PROGRESS_LOSS = edict({
    'enabled': False,
    'method': 'cumulative_l2',      # 'cumulative_l2' or 'learnable'
    'lambda_fixed': 0.1,
    'learnable': {
        'hidden_dim': 64,
        'use_gru': True,
        'min_segment_len': 3,
    },
})
```

---

## Step 3: Parse Args into Config

**File:** `configs/entry_config.py`

Add after the output dimensions config (around line 88):

```python
# Progress loss configuration
progress_loss_arg = args_given.get('progress_loss', None)
if progress_loss_arg is not None:
    CONFIG.PROGRESS_LOSS['enabled'] = True
    CONFIG.PROGRESS_LOSS['method'] = progress_loss_arg
    CONFIG.PROGRESS_LOSS['lambda_fixed'] = float(args_given.get('progress_lambda', 0.1))
```

---

## Step 4: Add/Modify Helper Functions

**File:** `utils/tensorops.py`

### 4a. Modify `get_trueprogress_per_action` to handle action IDs

**IMPORTANT:** JSON uses action IDs ('0' for background), not names ('background', 'SIL').

```python
def get_trueprogress_per_action(time_dict):
    """Each action goes 0->1 independently"""
    # '0' is the action ID for 'background' in egoprocel.json
    BACKGROUND_LABELS = ['0', 'SIL', 'background']
    N = time_dict['end_frame'][-1] + 1
    progress = torch.zeros(N)

    for step, start, end in zip(time_dict['step'], time_dict['start_frame'], time_dict['end_frame']):
        if step not in BACKGROUND_LABELS:
            segment_length = end - start + 1
            progress[start:end+1] = torch.arange(1, segment_length + 1, dtype=torch.float) / segment_length
        else:
            progress[start:end+1] = 0

    return progress
```

### 4b. Add `get_normalized_predicted_progress_action`

```python
def get_normalized_predicted_progress_action(features, time_dict):
    """
    Compute predicted progress normalized to [0, 1] for ACTION-LEVEL.
    Cumulative L2 distance resets at each action boundary.
    SIL/background segments get progress = 0.
    """
    # '0' is the action ID for 'background' in egoprocel.json
    BACKGROUND_LABELS = ['0', 'SIL', 'background']
    T = features.shape[0]
    pred_progress = torch.zeros(T, device=features.device)

    for step, start, end in zip(time_dict['step'],
                                 time_dict['start_frame'],
                                 time_dict['end_frame']):
        start = max(0, min(start, T - 1))
        end = max(0, min(end, T - 1))

        if step in BACKGROUND_LABELS:
            pred_progress[start:end+1] = 0
        else:
            segment_features = features[start:end+1]
            if segment_features.shape[0] < 2:
                pred_progress[start:end+1] = 0.5
            else:
                segment_cum = get_cum_matrix(segment_features)
                max_val = segment_cum[-1]
                if max_val > 0:
                    pred_progress[start:end+1] = segment_cum / max_val
                else:
                    seg_len = end - start + 1
                    pred_progress[start:end+1] = torch.linspace(0, 1, seg_len, device=features.device)

    return pred_progress
```

### 4c. Add `sample_action_segment_with_random_index`

```python
def sample_action_segment_with_random_index(embeddings, times_dict, min_segment_len=3):
    """
    Sample a random segment within a random non-SIL action.
    Returns segment embeddings up to random target index and GT progress.
    """
    # '0' is the action ID for 'background' in egoprocel.json
    BACKGROUND_LABELS = ['0', 'SIL', 'background']
    valid_actions = []
    for idx, (step, start, end) in enumerate(zip(
        times_dict['step'], times_dict['start_frame'], times_dict['end_frame']
    )):
        action_length = end - start + 1
        if step not in BACKGROUND_LABELS and action_length >= min_segment_len:
            valid_actions.append((idx, step, start, end))

    if len(valid_actions) == 0:
        return None, None, None

    _, action_name, action_start, action_end = random.choice(valid_actions)
    action_length = action_end - action_start + 1

    max_start = action_end - min_segment_len + 1
    if max_start < action_start:
        max_start = action_start

    seg_start = random.randint(action_start, max_start)
    seg_end = random.randint(seg_start + min_segment_len - 1, action_end)
    target_idx = random.randint(seg_start, seg_end)

    T = embeddings.shape[0]
    seg_start = max(0, min(seg_start, T - 1))
    target_idx = max(seg_start, min(target_idx, T - 1))

    segment_embeddings = embeddings[seg_start:target_idx + 1]
    progress_at_target = (target_idx - action_start + 1) / action_length
    gt_progress = min(1.0, max(0.0, progress_at_target))

    return segment_embeddings, gt_progress, action_name
```

---

## Step 5: Add ProgressHead Class

**File:** `models/model_multiprong.py`

Add before `class MultiProngAttDropoutModel`:

```python
class ProgressHead(nn.Module):
    """
    Learnable progress prediction head.
    Takes segment embeddings and predicts progress at the final frame.
    """
    def __init__(self, input_dim=128, hidden_dim=64, use_gru=True):
        super(ProgressHead, self).__init__()
        self.use_gru = use_gru

        if use_gru:
            self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1,
                              batch_first=True, bidirectional=False)
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
            _, h_n = self.gru(x)                  # h_n: (1, 1, hidden_dim)
            progress = self.fc(h_n.squeeze())
        else:
            x = segment_embeddings.mean(dim=0)    # Mean pool: (D,)
            progress = self.fc(x)
        return progress.squeeze()
```

---

## Step 6: Modify MultiProngAttDropoutModel

**File:** `models/model_multiprong.py`

Modify `__init__` signature to add:
```python
def __init__(
    self,
    base_model_class,
    base_model_params,
    output_dimensionality,
    num_heads,
    dropping=False,
    attn_layers = [512, 256],
    drop_layers = [512, 128, 256],
    use_progress_head=False,           # NEW
    progress_head_config=None,         # NEW
):
```

Add after dropout initialization:
```python
        # Progress head (for learnable progress loss)
        self.use_progress_head = use_progress_head
        if use_progress_head and progress_head_config is not None:
            self.progress_head = ProgressHead(
                input_dim=output_dimensionality,
                hidden_dim=progress_head_config.get('hidden_dim', 64),
                use_gru=progress_head_config.get('use_gru', True)
            )
```

Modify `forward` return:
```python
        result = {'outputs': outputs, 'attentions': attentions}
        if self.dropping:
            result['dropouts'] = dropouts
        if self.use_progress_head:
            result['progress_head'] = self.progress_head
        return result
```

---

## Step 7: Modify Loss Entry

**File:** `utils/loss_entry.py`

Add imports at top:
```python
from utils.tensorops import (
    get_trueprogress_per_action,
    get_normalized_predicted_progress_action,
    sample_action_segment_with_random_index
)
import random
import copy
```

Modify `get_loss_function` signature:
```python
def get_loss_function(config_obj, num_epochs=None):
```

Add after VAVA_PARAMS:
```python
    PROGRESS_CONFIG = getattr(config_obj, 'PROGRESS_LOSS', {'enabled': False})
```

Modify `_alignment_loss_fn` signature:
```python
    def _alignment_loss_fn(output_dict_list, epoch, times=None):
```

Add after initializing loss_return_dict:
```python
        # Track alignment loss separately for dual checkpoints
        loss_return_dict['alignment_loss'] = torch.tensor(0).float().to(device)
        if PROGRESS_CONFIG.get('enabled', False):
            loss_return_dict['progress_loss'] = torch.tensor(0).float().to(device)
```

Add inside the loop, after alignment loss:
```python
                    loss_return_dict['alignment_loss'] += coefficient * specific_loss
```

Add after the alignment loss loop, before `return`:
```python
            # Progress loss computation
            if PROGRESS_CONFIG.get('enabled', False) and times is not None:
                progress_lambda = PROGRESS_CONFIG.get('lambda_fixed', 0.1)
                method = PROGRESS_CONFIG.get('method', 'cumulative_l2')

                if method == 'cumulative_l2':
                    # Cumulative L2 method - action level
                    if len(output_dict['outputs']) > 0:
                        video_idx = random.randint(0, len(output_dict['outputs']) - 1)
                        video_features = output_dict['outputs'][video_idx]
                        time_dict = copy.deepcopy(times[video_idx])

                        T = video_features.shape[0]
                        if T >= 2:
                            frame_idx = random.randint(0, T - 1)
                            time_dict['end_frame'][-1] = T - 1

                            pred_progress = get_normalized_predicted_progress_action(video_features, time_dict)
                            gt_progress = get_trueprogress_per_action(time_dict).to(video_features.device)

                            p_loss = torch.abs(pred_progress[frame_idx] - gt_progress[frame_idx])
                            loss_return_dict['progress_loss'] += p_loss
                            loss_return_dict['total_loss'] += progress_lambda * p_loss

                elif method == 'learnable' and 'progress_head' in output_dict:
                    # Learnable head method - action level (one sample per batch)
                    progress_head = output_dict['progress_head']
                    learnable_config = PROGRESS_CONFIG.get('learnable', {})
                    min_seg_len = learnable_config.get('min_segment_len', 3)

                    if len(output_dict['outputs']) > 0:
                        # Sample ONE random video from the batch
                        vid_idx = random.randint(0, len(output_dict['outputs']) - 1)
                        if vid_idx < len(times):
                            vid_emb = output_dict['outputs'][vid_idx]

                            seg_emb, gt_prog, _ = sample_action_segment_with_random_index(
                                vid_emb, times[vid_idx], min_segment_len=min_seg_len
                            )

                            if seg_emb is not None and seg_emb.shape[0] >= 2:
                                pred_prog = progress_head(seg_emb)
                                gt_tensor = torch.tensor(gt_prog, device=pred_prog.device, dtype=pred_prog.dtype)
                                p_loss = torch.abs(pred_prog - gt_tensor)
                                loss_return_dict['progress_loss'] += p_loss
                                loss_return_dict['total_loss'] += progress_lambda * p_loss
```

---

## Step 8: Modify Training Loop

**File:** `models/alignment_training_loop.py`

**8a. Modify loss function call**:
```python
loss_dict = loss_fn(output_dict, epoch, times=times)
```

**8b. Add dual checkpoint tracking**:
```python
best_val_loss_combined = float('inf')
best_val_loss_alignment = float('inf')
```

**8c. Modify validation checkpoint saving**:
```python
# Get alignment loss for dual checkpointing
val_loss_combined = val_loss
val_loss_alignment = val_loss  # Default if no alignment_loss key

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

# Save best alignment model (same as combined when no progress loss)
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

**8d. Modify `run_validation_epoch`**:
```python
loss_dict = loss_fn(output_dict, epoch, times=times)
```

---

## Step 9: Wire Config to Model

**File:** `multitask_train.py`

Modify model creation:
```python
# Determine if progress head is needed
use_progress_head = (
    CFG.PROGRESS_LOSS.get('enabled', False) and
    CFG.PROGRESS_LOSS.get('method') == 'learnable'
)

model = MultiProngAttDropoutModel(
    base_model_class=base_model_class,
    base_model_params=base_model_params,
    output_dimensionality=CFG.OUTPUT_DIMENSIONALITY,
    num_heads=num_tks,
    dropping=CFG.LOSS_TYPE['GTCC'],
    attn_layers=CFG.ARCHITECTURE['attn_layers'],
    drop_layers=CFG.ARCHITECTURE['drop_layers'],
    use_progress_head=use_progress_head,
    progress_head_config=CFG.PROGRESS_LOSS.get('learnable', {}) if use_progress_head else None,
)
```

Update loss function call:
```python
get_loss_function(CFG, num_epochs=CFG.NUM_EPOCHS)
```

---

## Checkpoint Files Explained

| Checkpoint | What it tracks | When to use |
|------------|----------------|-------------|
| `best_model.pt` | Best validation loss (legacy) | Backwards compatibility |
| `best_model_combined.pt` | Best total loss (alignment + λ*progress) | When you want model optimized for both objectives |
| `best_model_alignment.pt` | Best alignment-only loss | **For evaluation** - alignment metrics are what we care about |

**Note:** Currently, loss plots show **combined loss** only.

---

## Progress Loss Flow

### Cumulative L2 Method

```
1. Pick ONE random video from batch
2. For each action segment (skip background '0'):
   - Extract segment features
   - Compute cumulative L2 distance from segment start
   - Normalize by max distance → predicted progress [0, 1]
3. Ground truth: linear progress within each action
4. Pick ONE random frame, compute |pred - gt|
5. Add λ * loss to total_loss
```

### Learnable Head Method

```
1. Pick ONE random video from batch
2. Pick ONE random non-background action
3. Pick random segment [seg_start, seg_end] within action
4. Pick random target frame within segment
5. Extract embeddings from seg_start to target_idx
6. Pass through GRU → MLP → Sigmoid → predicted progress
7. GT = (target_idx - action_start + 1) / action_length
8. Loss = |pred - gt|
9. Add λ * loss to total_loss
```

---

## Example Usage

```bash
# Set output path
export OUTPUT_PATH=/vision/anishn/GTCC_CVPR2024/output_learnable_progress

# GTCC with learnable progress loss
CUDA_VISIBLE_DEVICES=6 python multitask_train.py 1 --gtcc --ego --resnet --mcn --progress_loss learnable --progress_lambda 0.1

# TCC with learnable progress loss
CUDA_VISIBLE_DEVICES=7 python multitask_train.py 1 --tcc --ego --resnet --mcn --progress_loss learnable --progress_lambda 0.1

# GTCC with cumulative L2 progress loss
CUDA_VISIBLE_DEVICES=0 python multitask_train.py 1 --gtcc --ego --resnet --mcn --progress_loss cumulative_l2 --progress_lambda 0.1

# No progress loss (default, unchanged behavior)
python multitask_train.py 1 --gtcc --ego --resnet --mcn
```

---

## Key Bug Fix: Background Label Detection

**Problem:** JSON stores action IDs (`'0'` for background), but original code checked for names (`'SIL'`, `'background'`).

**Solution:** All background checks now use:
```python
BACKGROUND_LABELS = ['0', 'SIL', 'background']
```

This affects:
- `get_trueprogress_per_action()`
- `get_normalized_predicted_progress_action()`
- `sample_action_segment_with_random_index()`

---

## Key Files Summary

| File | Purpose |
|------|---------|
| `utils/parser_util.py` | CLI argument `--progress_loss` |
| `configs/generic_config.py` | `CONFIG.PROGRESS_LOSS` defaults |
| `configs/entry_config.py` | Parse CLI args to config |
| `utils/tensorops.py` | `get_normalized_predicted_progress_action`, `sample_action_segment_with_random_index`, modified `get_trueprogress_per_action` |
| `utils/loss_entry.py` | Progress loss computation, modified `_alignment_loss_fn(output_dict_list, epoch, times=None)` |
| `models/model_multiprong.py` | `ProgressHead` class, model integration |
| `models/alignment_training_loop.py` | Pass times, dual checkpoints |
| `multitask_train.py` | Wire config to model |
