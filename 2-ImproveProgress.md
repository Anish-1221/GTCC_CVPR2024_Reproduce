# Plan: Fix Progress Head Training — V6/V7/V8

## How the Loss is Being Propagated Right Now

Let me trace through a single training step, using actual values from the config:

### Step 1: Forward Pass

The model processes a batch and produces:
- `output_dict['outputs']` — video embeddings from the encoder
- `output_dict['progress_head']` — a reference to the progress head module

### Step 2: Loss Computation (`loss_entry.py:24-256`)

Two losses are computed independently:

**Alignment loss** (e.g., GTCC): Measures temporal cycle-consistency between video pairs. This produces a scalar, say `alignment_loss = 50000` (typical for GTCC).

**Progress loss** (`loss_entry.py:141-254`): For each video in the batch:
1. Sample random actions (20 per video)
2. For each action, sample segments and target frames (5 per segment)
3. Feed segment embeddings through `progress_head(segment_embeddings)` to get predictions
4. Compute `weight * |prediction - ground_truth|` with weighted loss (early frames get up to 20x weight)
5. Add boundary loss (first frame → target `1/action_len`, last frame → target `1.0`)
6. Average everything → `progress_loss ≈ 0.3` (a small number since predictions are in [0,1])

**Combined loss** (`loss_entry.py:254`):
```
total_loss = alignment_loss + lambda * progress_loss
total_loss = 50000 + 500000 * 0.3
total_loss = 50000 + 150000 = 200000
```

### Step 3: Backward Pass (`alignment_training_loop.py:401`)

`total_loss.backward()` computes gradients for ALL parameters via the chain rule.

**For progress head parameters:**
The progress head only participates in `progress_loss`. Since `total_loss` includes `500000 * progress_loss`, the chain rule gives:
```
progress_head.param.grad = 500000 * d(progress_loss)/d(param)
```
If the true gradient is `0.01`, the stored gradient is `5000`.

**For encoder parameters:**
The encoder participates in BOTH losses. The embeddings flow into both alignment loss and progress loss (via the progress head). So:
```
encoder.param.grad = d(alignment_loss)/d(param) + 500000 * d(progress_loss)/d(param)
```

### Step 4: Gradient Rescaling (`alignment_training_loop.py:425-432`)

The code detects this is a learnable progress setup and divides progress_head gradients by lambda:
```python
for param in model.progress_head.parameters():
    param.grad = param.grad / 500000  # 5000 → 0.01
```

**After this step:**
- Progress head gradients: **small** (true scale, e.g., ~0.01 per param)
- Encoder gradients: **massive** (alignment_grad + 500000 * progress_grad, e.g., ~1000s per param)

The intention is good — give the progress head "unscaled" gradients. But here's where it breaks.

### Step 5: Global Gradient Clipping (`alignment_training_loop.py:435`)

```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.00001, norm_type=2)
```

This function:
1. Computes the **total L2 norm** across ALL parameters: `total_norm = sqrt(sum(grad^2 for all params))`
2. If `total_norm > max_norm`, multiplies ALL gradients by `clip_factor = max_norm / total_norm`

**Example with real numbers:**

The encoder has ~2M parameters with gradients averaging ~1000 each.
The progress head has ~77K parameters with gradients averaging ~0.01 each.

```
total_norm ≈ sqrt(2,000,000 * 1000^2 + 77,000 * 0.01^2)
           ≈ sqrt(2 * 10^12 + 0.0077)
           ≈ 1,414,213

clip_factor = 0.00001 / 1,414,213 = 7.07 × 10^-12
```

**After clipping:**
- Encoder grad per param: `1000 * 7.07e-12 ≈ 7.07e-9` (tiny, but that was already the original design)
- Progress head grad per param: `0.01 * 7.07e-12 ≈ 7.07e-14` (**effectively zero**)

### Step 6: Optimizer Step

`optimizer.step()` applies these clipped gradients. The progress head receives gradients that are 14 orders of magnitude below any meaningful threshold. **It literally cannot learn.**

---

## Why V7 (Transformer) Collapses but V6 (GRU) Partially Works

Even with near-zero gradients, the GRU has some advantages:
1. **Inductive bias**: GRUs inherently process sequences left-to-right. With the frame count feature and bias initialization (-2.0), the GRU starts outputting ~0.12 and its recurrent structure can pick up sequential patterns even from minuscule gradient updates over many epochs.
2. **Fewer parameters** (39K vs 77K): Less competition for the tiny gradient budget.
3. **Simpler optimization landscape**: GRU weights have more direct influence on outputs.

The Transformer has none of these advantages:
1. **No sequential inductive bias**: Attention patterns must be LEARNED, requiring meaningful gradients.
2. **More parameters**: Each gets an even smaller share of the gradient budget.
3. **Complex optimization**: Multi-head attention with ALiBi creates a harder loss landscape that needs real gradient signal to navigate.

---

## What We Want to Change

### Fix 1: Separate Gradient Clipping

**The problem**: One global `clip_grad_norm_` treats all parameters equally. Since encoder gradients dominate the norm, the progress head's already-small gradients get killed.

**The fix**: Clip encoder and progress head SEPARATELY:
- Encoder: `clip_grad_norm_(encoder_params, max_norm=0.00001)` — same as before, preserving alignment training behavior
- Progress head: `clip_grad_norm_(progress_params, max_norm=1.0)` — 100,000x more generous, letting the progress head actually learn

This means the progress head's gradient norm is computed independently. A norm of 0.5 (reasonable for a small MLP/GRU/Transformer) won't be clipped at all, and even large norms only get clipped to 1.0 — still meaningful for learning.

### Fix 2: Progress-Head-Only Training

**The problem**: Joint training creates conflicting objectives. The alignment loss wants embeddings to cluster by action type. The progress loss (via backprop through embeddings into the encoder) wants embeddings to encode temporal progress. With lambda=500000, the encoder gets overwhelmed by progress gradients, potentially harming alignment.

**The fix**: Completely separate the two training phases:
1. Train alignment first (already done — you have checkpoints)
2. Freeze the encoder, train ONLY the progress head
   - No alignment loss computed at all (saves compute)
   - No lambda needed (only progress loss exists)
   - No gradient rescaling needed
   - Separate optimizer with `lr=1e-3` (vs `1e-4` for the full model)
   - Separate gradient clipping at `max_norm=1.0`
   - The progress head learns to map frozen embeddings → progress values

This is cleaner because the progress head gets 100% of the optimizer's attention. No conflicting gradients, no domination by encoder norms, no rescaling artifacts.

---

## How the Code Changes Align

| Problem | Code Change | Why It Fixes It |
|---------|-------------|-----------------|
| Global clip kills progress head grads | Separate `clip_grad_norm_` calls for encoder vs progress head | Each gets its own norm computation and threshold |
| Lambda rescaling + clip ordering | Fix 2 removes lambda entirely (progress-only loss) | No rescaling needed when only training progress |
| Conflicting alignment/progress gradients on encoder | Fix 2 freezes encoder | Encoder embeddings don't change, progress head learns a clean mapping |
| Shared optimizer, shared LR | Fix 2 uses separate optimizer with `lr=1e-3` | Progress head gets 10x higher LR, dedicated optimizer state |

---

## Context

The learnable progress heads (V6=GRU, V7=Transformer, V8=DilatedConv) fail to learn meaningful progress predictions.

- **V7 (Transformer)**: Complete mode collapse — identical outputs `[0.3308, 0.4642, 0.4644, 0.4645]` across ALL videos/tasks
- **V6 (GRU)**: Partial learning but saturates at ~0.42 instead of reaching 1.0
- **Most recent .md file**: `ImproveProgress.md` (Feb 27 20:04)

## Root Cause: Gradient Clipping Kills Progress Head

**`alignment_training_loop.py:435`**: `clip_grad_norm_(model.parameters(), max_norm=0.00001)`

The gradient flow is broken:

1. `total_loss = alignment_loss + 500000 * progress_loss`
2. Backward: progress_head gets `500000 × true_grad`, encoder gets `align_grad + 500000 × progress_grad`
3. **Line 430-432**: progress_head grads divided by `500000` → now tiny (true scale)
4. **Line 435**: Global L2 norm clipped to `0.00001` — dominated by encoder's massive grads
5. Result: progress_head gets **effectively zero gradient**

Transformer (77K params, needs strong gradients for attention) → collapses completely.
GRU (39K params, stronger sequential inductive bias) → partially learns but saturates.

---

## Implementation: Two Fixes

### Fix 1: Separate Gradient Clipping in Joint Training

**File: `models/alignment_training_loop.py` (~line 435)**

Replace the single global clip with separate clips for encoder and progress head:

```python
# BEFORE (broken):
nn.utils.clip_grad_norm_(model.parameters(), max_norm=.00001, norm_type=2)

# AFTER (fix):
# Clip encoder parameters (same as original)
encoder_params = [p for n, p in model_to_check.named_parameters()
                  if 'progress_head' not in n and p.grad is not None]
if encoder_params:
    nn.utils.clip_grad_norm_(encoder_params, max_norm=0.00001, norm_type=2)

# Clip progress head separately with reasonable threshold
if hasattr(model_to_check, 'progress_head'):
    progress_params = [p for p in model_to_check.progress_head.parameters()
                       if p.grad is not None]
    if progress_params:
        nn.utils.clip_grad_norm_(progress_params, max_norm=1.0, norm_type=2)
```

This is a minimal change — it preserves the existing encoder clipping behavior while giving the progress head 100,000x more gradient signal.

### Fix 2: Progress-Head-Only Training Mode

Add a new training mode that loads an existing alignment checkpoint, freezes the encoder, and trains only the progress head with its own optimizer.

#### 2a. New CLI arguments

**File: `utils/parser_util.py`** — add after line 32:
```python
parser.add_argument('--train_progress_only', action='store_true',
    help='Train only the progress head (freeze encoder). Requires --alignment_checkpoint.')
parser.add_argument('--alignment_checkpoint', type=str, default=None,
    help='Path to alignment checkpoint to load before training progress head.')
parser.add_argument('--progress_lr', type=float, default=1e-3,
    help='Learning rate for progress head training (used with --train_progress_only)')
parser.add_argument('--progress_epochs', type=int, default=50,
    help='Number of epochs for progress-only training')
```

**File: `utils/parser_util.py`** — add to return dict:
```python
'train_progress_only': args.train_progress_only,
'alignment_checkpoint': args.alignment_checkpoint,
'progress_lr': args.progress_lr,
'progress_epochs': args.progress_epochs,
```

#### 2b. New progress-only loss function

**File: `utils/loss_entry.py`** — add new function `get_progress_only_loss_function()`:
- Computes ONLY the learnable progress loss (no alignment loss, no lambda multiplier)
- Reuses the existing progress loss computation logic (lines 141-254) but without wrapping it in alignment
- Returns `{'total_loss': progress_loss, 'progress_loss': progress_loss}`

#### 2c. New progress-only training loop

**File: `models/alignment_training_loop.py`** — add new function `progress_head_training_loop()`:

```python
def progress_head_training_loop(
    model, train_dl_dict, loss_fn, foldername, CONFIG,
    val_dl_dict=None, local_rank=0, train_samplers=None,
    progress_lr=1e-3, num_epochs=50
):
    """Train only the progress head with encoder frozen."""

    model_to_check = model.module if hasattr(model, 'module') else model

    # Freeze everything except progress_head
    for name, param in model_to_check.named_parameters():
        if 'progress_head' not in name:
            param.requires_grad = False

    # Separate optimizer for progress head only
    optimizer = optim.Adam(model_to_check.progress_head.parameters(), lr=progress_lr)

    # Standard training loop but:
    # - Only progress loss (no alignment)
    # - Reasonable gradient clipping (max_norm=1.0)
    # - No gradient rescaling (no lambda)
    # - Checkpoint saves based on progress_loss only
    ...
```

The body reuses the same batch iteration pattern from `alignment_training_loop()` but simplified:
- No alignment loss computation (saves GPU memory and time)
- `nn.utils.clip_grad_norm_(model_to_check.progress_head.parameters(), max_norm=1.0)`
- No gradient rescaling step
- Checkpoints: `best_model_progress.pt`

#### 2d. Wire up in multitask_train.py

**File: `multitask_train.py`** — after model creation (~line 224), add:

```python
# Check if progress-only training mode
train_progress_only = CFG.get('TRAIN_PROGRESS_ONLY', False)

if train_progress_only:
    # Load alignment checkpoint
    alignment_ckpt_path = CFG.get('ALIGNMENT_CHECKPOINT')
    checkpoint = torch.load(alignment_ckpt_path, map_location=device)
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=False)
    logger.info(f"Loaded alignment checkpoint from {alignment_ckpt_path}")

    from models.alignment_training_loop import progress_head_training_loop
    progress_head_training_loop(
        model, train_dataloaders,
        get_progress_only_loss_function(CFG),
        master_experiment_foldername, CONFIG=CFG,
        val_dl_dict=val_dataloaders,
        local_rank=local_rank,
        train_samplers=train_samplers,
        progress_lr=CFG.get('PROGRESS_LR', 1e-3),
        num_epochs=CFG.get('PROGRESS_EPOCHS', 50),
    )
else:
    alignment_training_loop(...)  # existing code
```

**File: `configs/entry_config.py`** — add config propagation:
```python
CONFIG.TRAIN_PROGRESS_ONLY = args_given.get('train_progress_only', False)
CONFIG.ALIGNMENT_CHECKPOINT = args_given.get('alignment_checkpoint', None)
CONFIG.PROGRESS_LR = args_given.get('progress_lr', 1e-3)
CONFIG.PROGRESS_EPOCHS = args_given.get('progress_epochs', 50)
```

---

## Files to Modify

| File | Change |
|------|--------|
| `models/alignment_training_loop.py` | (1) Fix gradient clipping to use separate clips for encoder vs progress head. (2) Add `progress_head_training_loop()` function |
| `utils/loss_entry.py` | Add `get_progress_only_loss_function()` |
| `utils/parser_util.py` | Add `--train_progress_only`, `--alignment_checkpoint`, `--progress_lr`, `--progress_epochs` args |
| `configs/entry_config.py` | Propagate new args to CONFIG |
| `multitask_train.py` | Add progress-only training branch before `alignment_training_loop()` call |

---

## Usage

### Joint training (with Fix 1 applied):
```bash
# Same command as before — gradient clipping fix is automatic
CUDA_VISIBLE_DEVICES=0 python multitask_train.py 1 --gtcc --egoprocel --resnet --mcn \
  --progress_loss learnable --progress_arch transformer --progress_lambda 500000.0
```

### Progress-head-only training (Fix 2):
```bash
# Step 1: Use existing alignment checkpoint (already trained)
# Step 2: Train progress head only
CUDA_VISIBLE_DEVICES=0 python multitask_train.py 1 --gtcc --egoprocel --resnet --mcn \
  --progress_loss learnable --progress_arch transformer --progress_lambda 500000.0 \
  --train_progress_only \
  --alignment_checkpoint output_learnable_progress_v7/multi-task-setting_val/V1___GTCC_egoprocel/ckpt/best_model.pt \
  --progress_lr 0.001 --progress_epochs 50
```

---

## Verification

1. **Quick sanity**: Train progress-only for 5 epochs on V6 GRU, check logs show decreasing progress_loss
2. **V6 full**: Run 50 epochs, `python extract_progress.py`, verify progress reaches 0.8-1.0 for full actions
3. **V7 full**: Same for Transformer — should no longer collapse to fixed values
4. **V8 full**: Same for DilatedConv
5. **Alignment preserved**: Run `python eval_protas_action_level.py` — metrics should be identical to base alignment checkpoint (encoder frozen)

## Expected Outcomes

- **V7 Transformer**: Varied progress predictions (not collapsed) — values should vary by video content
- **V6 GRU**: Progress reaches 0.8-1.0 for full actions instead of saturating at 0.42
- **V8 DilatedConv**: Comparable or better than V6
- **No alignment degradation**: Encoder is frozen in Fix 2, so alignment unchanged
