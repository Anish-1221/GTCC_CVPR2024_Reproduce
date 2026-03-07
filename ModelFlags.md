# GTCC_CVPR2024 — CLI Flags Reference

## Baseline Commands (Alignment Training)

```bash
# GTCC alignment
python multitask_train.py 1 --gtcc --ego --resnet --mcn

# VAVA alignment
python multitask_train.py 1 --vava --ego --resnet --mcn

# LAV alignment
python multitask_train.py 1 --lav --ego --resnet --mcn

# TCC alignment
python multitask_train.py 1 --tcc --ego --resnet --mcn
```

Every command requires exactly: `version` + one loss type + one dataset + one architecture.
`--mcn` is optional but standard for multi-task (MCN = Multi-task Compliant Network).

---

## 1. Positional Argument

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `version` | string | Yes | Version identifier. Used to name the experiment folder: `V{version}___{loss}_{dataset}` |

- **Config key**: `CONFIG.VERSION`
- **Code path**: `entry_config.py:119` → `multitask_train.py:80` (experiment folder naming)
- **Example**: `python multitask_train.py v1 ...` → folder `V1___GTCC_egoprocel`

---

## 2. Loss Type (mutually exclusive, required)

Exactly one must be specified. Determines the alignment loss function.

| Flag | Aliases | Constant | Description |
|------|---------|----------|-------------|
| `--GTCC` | `--gtcc` | `'GTCC'` | Generalized Temporal Cycle Consistency loss |
| `--TCC` | `--tcc` | `'tcc'` | Temporal Cycle Consistency loss |
| `--LAV` | `--lav` | `'LAV'` | Learning by Aligning Videos loss |
| `--VAVA` | `--vava` | `'VAVA'` | Video Alignment via Variational Alignment loss |

- **Config key**: `CONFIG.LOSS_TYPE[loss_type] = True`
- **Code path**: `parser_util.py:50-54` → `entry_config.py:67` → `loss_entry.py:481-498` (loss dispatch)

### GTCC-specific behavior
When `--gtcc` is used:
- Sets `CONFIG.GTCC_PARAMS['delta']`, `n_components`, `gamma` based on dataset defaults (`entry_config.py:70-73`)
- If `--resnet` and not `--mcn`: enables dropout in base encoder (`entry_config.py:63-64`)
- If not `--mcn`: enables dropping network in temporal stacking arch (`entry_config.py:76-78`)

### Dataset-specific GTCC defaults

| Dataset | delta | n_components (K) | gamma |
|---------|-------|-------------------|-------|
| egoprocel, cmu, egtea | 0.2 | 15 | 0.95 |
| penn-action, pouring, coin | 0.5 | 5 | 1.0 |

### Memory thresholds (max pairwise product before skipping pair)

| Loss | Threshold |
|------|-----------|
| TCC | 8,000,000 |
| GTCC | 8,000,000 |
| VAVA | 6,000,000 |
| LAV | 3,000,000 (strictest — SoftDTW backward pass) |

---

## 3. Base Architecture (mutually exclusive, required)

Exactly one must be specified. Determines the base encoder architecture.

| Flag | Aliases | Constant | Description |
|------|---------|----------|-------------|
| `--resnet50` | `--rnet50` | `'resnet50'` | ResNet-50 base encoder |
| `--temporal_stacking` | `--tstack` | `'temporal_stacking'` | Temporal stacking encoder |
| `--naive` | `--naive` | `'naive'` | Naive MLP encoder |

- **Config key**: `CONFIG.BASEARCH.ARCHITECTURE`
- **Code path**: `parser_util.py:57-60` → `entry_config.py:59` → `multitask_train.py:95` (data subfolder), `multitask_train.py:212` (model creation)
- **Note**: Architecture determines which data subfolder to load features from (e.g., `resnet50` features vs `temporal_stacking` features)

---

## 4. Dataset (mutually exclusive, required)

Exactly one must be specified. Determines dataset and data path.

| Flag | Aliases | Constant | Data folder |
|------|---------|----------|-------------|
| `--egoprocel` | `--ego` | `'egoprocel'` | `$DATASET_PATH/egoprocel` |
| `--cmu` | `--cmu` | `'cmu'` | `$DATASET_PATH/egoprocel` (shared folder) |
| `--egtea` | `--egtea` | `'egtea'` | `$DATASET_PATH/egoprocel` (shared folder) |
| `--penn_action` | `--penn` | `'penn-action'` | `$DATASET_PATH/penn-action` |
| `--pouring` | `--pour` | `'pouring'` | `$DATASET_PATH/pouring` |
| `--coin` | `--coin` | `'coin'` | `$DATASET_PATH/coin` |

- **Config key**: `CONFIG.DATASET_NAME`, `CONFIG.DATAFOLDER`
- **Code path**: `parser_util.py:63-69` → `entry_config.py:48-52` → `multitask_train.py:93` (data loading)
- **Note**: `cmu` and `egtea` share the `egoprocel` data folder (`entry_config.py:49-50`)
- **Validation**: Only `egoprocel`, `cmu`, `egtea` pass validation (`entry_config.py:145`)

---

## 5. Training Hyperparameters (optional)

| Flag | Aliases | Type | Default | Config Key | Description |
|------|---------|------|---------|------------|-------------|
| `-lr` | `--learning_rate` | float | `0.0001` | `CONFIG.LEARNING_RATE` | Adam optimizer learning rate for alignment training |
| `-bs` | `--batch_size` | int | `2` | `CONFIG.BATCH_SIZE` | DataLoader batch size (number of video pairs per batch) |
| `-ep` | `--epochs` | int | `50` | `CONFIG.NUM_EPOCHS` | Number of training epochs for alignment |
| `-od` | `--output_dimensions` | int | `128` | `CONFIG.OUTPUT_DIMENSIONALITY` | Embedding dimensionality of the encoder output |
| `-m` | `--message` | string | `None` | (used in experiment name) | Appended to experiment folder name: `V1___GTCC_egoprocel.message` |

### Code paths
- **`-lr`**: `entry_config.py:86` → `alignment_training_loop.py:239` (`optim.Adam(..., lr=learning_rate)`)
- **`-bs`**: `entry_config.py:87` → `multitask_train.py:109,138,158` (DataLoader creation)
- **`-ep`**: `entry_config.py:88` → `alignment_training_loop.py:233,258` (training loop range)
- **`-od`**: `entry_config.py:89` → `multitask_train.py:212` (model output dim); also used for GTCC dropping network (`entry_config.py:78`)
- **`-m`**: `entry_config.py:120` → experiment name suffix

---

## 6. Model Flags (optional)

### `--mcn`
- **Type**: boolean flag (`store_true`)
- **Default**: `False`
- **Config key**: `CONFIG.ARCHITECTURE['MCN']`
- **Description**: Enables Multi-task Compliant Network architecture
- **Impact**:
  - If `True`: Creates `MultiProngAttDropoutModel` with attention heads and multi-task architecture (`multitask_train.py:196-222`)
  - If `False`: Creates base model directly; enables GTCC dropout/dropping network (`entry_config.py:63-64, 76-78`)
  - Controls number of attention heads based on number of tasks (`multitask_train.py:214`)

### `--debug`
- **Type**: boolean flag (`store_true`)
- **Default**: `False`
- **Config key**: `CONFIG.DEBUG`
- **Description**: Enables debug mode for additional logging and conditional execution

---

## 7. Progress Loss Flags (optional)

These flags enable and configure the learnable progress prediction head.

### `--progress_loss`
- **Type**: string
- **Choices**: `cumulative_l2`, `learnable`
- **Default**: `None` (progress loss disabled)
- **Config key**: `CONFIG.PROGRESS_LOSS['enabled']`, `CONFIG.PROGRESS_LOSS['method']`
- **Description**: Enables progress loss. `cumulative_l2` computes progress directly from embeddings; `learnable` adds a dedicated progress prediction head.
- **Impact**:
  - When set: `CONFIG.PROGRESS_LOSS['enabled'] = True` (`entry_config.py:94`)
  - `learnable`: Creates a progress head in the model (`multitask_train.py:204-206`), enables separate gradient clipping (`alignment_training_loop.py:440-451`)
  - `cumulative_l2`: Computes cumulative L2 distance over predicted/GT progress (`loss_entry.py:513-566`)

### `--progress_lambda`
- **Type**: float
- **Default**: `0.1`
- **Config key**: `CONFIG.PROGRESS_LOSS['lambda_fixed']`
- **Description**: Weight coefficient for progress loss in total loss: `total_loss += lambda * progress_loss`
- **Impact**:
  - Applied in `loss_entry.py:566`: `loss_return_dict['total_loss'] += progress_lambda * avg_progress_loss`
  - Used for gradient rescaling in `alignment_training_loop.py:427-432`: progress head gradients divided by lambda to normalize scale
  - **Typical values**: `0.1` for joint training, `500000.0` for progress-head-only training (where alignment loss is 0)

### `--progress_arch`
- **Type**: string
- **Choices**: `gru`, `transformer`, `dilated_conv`
- **Default**: `gru`
- **Config key**: `CONFIG.PROGRESS_LOSS['learnable']['architecture']`
- **Description**: Architecture of the learnable progress head (only used with `--progress_loss learnable`)
- **Architectures**:
  - **`gru`**: Unidirectional GRU with FC output layer. Hidden dim=64. Causal by design. (`model_multiprong.py` ProgressHead class)
  - **`transformer`**: Causal transformer with ALiBi positional bias. d_model=64, 4 heads, 2 layers. (`model_multiprong.py` TransformerProgressHead class)
  - **`dilated_conv`**: Stack of causal dilated convolution blocks (dilations 1,2,4,8,16,32). Hidden dim=64, kernel=3. (`model_multiprong.py` DilatedConvProgressHead class)
- **Note**: All three architectures are causal — output at position t depends only on frames 0..t

### V9 Architecture Flags (Anti-Saturation)

These flags fix the early saturation problem where the GRU plateaus at ~0.73-0.80 within the first few frames when processing 2048-d raw features.

### `--use_input_projection`
- **Type**: boolean flag (`store_true`)
- **Default**: `False`
- **Config key**: `CONFIG.PROGRESS_LOSS['learnable']['use_input_projection']`
- **Description**: Add `Linear(input_dim→projection_dim) + ReLU` before the GRU. Reduces the compression ratio from 32:1 (2048→64) to 1:1 (128→128), preventing the GRU from being overwhelmed by high-dimensional input.
- **Impact**: GRU sees 128-d projected features instead of raw 2048-d. Critical when using `--progress_features raw`.

### `--projection_dim`
- **Type**: int
- **Default**: `128`
- **Config key**: `CONFIG.PROGRESS_LOSS['learnable']['projection_dim']`
- **Description**: Target dimension for the input projection layer. Only used with `--use_input_projection`.

### `--progress_hidden_dim`
- **Type**: int
- **Default**: `64`
- **Config key**: `CONFIG.PROGRESS_LOSS['learnable']['hidden_dim']`
- **Description**: GRU hidden state dimension. Use `128` with input projection for balanced capacity (128-d input → 128-d hidden).

### `--output_activation`
- **Type**: string
- **Choices**: `sigmoid`, `clamp`
- **Default**: `sigmoid`
- **Config key**: `CONFIG.PROGRESS_LOSS['learnable']['output_activation']`
- **Description**: Output activation function for the progress head.
  - **`sigmoid`** (default): Classic sigmoid squash. Has gradient compression near 0 and 1, making it disproportionately hard to predict values >0.8.
  - **`clamp`**: Clamped linear `torch.clamp(output, 0, 1)`. Equal gradient flow at all output levels, allowing the model to reach values close to 1.0 more easily.

### `--per_frame_count`
- **Type**: boolean flag (`store_true`)
- **Default**: `False`
- **Config key**: `CONFIG.PROGRESS_LOSS['learnable']['per_frame_count']`
- **Description**: Use per-frame running count `log(1+i)/log(1+300)` for each frame i, instead of broadcast frame count `log(1+T)/log(1+300)` (same for all frames). Gives each frame a unique temporal signal, breaking GRU update gate saturation within a forward pass.

### V10 Architecture Flags (Action Conditioning + Rate-of-Change)

### `--use_action_conditioning`
- **Type**: boolean flag (`store_true`)
- **Default**: `False`
- **Config key**: `CONFIG.PROGRESS_LOSS['learnable']['use_action_conditioning']`
- **Description**: Add an action-class embedding to the progress head input. Tells the model *which* action is happening so it can learn action-specific progress curves. Without this, the model learns one average curve for all actions, causing early jumps to ~0.7 and ceilings below 1.0.
- **How it works**:
  - Adds `nn.Embedding(116, 16)` to ProgressHead (116 action classes: 0=unknown + 1-115 real actions)
  - The 16-d embedding is looked up for the current action class and **broadcast to all frames** in the segment
  - Concatenated to each frame's features before the GRU
  - The embedding vectors are **learned during training** — the model discovers action-specific duration patterns
  - Action labels come from `times_dict['step']` (already available, no dataset changes needed)
  - Backward compatible: `action_idx=None` falls back to index 0
- **Impact**: GRU input dimension increases by `action_embed_dim` (default 16). E.g., 129 → 145.

### `--action_embed_dim`
- **Type**: int
- **Default**: `16`
- **Config key**: `CONFIG.PROGRESS_LOSS['learnable']['action_embed_dim']`
- **Description**: Dimension of the action class embedding vector. Only used with `--use_action_conditioning`.

### `--use_rate_of_change`
- **Type**: boolean flag (`store_true`)
- **Default**: `False`
- **Config key**: `CONFIG.PROGRESS_LOSS['learnable']['use_rate_of_change']`
- **Description**: Add frame-to-frame L2 distance as an extra scalar input feature. Provides the GRU with an explicit "visual velocity" signal — how much the features are changing between consecutive frames.
- **How it works**:
  - Computes `||features[t] - features[t-1]||₂` for each frame (on projected 128-d features)
  - First frame gets diff=0
  - Concatenated as a (T, 1) tensor to the feature stack before GRU
- **Why it helps**: Actions have characteristic velocity profiles. Feature changes are rapid at transitions (start/end of actions) and slow in the middle. This gives the GRU an explicit signal about where in the action the model is, complementing the temporal position from `per_frame_count`.
- **Impact**: GRU input dimension increases by 1.

### `--progress_loss_mode`
- **Type**: string
- **Choices**: `uniform_mono`, `sqrt_weighted`, `mse`, `legacy`, `dense`
- **Default**: `uniform_mono`
- **Config key**: `CONFIG.PROGRESS_LOSS['learnable']['progress_loss_mode']`
- **Description**: Loss formulation for the learnable progress head
- **Modes**:
  - **`uniform_mono`** (default): Uniform L1 loss + monotonicity penalty (weight=2.0, margin=0.01) + endpoint regularization (weight=1.0). Samples random frames within actions. (`loss_entry.py` `_progress_loss_uniform_mono()`)
  - **`sqrt_weighted`**: Sqrt-weighted L1 loss on progress values. (`loss_entry.py` `_progress_loss_sqrt_weighted()`)
  - **`mse`**: MSE loss with endpoint regularization. (`loss_entry.py` `_progress_loss_mse()`)
  - **`legacy`**: Original formulation — weighted L1 + boundary loss (weight_cap=20.0, boundary_weight=5.0). (`loss_entry.py` `_progress_loss_legacy()`)
  - **`dense`**: Dense per-frame MSE on full action segments. Feeds entire action to progress head with `dense_output=True`, computes MSE against linear GT `[1/L, 2/L, ..., 1.0]` at every frame. **Automatically disables `use_frame_count`**. (`loss_entry.py` `_progress_loss_dense()`)

### `--progress_features`
- **Type**: string
- **Choices**: `aligned`, `raw`
- **Default**: `aligned`
- **Config key**: `CONFIG.PROGRESS_LOSS['learnable']['features']`
- **Description**: Feature source for the progress head
- **Options**:
  - **`aligned`** (default): Uses 128-d post-alignment-encoder embeddings (same as `output_dict['outputs']`)
  - **`raw`**: Uses 2048-d pre-computed features loaded from disk (specified via `--raw_features_path`). These features come from a pre-trained ResNet backbone and have zero alignment training influence, retaining full video-specific information.
- **Impact**:
  - Changes progress head `input_dim` from 128 to 2048
  - Training loop loads features from `--raw_features_path` and injects into `output_dict['raw_features']`
  - Loss functions use `output_dict['raw_features']` instead of `output_dict['outputs']`
- **Requires**: `--raw_features_path` must be set when using `raw`

### `--raw_features_path`
- **Type**: string (path)
- **Default**: None
- **Config key**: `CONFIG.RAW_FEATURES_PATH`
- **Description**: Path to folder containing 2048-d raw feature `.npy` files. Each file should be named `{video_handle}.npy` with shape `(T, 2048)`.
- **Example**: `/vision/anishn/GTCC_Data_Processed_1fps_2048/egoprocel/features`
- **Impact**: Training loop and extract_progress.py load features from this path instead of using encoder intermediates
- **Note**: Also available in `extract_progress.py` via `--raw_features_path` flag (auto-detected from stored config if not provided)

---

## 8. Progress-Head-Only Training Flags (optional)

These flags enable a special training mode where the alignment encoder is frozen and only the progress head is trained.

### `--train_progress_only`
- **Type**: boolean flag (`store_true`)
- **Default**: `False`
- **Config key**: `CONFIG.TRAIN_PROGRESS_ONLY`
- **Description**: Freeze the alignment encoder and only train the progress head
- **Requirements**: Must specify `--alignment_checkpoint` (raises ValueError otherwise)
- **Impact**:
  - Loads encoder weights from checkpoint (`multitask_train.py:248-273`)
  - Calls `progress_head_training_loop()` instead of `alignment_training_loop()` (`multitask_train.py:274-285`)
  - All encoder/attention/head parameters frozen (`alignment_training_loop.py:845-846`)
  - Separate optimizer for progress head only (`alignment_training_loop.py:860`)
  - No alignment loss computed (saves GPU memory)

### `--alignment_checkpoint`
- **Type**: string (file path)
- **Default**: `None`
- **Config key**: `CONFIG.ALIGNMENT_CHECKPOINT`
- **Description**: Path to pre-trained alignment checkpoint to load
- **Required when**: `--train_progress_only` is set
- **Impact**: Loaded with `torch.load()`, state dict applied with `strict=False` (`multitask_train.py:248-252`)

### `--reinit_progress_head`
- **Type**: boolean flag (`store_true`)
- **Default**: `False`
- **Config key**: `CONFIG.REINIT_PROGRESS_HEAD`
- **Description**: Ignore progress head weights from checkpoint, initialize fresh (bias=-2.0)
- **Only relevant with**: `--train_progress_only --alignment_checkpoint <path>`
- **Impact**: Excludes `progress_head.*` keys from checkpoint state dict (`multitask_train.py:260-265`)

### `--progress_lr`
- **Type**: float
- **Default**: `0.001` (1e-3)
- **Config key**: `CONFIG.PROGRESS_LR`
- **Description**: Learning rate for progress head optimizer (used in progress-head-only training)
- **Impact**: `optim.Adam(progress_head.parameters(), lr=progress_lr)` (`alignment_training_loop.py:860`)

### `--progress_epochs`
- **Type**: int
- **Default**: `50`
- **Config key**: `CONFIG.PROGRESS_EPOCHS`
- **Description**: Number of epochs for progress-head-only training (separate from `-ep`)
- **Impact**: Controls training loop range in `progress_head_training_loop()` (`alignment_training_loop.py:821`)

---

## 9. extract_progress.py Flags

Separate CLI for extracting progress predictions from a trained model.

```bash
python extract_progress.py -f <experiment_folder> [--ckpt <filename>] [--max_videos <N>]
```

| Flag | Aliases | Type | Default | Description |
|------|---------|------|---------|-------------|
| `-f` | `--folder` | string | Required | Path to experiment folder (e.g., `output_learnable_progress_v7/multi-task-setting_val/V1___GTCC_egoprocel`) |
| `--ckpt` | — | string | `best_model.pt` | Checkpoint filename to load (e.g., `best_model_progress.pt` for progress-only checkpoints) |
| `--max_videos` | — | int | `None` (all) | Limit number of videos to process (useful for quick testing) |

- **Note**: `--progress_arch` is NOT needed — architecture is read from the stored config in the experiment folder
- **Inference mode**: Always uses `dense_output=False` (growing prefix, frame-by-frame)

---

## 10. Flag Interaction Notes

### Dependencies
| If you use... | You must also use... | Reason |
|---------------|---------------------|--------|
| `--train_progress_only` | `--alignment_checkpoint <path>` | Needs pre-trained encoder weights |
| `--train_progress_only` | `--progress_loss learnable` | Progress-only mode requires a learnable head |
| `--reinit_progress_head` | `--train_progress_only` | Only meaningful when loading from checkpoint |
| `--progress_arch` | `--progress_loss learnable` | Only used with learnable progress head |
| `--progress_loss_mode` | `--progress_loss learnable` | Only affects learnable progress head |
| `--progress_features raw` | `--raw_features_path <path>` | Needs path to 2048-d feature files on disk |
| `--progress_features` | `--progress_loss learnable` | Only used with learnable progress head |
| `--use_input_projection` | `--progress_features raw` (practically) | Needed to reduce 2048→128 before GRU |
| `--use_action_conditioning` | `--progress_loss learnable` | Only used with learnable progress head |
| `--use_rate_of_change` | `--progress_loss learnable` | Only used with learnable progress head |

### Conditional behavior
- `--gtcc` + `--resnet` + no `--mcn` → enables dropout in resnet50 base encoder
- `--gtcc` + no `--mcn` → enables dropping network in temporal stacking arch
- `--progress_loss_mode dense` → automatically sets `use_frame_count=False` (removes length shortcut)
- `--progress_features raw` → progress head input_dim changes from 128 to 2048; loads features from `--raw_features_path`
- `--mcn` → creates `MultiProngAttDropoutModel` with attention; without → creates base model directly
- `--use_action_conditioning` → GRU input dim increases by `action_embed_dim` (default 16)
- `--use_rate_of_change` → GRU input dim increases by 1

### Gradient handling
- **Joint training** (alignment + progress): encoder gradients clipped at `max_norm=0.00001`, progress head gradients clipped separately at `max_norm=1.0` (`alignment_training_loop.py:440-454`)
- **Progress-only training**: only progress head gradients, clipped at `max_norm=1.0` (`alignment_training_loop.py:981`)
- Progress head gradients are rescaled by dividing by `--progress_lambda` before clipping (`alignment_training_loop.py:427-432`)

---

## 11. Example Commands

### Alignment training (baseline)
```bash
# Standard GTCC alignment on egoprocel
python multitask_train.py 1 --gtcc --ego --resnet --mcn

# With custom hyperparameters
python multitask_train.py 1 --gtcc --ego --resnet --mcn -lr 0.001 -bs 4 -ep 100 -od 256

# With message tag for experiment tracking
python multitask_train.py 1 --gtcc --ego --resnet --mcn -m "baseline_run"
```

### Joint training (alignment + progress)
```bash
# GRU progress head with dense supervision
python multitask_train.py 1 --gtcc --ego --resnet --mcn \
  --progress_loss learnable --progress_arch gru \
  --progress_lambda 0.1 --progress_loss_mode dense

# Transformer progress head with uniform_mono loss
python multitask_train.py 1 --gtcc --ego --resnet --mcn \
  --progress_loss learnable --progress_arch transformer \
  --progress_lambda 0.1 --progress_loss_mode uniform_mono
```

### Progress-head-only training (frozen encoder)
```bash
# Train transformer progress head from scratch on pre-trained encoder
python multitask_train.py 1 --gtcc --ego --resnet --mcn \
  --progress_loss learnable --progress_arch transformer \
  --progress_lambda 500000.0 \
  --train_progress_only --reinit_progress_head \
  --alignment_checkpoint <path>/ckpt/best_model.pt \
  --progress_lr 0.001 --progress_epochs 50 \
  --progress_loss_mode dense

# Continue training existing progress head (don't reinit)
python multitask_train.py 1 --gtcc --ego --resnet --mcn \
  --progress_loss learnable --progress_arch gru \
  --progress_lambda 500000.0 \
  --train_progress_only \
  --alignment_checkpoint <path>/ckpt/best_model.pt \
  --progress_lr 0.0001 --progress_epochs 30
```

### V9: Anti-saturation architecture (raw 2048-d features)
```bash
CUDA_VISIBLE_DEVICES=2 python multitask_train.py 1 --gtcc --egoprocel --resnet --mcn \
  --progress_loss learnable --progress_lambda 500000.0 \
  --train_progress_only --reinit_progress_head \
  --progress_lr 0.001 --progress_epochs 50 \
  --progress_loss_mode dense --progress_features raw \
  --raw_features_path /vision/anishn/GTCC_Data_Processed_1fps_2048/egoprocel/features \
  --use_input_projection --projection_dim 128 --progress_hidden_dim 128 \
  --output_activation clamp --per_frame_count
```

### V10: Action conditioning + rate-of-change (builds on V9)
```bash
CUDA_VISIBLE_DEVICES=2 python multitask_train.py 1 --gtcc --egoprocel --resnet --mcn \
  --progress_loss learnable --progress_lambda 500000.0 \
  --train_progress_only --reinit_progress_head \
  --progress_lr 0.0003 --progress_epochs 50 \
  --progress_loss_mode dense --progress_features raw \
  --raw_features_path /vision/anishn/GTCC_Data_Processed_1fps_2048/egoprocel/features \
  --use_input_projection --projection_dim 128 --progress_hidden_dim 128 \
  --output_activation clamp --per_frame_count \
  --use_action_conditioning --action_embed_dim 16 \
  --use_rate_of_change
```

### Extracting progress predictions
```bash
# Extract from best alignment checkpoint
python extract_progress.py -f output_folder/multi-task-setting_val/V1___GTCC_egoprocel

# Extract from progress-specific checkpoint
python extract_progress.py -f output_folder/multi-task-setting_val/V1___GTCC_egoprocel \
  --ckpt best_model_progress.pt

# Quick test on 5 videos
python extract_progress.py -f output_folder/multi-task-setting_val/V1___GTCC_egoprocel \
  --ckpt best_model_progress.pt --max_videos 5
```
