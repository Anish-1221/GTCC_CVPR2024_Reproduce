# GTCC Architecture Understanding Document

## Executive Summary

**GTCC (Gaussian Temporal Cycle Consistency)** is a self-supervised video alignment method that learns to align videos of the same action without frame-level labels. The key innovation over TCC (Temporal Cycle Consistency) is the use of **Gaussian Mixture Models (GMMs)** to capture multi-modal temporal correspondences, enabling better alignment when multiple valid temporal mappings exist.

The system:
1. Takes pre-extracted video features (e.g., from ResNet/I3D)
2. Passes them through an encoder network to produce frame-level embeddings
3. Computes alignment loss between video pairs using GMM-based soft correspondences
4. Learns embeddings where temporally corresponding frames across videos are close in embedding space

---

## 1. Model Architecture

### 1.1 Overall Architecture Options

The codebase supports two main architecture modes:

| Mode | Description | When Used |
|------|-------------|-----------|
| **MCN (Multi-Component Network)** | `MultiProngAttDropoutModel` with attention-based multi-head design | `CONFIG.ARCHITECTURE['MCN'] = True` |
| **Simple Encoder** | Direct encoder (StackingEncoder, Resnet50Encoder, or NaiveEncoder) | `CONFIG.ARCHITECTURE['MCN'] = False` |

### 1.2 MultiProngAttDropoutModel

**Location:** `models/model_multiprong.py:143-202`

The main model for GTCC experiments when using multi-head attention.

```
Input: List[Tensor] of shape [(T_i, D_in) for i in batch]
       where T_i = sequence length, D_in = input feature dim (e.g., 2048)

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                    MultiProngAttDropoutModel                     │
├─────────────────────────────────────────────────────────────────┤
│  base_model (StackingEncoder)                                    │
│       ↓                                                          │
│  [general_features] (T, D_out) per video                        │
│       ↓                                                          │
│  head_models[0..num_heads-1] (HeadModel)                        │
│       ↓                                                          │
│  prong_outputs: List of (T, D_out) per head per video           │
│       ↓                                                          │
│  attention_layer: Concat heads → softmax weights                │
│       ↓                                                          │
│  weighted_combination: Sum over heads                            │
│       ↓                                                          │
│  [outputs]: Final embeddings (T, D_out) per video               │
│                                                                  │
│  (if dropping=True):                                             │
│  dropout: Predict drop weights for GTCC alignment               │
└─────────────────────────────────────────────────────────────────┘

Output: {
    'outputs': List[(T_i, D_out)],      # Final embeddings
    'attentions': List[(T_i, num_heads)], # Attention weights
    'dropouts': List[(D_out+1,)]        # Drop vectors (if enabled)
}
```

**Key Parameters:**
- `base_model_class`: The encoder class (typically `StackingEncoder`)
- `output_dimensionality`: Embedding dimension (default: 128)
- `num_heads`: Number of parallel "prong" heads
- `dropping`: Boolean to enable dropout network for GTCC
- `attn_layers`: Layer sizes for attention network [512, 256]
- `drop_layers`: Layer sizes for dropout network [512, 128, 256]

### 1.3 HeadModel

**Location:** `models/model_multiprong.py:204-220`

Each head processes the base features independently:

```python
HeadModel:
    fc_layers: Linear(D_out, 512) → ReLU → Linear(512, 128) → ReLU →
               Linear(128, 256) → ReLU → Linear(256, D_out)
```

### 1.4 Base Encoders

#### StackingEncoder (Primary)
**Location:** `models/model_singleprong.py:84-177`

The most commonly used encoder. Uses 1D convolutions for temporal modeling.

```
Input: (T, D_in) per video, e.g., (T, 2048)

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│  sequence.permute(1,0)  →  (D_in, T)                            │
│       ↓                                                          │
│  Conv1d(D_in → conv_num_channels, kernel=temporal_depth)        │
│  ReLU                                                            │
│  Conv1d → Conv1d → ... (descending channels)                    │
│  Until: channels ≤ output_dimensions AND num_layers ≥ 7         │
│       ↓                                                          │
│  output.permute(1,0)  →  (T, D_out)                             │
└─────────────────────────────────────────────────────────────────┘

Output: (T, output_dimensions) per video, e.g., (T, 128)
```

**Key Parameters:**
- `temporal_depth`: Kernel size for first conv layer (controls temporal receptive field)
- `conv_num_channels`: Starting channel count (typically 256)
- `output_dimensions`: Final embedding size (typically 128)
- `input_dimensions`: Input feature size (typically 2048)

#### Resnet50Encoder
**Location:** `models/model_singleprong.py:11-82`

Uses 3D convolutions for spatial-temporal processing (for raw spatial features).

```
Input: (T, C, H, W) per video, e.g., (T, 1024, 14, 14)

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│  For each frame t in [k, T]:                                    │
│    stack = frames[t-k:t]  →  (C, k, H, W)                       │
│    Conv3D(C→C, k=3) → ReLU → Conv3D(C→512, k=3) → ReLU          │
│    MaxPool3D → squeeze → (512,)                                 │
│    Linear(512→512) → ReLU → Linear(512→512) → ReLU              │
│    Linear(512→128) → ReLU                                       │
└─────────────────────────────────────────────────────────────────┘

Output: (T-k, 128) per video
```

#### NaiveEncoder
**Location:** `models/model_singleprong.py:180-227`

Simple MLP encoder for testing purposes.

---

## 2. Data Flow

### 2.1 Complete Forward Pass

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING ITERATION                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. DataLoader yields batch:                                     │
│     inputs: List[str] (file paths) or List[Tensor]              │
│     times: List[dict] with action annotations                   │
│                                                                  │
│  2. preprocess_batch():                                          │
│     - Load .npy files if lazy loading                           │
│     - Apply skip_rate subsampling                               │
│     - Scale time annotations accordingly                        │
│     → inputs: List[Tensor(T_i, 2048)]                           │
│                                                                  │
│  3. model.forward(inputs):                                       │
│     ┌───────────────────────────────────────────────────────┐   │
│     │ base_model(videos)                                     │   │
│     │   For each video in batch:                             │   │
│     │     sequence: (T, 2048) → permute → (2048, T)          │   │
│     │     Conv1D stack → (D_out, T) → permute → (T, D_out)   │   │
│     │   → general_features: List[(T_i, D_out)]               │   │
│     │                                                         │   │
│     │ head_models (if MCN):                                   │   │
│     │   For each head in num_heads:                           │   │
│     │     For each video:                                     │   │
│     │       fc_layers(general_features) → (T, D_out)          │   │
│     │   → prong_outputs: List[List[(T, D_out)]]              │   │
│     │                                                         │   │
│     │ attention_layer:                                        │   │
│     │   concatenate heads → (T, D_out * num_heads)            │   │
│     │   Linear → Tanh → Linear → Softmax                      │   │
│     │   → attention_weights: (T, num_heads)                   │   │
│     │                                                         │   │
│     │ weighted_combination:                                   │   │
│     │   Σ(prong_output * attention_weight) over heads         │   │
│     │   → outputs: List[(T_i, D_out)]                         │   │
│     │                                                         │   │
│     │ dropout (if dropping=True):                             │   │
│     │   For each output: Linear layers → mean over T          │   │
│     │   → dropouts: List[(D_out+1,)]                          │   │
│     └───────────────────────────────────────────────────────┘   │
│                                                                  │
│  4. loss_fn(output_dict, epoch):                                │
│     Computes GTCC_loss / TCC_loss / LAV_loss / VAVA_loss        │
│     → loss_dict: {'total_loss': Tensor, ...}                    │
│                                                                  │
│  5. Backward + Optimizer step                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Tensor Dimensions Example

Assuming batch_size=4, D_in=2048, D_out=128, num_heads=3:

| Stage | Shape | Description |
|-------|-------|-------------|
| Input | `[(T1, 2048), (T2, 2048), (T3, 2048), (T4, 2048)]` | Raw features per video |
| After base_model | `[(T1, 128), (T2, 128), (T3, 128), (T4, 128)]` | Encoded features |
| After heads | `[[(T1,128)]*3, [(T2,128)]*3, ...]` | Per-head outputs |
| Attention weights | `[(T1, 3), (T2, 3), ...]` | Head weights per frame |
| Final outputs | `[(T1, 128), (T2, 128), ...]` | Weighted combination |

---

## 3. Loss Functions

### 3.1 GTCC Loss (Core Innovation)

**Location:** `utils/loss_functions.py:341-450`

GTCC extends TCC by fitting GMMs to capture multi-modal correspondences.

#### Mathematical Formulation

For primary video $X$ (frames $x_i$) and secondary video $Y$ (frames $y_j$):

**Step 1: Compute soft nearest neighbors**
$$\alpha_{ij} = \frac{\exp(-\|x_i - y_j\|_2 / \tau)}{\sum_k \exp(-\|x_i - y_k\|_2 / \tau)}$$

**Step 2: Fit GMM to $\alpha_i$ (row of ALPHA matrix)**

For each frame $i$ in the primary video, fit a GMM with $K$ components to the soft correspondence distribution $\alpha_i$:
$$p(j | i) = \sum_{k=1}^K \pi_k \cdot \mathcal{N}(j; \mu_k, \sigma_k^2)$$

The GMM is fit using L-BFGS optimization to minimize KL divergence between the GMM and $\alpha_i$.

**Step 3: Compute cycle-back via GMM soft nearest neighbors**

$$SNN_k = \sum_j g_{ik,j} \cdot y_j$$

Where $g_{ik,j}$ is the $k$-th Gaussian component's contribution at position $j$.

**Step 4: Compute cycle consistency loss**

For each GMM component $k$:
$$\beta_{ik,j} = \frac{\exp(-\|SNN_k - x_j\|_2 / \tau) \cdot M_{ij}}{\sum_l \exp(-\|SNN_k - x_l\|_2 / \tau) \cdot M_{il}}$$

Where $M_{ij}$ is a margin identity matrix (diagonal band).

Expected return position:
$$\mu_{ik} = \sum_j \beta_{ik,j} \cdot j$$

**Step 5: Alignment loss**

$$\mathcal{L}_{align} = \sum_i \sum_k \pi_k \cdot (i - \mu_{ik})^2$$

Plus optional variance term if `alignment_variance > 0`.

**Step 6: Dropout weighting (optional)**

If `gamma < 1`:
$$\mathcal{L}_{GTCC} = B_i \cdot \mathcal{L}_{align} + (1 - B_i) \cdot \frac{1}{\mathcal{L}_{align}}$$

Where $B_i$ is the learned dropout weight for frame $i$.

#### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_components` | Number of GMM components | Configurable |
| `gamma` | Dropout annealing factor (1.0 = no dropout) | 1.0 |
| `delta` | Margin ratio for identity matrix (0.05-1.0) | Configurable |
| `softmax_temp` | Temperature for softmax (lower = sharper) | 0.1 |
| `max_gmm_iters` | L-BFGS iterations for GMM fitting | 8 |
| `alignment_variance` | Weight for variance penalty | 0 |

#### GMM Fitting Details

**Location:** `utils/tensorops.py:124-235`

```python
def get_gmm_lfbgf(probability, n_components, ...):
    # Initialize means evenly spaced
    means = [1 + i * (N-2)/n_components for i in range(n_components)]
    # Initialize stds to sequence length
    stds = [N for _ in range(n_components)]
    # Initialize spreads uniformly
    spreads = [1 for _ in range(n_components)]

    # L-BFGS optimization minimizing KL divergence
    for _ in range(max_iters):
        lbfgs.step(kl_divergence_loss)

    return gaussians, spread_weights, means
```

### 3.2 TCC Loss (Baseline)

**Location:** `utils/loss_functions.py:18-49`

Original Temporal Cycle Consistency loss.

```python
def TCC_loss(sequences, alignment_variance=0.1, softmax_temp=1):
    for primary in sequences:
        for secondary in sequences:
            ALPHA = softmax(-cdist(primary, secondary) / temp)
            SNN = ALPHA @ secondary
            BETA = softmax(-cdist(SNN, primary) / temp)
            mus = BETA @ idx_range
            variances = BETA @ (idx_range - mus)^2
            loss += (idx_range - mus)^2 / variances + alignment_variance * log(sqrt(variances))
```

Key difference from GTCC: TCC uses a single soft nearest neighbor per frame, while GTCC uses multiple GMM components.

### 3.3 LAV Loss

**Location:** `utils/loss_functions.py:137-190`

Learning by Aligning Videos loss using SoftDTW.

```python
def LAV_loss(sequences, min_temp=0.1, cr_coefficient=0.01):
    softdtw = SoftDTW(gamma=min_temp)
    for v1, v2 in pairs(sequences):
        soft_dtw_term = softdtw(v1, v2)
        x_cr_term = IntraContrast_loss(v1, window=15)
        y_cr_term = IntraContrast_loss(v2, window=15)
        loss += soft_dtw_term + cr_coefficient * (x_cr_term + y_cr_term)
```

Components:
- **SoftDTW**: Differentiable Dynamic Time Warping for alignment
- **IntraContrast**: Encourages temporal smoothness within videos

### 3.4 VAVA Loss

**Location:** `utils/loss_functions.py:266-338`

Video Alignment via Variational Approach using optimal transport.

```python
def VAVA_loss(sequences, global_step, maxIter=20, zeta=0.5, delta=0.6, gamma=0.5):
    phi = 0.999 ** sqrt(global_step + 1)  # Annealing schedule

    for v1, v2 in pairs(sequences):
        D = cdist(v1, v2)
        # Add virtual bins
        D = add_dustbin_rows_cols(D, zeta)
        # Sinkhorn optimal transport
        dist, T = sink(D, reg=lambda2, ...)

        # Alignment regularization
        lc = diag_consistency_matrix(...)
        lo = diag_optimality_matrix(T, ...)
        P = phi * exp(-lc^2/2δ²) + (1-phi) * exp(-lo^2/2δ²)

        # KL divergence
        KL = sum(T * log(T/P))

        # Information terms
        I_T = phi * Ic + (1-phi) * Io

        vava_loss = dist - lambda1 * I_T + lambda2 * KL
        cr_loss = IntraContrast(v1) + IntraContrast(v2)
        loss += vava_loss + gamma * cr_loss
```

### 3.5 Loss Comparison Table

| Loss | Method | Key Feature | Memory |
|------|--------|-------------|--------|
| **GTCC** | GMM + Cycle Consistency | Multi-modal correspondences | Medium |
| **TCC** | Soft NN + Cycle Consistency | Single correspondence | Low |
| **LAV** | SoftDTW + Contrastive | DTW-based alignment | High |
| **VAVA** | Optimal Transport + Prior | Sinkhorn algorithm | Medium-High |

---

## 4. Training Pipeline

### 4.1 Entry Points

| Script | Purpose | Description |
|--------|---------|-------------|
| `singletask_train.py` | Single task training | Trains one model per action class |
| `multitask_train.py` | Multi-task training | Trains one model across all action classes |

### 4.2 Configuration System

**Location:** `configs/generic_config.py`

```python
CONFIG = edict()

# Dataset settings
CONFIG.DATASET_NAME = None      # 'egoprocel', 'cmu', 'egtea', etc.
CONFIG.DATAFOLDER = None        # Path to features
CONFIG.EVAL_PLOTFOLDER = None   # Output directory

# Base architecture
CONFIG.BASEARCH.ARCHITECTURE = None  # 'temporal-stacking', 'resnet50', 'naive'
CONFIG.BASEARCH.TEMPORAL_STACKING_ARCH = {
    'temporal_depth': 2,
    'conv_num_channels': 256,
    'drop_layers': [256, 1024, 512, 256],
}

# Overall architecture
CONFIG.ARCHITECTURE = {
    'MCN': False,               # Use multi-head attention
    'num_heads': None,          # Number of attention heads
    'attn_layers': [512, 1024, 512, 512],
    'drop_layers': [256, 1024, 512, 256],
}

# Loss selection
CONFIG.LOSS_TYPE = {
    'tcc': False,
    'GTCC': False,
    'LAV': False,
    'VAVA': False,
}

# GTCC-specific parameters
CONFIG.GTCC_PARAMS = {
    'softmax_temp': 0.1,
    'max_gmm_iters': 8,
    'n_components': None,
    'delta': None,
    'gamma': None,
    'alignment_variance': 0,
}

# Training parameters
CONFIG.SKIP_RATE = None         # Frame subsampling
CONFIG.BATCH_SIZE = None        # Batch size
CONFIG.LEARNING_RATE = None     # Learning rate
CONFIG.NUM_EPOCHS = None        # Training epochs
CONFIG.LAZY_LOAD = True         # Load data on-demand
```

### 4.3 Training Loop

**Location:** `models/alignment_training_loop.py:199-431`

```python
def alignment_training_loop(model, train_dl_dict, loss_fn, foldername, CONFIG, ...):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        all_sub_batches = _get_all_batches_with_taskid(train_dl_dict)

        for task, (inputs, times) in all_sub_batches:
            optimizer.zero_grad()

            # Preprocess: load data, apply skip_rate
            inputs, times = preprocess_batch(inputs, times, skip_rate=CONFIG.SKIP_RATE)

            # Skip short sequences
            if any(seq.shape[0] < MIN_SEQ_LENGTH for seq in inputs):
                continue

            # Skip if pairwise product too large (OOM prevention)
            if max_pairwise > threshold:
                continue

            # Forward pass
            output_dict = model(inputs)

            # Compute loss
            loss_dict = loss_fn(output_dict, epoch)
            loss = loss_dict['total_loss']

            # Backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.00001)
            optimizer.step()

        # Checkpoint saving
        if epoch % 5 == 0:
            ckpt_save(model, optimizer, epoch, loss, ...)
```

### 4.4 Data Loading

**Location:** `models/json_dataset.py`

```python
class JSONDataset(Dataset):
    """
    Dataset structure:
        data_folder/
            - {handle}.npy       # Features (T, D)

    JSON structure:
        {
            task_name: {
                'handles': [video_names],
                'hdl_actions': [[action1, action2, ...]],
                'hdl_start_times': [[t1, t2, ...]],
                'hdl_end_times': [[t1, t2, ...]]
            }
        }
    """

    def __getitem__(self, index):
        filepath, time_dict, name = self.data_label_name[index]
        return filepath, time_dict, name  # Lazy loading
```

---

## 5. Evaluation Metrics

**Location:** `utils/evaluation.py`

### 5.1 Available Metrics

| Metric | Class | Description |
|--------|-------|-------------|
| **Phase Progression** | `PhaseProgression` | R² of linear regression from embeddings to progress |
| **Phase Classification** | `PhaseClassification` | SVM classification accuracy of action phases |
| **Kendall's Tau** | `KendallsTau` | Rank correlation of nearest neighbor ordering |
| **Aligned Kendall's Tau** | `WildKendallsTau` | KT respecting ground truth alignable frames |
| **EAE** | `EnclosedAreaError` | Enclosed area error between predicted and true alignment |
| **OGPE** | `OnlineGeoProgressError` | Online geodesic progress estimation error |

### 5.2 Progress Estimation

**Location:** `utils/tensorops.py:298-310`

Progress is computed as cumulative embedding distance normalized by training set statistics:

```python
def get_cum_matrix(video):
    """Compute cumulative distance (geodesic) through embedding space"""
    P = torch.zeros(video.shape[0])
    for t in range(1, video.shape[0]):
        P[t] = P[t-1] + torch.linalg.norm(video[t] - video[t-1])
    return P

# Normalization (in OnlineGeoProgressError):
pred_progress = get_cum_matrix(outputs) / train_cum_means[task]
```

Ground truth progress is computed from action annotations:
```python
def get_trueprogress(time_dict):
    """Each non-SIL action contributes 1/num_actions to progress"""
    prg_per_action = 1 / num_non_sil_actions
    for action in actions:
        if action != 'SIL':
            progress[start:end] = linspace(prev_prg, prev_prg + prg_per_action)
            prev_prg += prg_per_action
```

---

## 6. Key Files Reference

| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `singletask_train.py` | Main training entry point | Training loop setup |
| `multitask_train.py` | Multi-task training entry | Same as singletask |
| `models/model_multiprong.py` | Main GTCC model | `MultiProngAttDropoutModel`, `HeadModel` |
| `models/model_singleprong.py` | Base encoders | `StackingEncoder`, `Resnet50Encoder`, `NaiveEncoder` |
| `models/alignment_training_loop.py` | Training loop | `alignment_training_loop()` |
| `models/json_dataset.py` | Data loading | `JSONDataset`, `jsondataset_get_train_test()` |
| `utils/loss_functions.py` | All loss functions | `GTCC_loss()`, `TCC_loss()`, `LAV_loss()`, `VAVA_loss()` |
| `utils/loss_entry.py` | Loss function wrapper | `get_loss_function()` |
| `utils/tensorops.py` | GMM fitting, preprocessing | `get_gmm_lfbgf()`, `preprocess_batch()`, `get_cum_matrix()` |
| `utils/evaluation.py` | Evaluation metrics | `PhaseProgression`, `KendallsTau`, `OnlineGeoProgressError`, etc. |
| `configs/generic_config.py` | Configuration template | `CONFIG` easydict |
| `configs/entry_config.py` | Config parser | `get_generic_config()` |

---

## 7. Architectural Diagrams

### 7.1 GTCC Loss Computation Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                        GTCC LOSS COMPUTATION                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Primary Video X: (N, D)          Secondary Video Y: (M, D)          │
│         │                                  │                          │
│         └────────────┬────────────────────┘                          │
│                      ▼                                                │
│         ┌─────────────────────────┐                                  │
│         │ ALPHA = softmax(-cdist) │  Shape: (N, M)                   │
│         │ Soft correspondence     │                                  │
│         └───────────┬─────────────┘                                  │
│                     ▼                                                 │
│    ┌────────────────────────────────────┐                            │
│    │ For each row alpha_i (sampled):    │                            │
│    │   GMM = fit_gmm(alpha_i)           │                            │
│    │   → K Gaussians with (μ_k, σ_k, π_k)                            │
│    └────────────────┬───────────────────┘                            │
│                     ▼                                                 │
│    ┌────────────────────────────────────┐                            │
│    │ SNN_k = GMM_k @ Y                  │  Soft NN per component     │
│    │ Shape: (num_sampled, K, D)         │                            │
│    └────────────────┬───────────────────┘                            │
│                     ▼                                                 │
│    ┌────────────────────────────────────┐                            │
│    │ BETA = margin_mask * softmax(      │                            │
│    │        -cdist(SNN, X))             │  Cycle back                │
│    └────────────────┬───────────────────┘                            │
│                     ▼                                                 │
│    ┌────────────────────────────────────┐                            │
│    │ μ_return = BETA @ indices          │  Expected return position  │
│    │ error = (i - μ_return)²            │                            │
│    └────────────────┬───────────────────┘                            │
│                     ▼                                                 │
│    ┌────────────────────────────────────┐                            │
│    │ loss = Σ π_k * error_k             │  Weighted by GMM spreads   │
│    │ (optional: * dropout_weight)       │                            │
│    └────────────────────────────────────┘                            │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 7.2 Model Architecture (MCN Mode)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MultiProngAttDropoutModel (MCN)                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input: List[(T_i, 2048)]                                           │
│         │                                                            │
│         ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                      StackingEncoder                         │    │
│  │  Conv1d(2048→256, k=temporal_depth) → ReLU                  │    │
│  │  Conv1d(256→256) → ReLU → ... → Conv1d(→128)               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│         │                                                            │
│         ▼  general_features: List[(T_i, 128)]                       │
│         │                                                            │
│    ┌────┴────┬────────┬────────┐                                    │
│    ▼         ▼        ▼        ▼                                    │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                                │
│  │Head │  │Head │  │Head │  │Head │    num_heads = 4               │
│  │  0  │  │  1  │  │  2  │  │  3  │                                │
│  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘                                │
│     │        │        │        │       (T, 128) each                │
│     └────────┴────────┴────────┘                                    │
│                 │                                                    │
│                 ▼  Concatenate: (T, 128*4) = (T, 512)               │
│         ┌─────────────────────────────────────────┐                 │
│         │           Attention Layer                │                 │
│         │  Linear(512→512) → Tanh                 │                 │
│         │  Linear(512→256) → Tanh                 │                 │
│         │  Linear(256→4) → Softmax                │                 │
│         └─────────────────────────────────────────┘                 │
│                 │                                                    │
│                 ▼  attention_weights: (T, 4)                        │
│                 │                                                    │
│         ┌─────────────────────────────────────────┐                 │
│         │  Weighted Sum over heads                 │                 │
│         │  output = Σ attention[k] * head_k        │                 │
│         └─────────────────────────────────────────┘                 │
│                 │                                                    │
│                 ▼  outputs: List[(T_i, 128)]                        │
│                 │                                                    │
│         ┌───────┴───────┐ (if dropping=True)                        │
│         ▼               │                                            │
│  ┌─────────────┐        │                                           │
│  │Dropout Net  │        │                                           │
│  │Linear→ReLU  │        │                                           │
│  │ →(129,)     │        │                                           │
│  └─────────────┘        │                                           │
│         │               │                                            │
│         ▼               ▼                                            │
│  Output: {'outputs': [...], 'attentions': [...], 'dropouts': [...]} │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 8. Implementation Notes

### 8.1 Memory Management

The training loop includes safeguards for OOM:

```python
MAX_PAIRWISE_THRESHOLDS = {
    'LAV': 4_000_000,    # LAV uses more memory (SoftDTW)
    'GTCC': 8_000_000,   # ~8M pairwise products max
    'tcc': 8_000_000,    # Same as GTCC
    'VAVA': 6_000_000,   # VAVA moderate memory (OT)
}
```

Batches are skipped if `seq_lengths[0] * seq_lengths[1] > threshold`.

### 8.2 Gradient Clipping

Very aggressive gradient clipping is used:
```python
nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.00001, norm_type=2)
```

### 8.3 GMM Fitting Notes

- Uses L-BFGS optimizer with strong Wolfe line search
- `max_iters=8` is typically sufficient
- Means initialized evenly spaced across sequence
- Variances initialized to sequence length (conservative)
- If sequence length < 2, GMM fitting is skipped

---

## 9. Extension Points

### Adding a New Loss Function

1. Implement in `utils/loss_functions.py`:
```python
def NEW_loss(sequences, param1=default, ...):
    # Compute loss
    return loss_tensor
```

2. Add to `CONFIG.LOSS_TYPE` in `configs/generic_config.py`
3. Add parameters to config: `CONFIG.NEW_PARAMS = {...}`
4. Add to `utils/loss_entry.py`:
```python
elif loss_term == 'NEW':
    specific_loss = NEW_loss(output_dict['outputs'], **NEW_PARAMS)
```

### Adding a New Encoder

1. Implement in `models/model_singleprong.py`:
```python
class NewEncoder(nn.Module):
    def forward(self, videos):
        # Return: {'outputs': List[(T_i, D_out)], 'dropouts': [...] optional}
```

2. Add to `get_base_model_deets()` in `utils/train_util.py`
3. Add architecture config in `configs/generic_config.py`

---

*Document generated from GTCC codebase analysis. Last updated: 2026-02-12*
