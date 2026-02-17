# GTCC Architecture & Code Documentation

This document provides a comprehensive guide to the GTCC (Gaussian Temporal Cycle-Consistency) architecture, its code locations, how components work together in multitask settings, and how to make modifications.

---

## 1. Executive Summary

### Architecture Modes
GTCC supports two primary architecture modes:

1. **MCN (Multi-head Consistency Network)**: Uses multiple task-specific heads with an attention mechanism to dynamically combine head outputs per frame. Enabled via `CONFIG.ARCHITECTURE['MCN'] = True`.

2. **Simple Encoder**: Direct encoding without multi-head architecture. Uses `StackingEncoder`, `Resnet50Encoder`, or `NaiveEncoder` as the sole model.

### Key Architectural Innovation
The multi-head attention mechanism allows different frames to emphasize different task-specific heads, enabling the model to learn task-invariant representations while maintaining task-specific specializations.

### Purpose
Self-supervised video alignment for progress estimation using temporal cycle-consistency losses (TCC, GTCC, LAV, VAVA).

---

## 2. Architecture Components & Code Locations

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| `MultiProngAttDropoutModel` | `models/model_multiprong.py` | 143-202 | Main MCN model with attention over task-specific heads |
| `HeadModel` | `models/model_multiprong.py` | 204-220 | Task-specific head processing |
| `StackingEncoder` | `models/model_singleprong.py` | 84-177 | 1D Conv temporal encoder (recommended) |
| `Resnet50Encoder` | `models/model_singleprong.py` | 11-82 | 3D Conv spatial-temporal encoder |
| `NaiveEncoder` | `models/model_singleprong.py` | 180-227 | Simple MLP encoder |
| `GTCC_loss` | `utils/loss_functions.py` | 341-450 | Core GTCC loss with GMM fitting |
| `TCC_loss` | `utils/loss_functions.py` | 18-49 | Original TCC loss |
| `LAV_loss` | `utils/loss_functions.py` | 137-190 | LAV loss with SoftDTW |
| `VAVA_loss` | `utils/loss_functions.py` | 266-338 | VAVA loss with optimal transport |
| `get_gmm_lfbgf` | `utils/tensorops.py` | 124-235 | L-BFGS GMM fitting |
| `get_cum_matrix` | `utils/tensorops.py` | 327-331 | Non-learnable progress estimation |
| `alignment_training_loop` | `models/alignment_training_loop.py` | 199-431 | Main training loop |
| `get_loss_function` | `utils/loss_entry.py` | 8-65 | Loss function factory |
| `multitask_train.py` | root | 1-207 | Multi-task training entry point |
| Generic Config | `configs/generic_config.py` | 1-92 | Configuration template |

---

## 3. Data Flow Diagrams

### Complete Forward Pass

```
Input: List of videos [(T_i, 2048) for i in batch]
                    │
                    ▼
    ┌───────────────────────────────────┐
    │       Base Model (e.g., StackingEncoder)       │
    │  Input: [(T_i, 2048)]                          │
    │  Output: [(T_i, 128)]                          │
    │  File: models/model_singleprong.py:84-177      │
    └───────────────────────────────────┘
                    │
                    ▼ general_features: List[(T_i, 128)]
    ┌───────────────────────────────────┐
    │       HeadModel × num_heads                    │
    │  Each head: Linear(128→512)→ReLU→             │
    │             Linear(512→128)→ReLU→             │
    │             Linear(128→256)→ReLU→             │
    │             Linear(256→128)                   │
    │  File: models/model_multiprong.py:204-220     │
    └───────────────────────────────────┘
                    │
                    ▼ prong_outputs: List[List[(T_i, 128)]]
                      (batch × heads)
    ┌───────────────────────────────────┐
    │       Attention Mechanism                      │
    │  Concat heads: [T, 128×num_heads]              │
    │  Attention layer: → [T, num_heads]             │
    │  Weighted combination: → [T, 128]              │
    │  File: models/model_multiprong.py:186-193     │
    └───────────────────────────────────┘
                    │
                    ▼ combined_embedding: [(T_i, 128)]
    ┌───────────────────────────────────┐
    │       Dropout Network (if GTCC)                │
    │  Linear(128→256)→ReLU→                         │
    │  Linear(256→1024)→ReLU→                        │
    │  Linear(1024→512)→ReLU→                        │
    │  Linear(512→256)→ReLU→                         │
    │  Linear(256→129) → mean(dim=0)                 │
    │  Output: [129] (128 weights + 1 bias)          │
    │  File: models/model_multiprong.py:172-177     │
    └───────────────────────────────────┘
                    │
                    ▼
    Output Dict: {
        'outputs': [(T_i, 128)],      # Final embeddings
        'attentions': [(T_i, num_heads)],  # Attention weights
        'dropouts': [(129,)]          # Dropout vectors (if GTCC)
    }
```

### Tensor Shapes Throughout

```python
# Input
videos: List[Tensor]  # [(T_i, 2048) for i in batch_size]

# After base model
general_features: List[Tensor]  # [(T_i, 128) for i in batch_size]

# After heads (before attention)
prong_outputs: List[List[Tensor]]  # [batch_size][num_heads] = (T_i, 128)

# During attention computation (per video)
prong_output_t: Tensor  # [num_heads, T, 128]
concatenated_prongs: Tensor  # [T, 128*num_heads]
attention_weights: Tensor  # [T, num_heads]
weighted_combination: Tensor  # [128, T, num_heads]
combined_embedding: Tensor  # [T, 128]

# Final outputs
outputs: List[Tensor]  # [(T_i, 128) for i in batch_size]
attentions: List[Tensor]  # [(T_i, num_heads) for i in batch_size]
dropouts: List[Tensor]  # [(129,) for i in batch_size] (if GTCC)
```

---

## 4. Attention Mechanism Deep-Dive

### What It Does
The attention mechanism dynamically weights the contribution of each task-specific head for every frame. This allows the model to:
- Use different head combinations for different parts of a video
- Learn which heads are most relevant for specific frame types
- Produce task-invariant final embeddings from task-specific intermediate representations

### Code Flow (model_multiprong.py:186-193)

```python
for prong_output in prong_outputs:  # For each video in batch
    T = prong_output[0].shape[0]

    # Stack head outputs: [num_heads, T, 128]
    prong_output_t = torch.stack(prong_output, dim=0)

    # Concatenate for attention input: [T, 128*num_heads]
    concatenated_prongs = torch.stack(prong_output, dim=0).view(T, -1)

    # Compute attention weights: [T, num_heads]
    attention_weights = self.attention_layer(concatenated_prongs)

    # Weight each head's contribution: [128, T, num_heads]
    weighted_combination = prong_output_t.permute(2,1,0) * attention_weights

    # Sum across heads to get final embedding: [T, 128]
    combined_embedding = weighted_combination.sum(dim=2).T
```

### Attention Layer Architecture (model_multiprong.py:165-171)

```python
self.attention_layer = nn.Sequential(
    nn.Linear(output_dimensionality * num_heads, attn_layers[0]),  # 128*N → 512
    # Followed by layers from attn_layers with Tanh activations
    *get_linear_layers_w_activations(attn_layers, activation_at_end=True, activation=nn.Tanh()),
    nn.Linear(attn_layers[-1], num_heads),  # → num_heads
    nn.Softmax(dim=1)  # Normalize to sum to 1 per frame
)
```

Default `attn_layers = [512, 1024, 512, 512]` gives:
```
Input: [T, 128*num_heads]
  → Linear(128*N, 512) + Tanh
  → Linear(512, 1024) + Tanh
  → Linear(1024, 512) + Tanh
  → Linear(512, 512) + Tanh
  → Linear(512, num_heads)
  → Softmax
Output: [T, num_heads]
```

### Purpose of Multiple Heads
1. **Task Specialization**: Each head can learn features relevant to a specific task or action type
2. **Dynamic Combination**: Attention allows frames to weight heads differently based on content
3. **Information Preservation**: Different heads preserve different aspects of the base features
4. **Flexibility**: The model can adapt its representation based on what's happening in each frame

---

## 5. Head Modules Explanation

### HeadModel Architecture (model_multiprong.py:204-220)

```python
class HeadModel(nn.Module):
    def __init__(self, output_dimensionality, class_name, layers=[512, 128, 256]):
        self.fc_layers = nn.Sequential(
            nn.Linear(output_dimensionality, layers[0]),  # 128 → 512
            *get_linear_layers_w_activations(layers, activation_at_end=True, activation=nn.ReLU()),
            nn.Linear(layers[-1], output_dimensionality)  # 256 → 128
        )
```

This expands to:
```
Input: [T, 128] (from base model)
  → Linear(128, 512) + ReLU
  → Linear(512, 128) + ReLU
  → Linear(128, 256) + ReLU
  → Linear(256, 128)
Output: [T, 128] (task-specific embeddings)
```

### Purpose
- **Transform shared features**: Convert generic base features into task-specific representations
- **Non-linear mapping**: MLPs allow complex non-linear transformations
- **Same dimensionality**: Input and output are both 128-dim, allowing seamless attention combination
- **Bottleneck structure**: 512 → 128 → 256 → 128 forces the network to learn compressed representations

---

## 6. Dropout Network & GTCC Integration

### Dropout Network Architecture (model_multiprong.py:172-177)

When `dropping=True` (enabled for GTCC loss):

```python
self.dropout = nn.Sequential(
    nn.Linear(output_dimensionality, drop_layers[0]),  # 128 → 256
    *get_linear_layers_w_activations(drop_layers, activation_at_end=True, activation=nn.ReLU()),
    nn.Linear(drop_layers[-1], output_dimensionality + 1)  # 256 → 129
)
```

With default `drop_layers = [256, 1024, 512, 256]`:
```
Input: combined_embedding [T, 128]
  → Linear(128, 256) + ReLU
  → Linear(256, 1024) + ReLU
  → Linear(1024, 512) + ReLU
  → Linear(512, 256) + ReLU
  → Linear(256, 129)
  → mean(dim=0)  # Average over time
Output: [129] (128 weights + 1 bias)
```

### Usage in GTCC Loss (loss_functions.py:379-382)

```python
if gamma < 1:  # When curriculum learning is enabled
    # Apply dropout-learned linear projection
    BX = primary @ dropouts[j][:-1].squeeze() + dropouts[j][-1]  # Linear: w·x + b

    # Normalize to zero mean, unit variance
    BX = (BX - BX.mean()) / BX.std()

    # Apply curriculum scaling with sigmoid
    BX = drop_min + (1-drop_min) * nn.Sigmoid()(BX)
```

Where:
- `dropouts[j][:-1]` = 128-dim weight vector
- `dropouts[j][-1]` = scalar bias
- `drop_min = gamma^(epoch+1)` = curriculum scaling factor

### Purpose
The dropout network learns to identify which frames should contribute less to the loss:
- **Soft masking**: Frames with low BX values contribute less to alignment loss
- **Curriculum learning**: Early epochs (high `gamma^epoch`) treat all frames equally; later epochs allow selective weighting
- **Outlier handling**: Learns to downweight frames that don't align well across videos

---

## 7. Multitask Setting

### How Tasks Are Combined

**Entry Point**: `multitask_train.py:92-156`

1. **Separate DataLoaders per task**:
```python
train_dataloaders = {}
for task in TASKS:
    train_set, test_set = jsondataset_get_train_test(task=task, ...)
    train_dataloaders[task] = DataLoader(train_set, ...)
```

2. **Interleaved batch iteration** (alignment_training_loop.py:529-532):
```python
def _get_all_batches_with_taskid(dl_dict):
    all_batches = [(task, (inputs, times)) for task, dl in dl_dict.items()
                   for i, (inputs, times) in enumerate(dl)]
    random.shuffle(all_batches)  # Shuffle across all tasks
    return all_batches
```

3. **Single model processes all tasks**:
```python
for i, (task, (inputs, times)) in enumerate(all_sub_batches):
    output_dict = model(inputs)  # Same model for all tasks
    loss_dict = loss_fn(output_dict, epoch)
```

### MCN=True in Multitask (multitask_train.py:161-175)

```python
if CFG.ARCHITECTURE['MCN']:
    if CFG.ARCHITECTURE['num_heads'] is None:
        num_tks = len(TASKS)  # One head per task
    else:
        num_tks = CFG.ARCHITECTURE['num_heads']  # Fixed number of heads

    model = MultiProngAttDropoutModel(
        base_model_class=base_model_class,
        base_model_params=base_model_params,
        output_dimensionality=CFG.OUTPUT_DIMENSIONALITY,
        num_heads=num_tks,
        dropping=CFG.LOSS_TYPE['GTCC'],
        attn_layers=CFG.ARCHITECTURE['attn_layers'],
        drop_layers=CFG.ARCHITECTURE['drop_layers'],
    )
```

**Key Points**:
- `num_heads` can match number of tasks or be a fixed value
- Each head can specialize for different task characteristics
- Attention learns which heads are relevant per frame
- All tasks share the same base model weights

---

## 8. Current Progress Estimation (Non-Learnable)

### Location
- `utils/tensorops.py:327-331` - Core function
- `utils/evaluation.py:222-308` - Usage in `OnlineGeoProgressError`

### Method

```python
def get_cum_matrix(video):
    """Compute cumulative L2 distance between consecutive frames."""
    P = torch.zeros(video.shape[0])
    for t in range(1, video.shape[0]):
        P[t] = P[t-1] + torch.linalg.norm(video[t] - video[t-1])
    return P
```

### Usage in Evaluation (evaluation.py:281-286)

```python
# Get cumulative distance for test video
pred2_progress = get_cum_matrix(outputs)

# Normalize by training set mean
pred2_progress = pred2_progress / train_cum_means[task]

# Compare to ground truth
gpe = torch.mean(torch.abs(true_progress - pred2_progress))
```

### Limitations
1. **Pure heuristic**: Based only on embedding space distances, not learned
2. **Task-dependent scaling**: Requires training set statistics for normalization
3. **Sensitive to embedding quality**: Relies on well-structured embedding space
4. **No explicit progress signal**: Progress is emergent, not directly supervised

---

## 9. Adding a Learnable Progress Head

### Existing Architecture (Current State)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CURRENT ARCHITECTURE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

    Input: List[(T_i, 2048)]
              │
              ▼
    ┌─────────────────────┐
    │     Base Model      │
    │  (StackingEncoder)  │
    │   2048 → 128        │
    └─────────────────────┘
              │
              ▼  List[(T_i, 128)]
    ┌─────────────────────────────────────────────┐
    │              Task-Specific Heads             │
    │  ┌─────────┐  ┌─────────┐      ┌─────────┐  │
    │  │ Head 0  │  │ Head 1  │ ...  │ Head N  │  │
    │  │128→128  │  │128→128  │      │128→128  │  │
    │  └────┬────┘  └────┬────┘      └────┬────┘  │
    └───────┼────────────┼────────────────┼───────┘
            │            │                │
            └────────────┼────────────────┘
                         │
                         ▼  Concatenate: [T, 128*N]
    ┌─────────────────────────────────────────────┐
    │           Attention Mechanism                │
    │     Linear → Tanh → ... → Softmax           │
    │           [T, 128*N] → [T, N]               │
    └─────────────────────────────────────────────┘
                         │
                         ▼  Weighted sum
    ┌─────────────────────────────────────────────┐
    │          Combined Embedding                  │
    │              [T, 128]                        │
    └─────────────────────────────────────────────┘
                         │
           ┌─────────────┴─────────────┐
           │                           │
           ▼                           ▼
    ┌─────────────────┐      ┌─────────────────────┐
    │     outputs     │      │   Dropout Network   │
    │    [T, 128]     │      │  (only if GTCC)     │
    │                 │      │   128 → 129         │
    │                 │      │   mean(dim=0)       │
    │                 │      │   Output: [129]     │
    └─────────────────┘      └─────────────────────┘
           │                           │
           ▼                           ▼
    ┌─────────────────────────────────────────────┐
    │              Output Dictionary               │
    │  {                                          │
    │    'outputs': [T, 128],      ← Embeddings   │
    │    'attentions': [T, N],     ← Attn weights │
    │    'dropouts': [129]         ← For GTCC     │
    │  }                                          │
    └─────────────────────────────────────────────┘
                         │
                         ▼
    ┌─────────────────────────────────────────────┐
    │         Progress Estimation (Eval Only)      │
    │                                             │
    │   get_cum_matrix(outputs)                   │
    │   = Σ ||e[t] - e[t-1]||₂                    │
    │                                             │
    │   NON-LEARNABLE (pure heuristic)            │
    └─────────────────────────────────────────────┘
```

---

### Option A: Parallel Progress Head (Recommended)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    OPTION A: PARALLEL PROGRESS HEAD                          │
│                         (Recommended Approach)                               │
└─────────────────────────────────────────────────────────────────────────────┘

    Input: List[(T_i, 2048)]
              │
              ▼
    ┌─────────────────────┐
    │     Base Model      │
    │  (StackingEncoder)  │
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────────────────────────────┐
    │              Task-Specific Heads             │
    └─────────────────────────────────────────────┘
              │
              ▼
    ┌─────────────────────────────────────────────┐
    │           Attention Mechanism                │
    └─────────────────────────────────────────────┘
              │
              ▼
    ┌─────────────────────────────────────────────┐
    │          Combined Embedding [T, 128]         │
    └─────────────────────────────────────────────┘
              │
              ├─────────────────┬─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
    ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐
    │    outputs    │  │    Dropout    │  │  ★ PROGRESS HEAD  │  ← NEW
    │   [T, 128]    │  │    Network    │  │    (Learnable)    │
    │               │  │     [129]     │  │                   │
    │               │  │               │  │  Linear(128,256)  │
    │               │  │               │  │       + ReLU      │
    │               │  │               │  │  Linear(256,128)  │
    │               │  │               │  │       + ReLU      │
    │               │  │               │  │  Linear(128,1)    │
    │               │  │               │  │     + Sigmoid     │
    │               │  │               │  │                   │
    │               │  │               │  │  Output: [T, 1]   │
    └───────────────┘  └───────────────┘  └───────────────────┘
              │                 │                 │
              ▼                 ▼                 ▼
    ┌─────────────────────────────────────────────────────────┐
    │                    Output Dictionary                     │
    │  {                                                      │
    │    'outputs': [T, 128],                                 │
    │    'attentions': [T, N],                                │
    │    'dropouts': [129],                                   │
    │    'progress': [T]        ← NEW: Per-frame progress     │
    │  }                                                      │
    └─────────────────────────────────────────────────────────┘
```

**Advantages:**
- Clean separation of concerns
- Progress head trained jointly with alignment
- Shares base features with alignment task
- Can be easily enabled/disabled via config flag

**Code Changes Required:**

**1. Modify `MultiProngAttDropoutModel.__init__`** (model_multiprong.py:153-177):

```python
def __init__(
    self,
    base_model_class,
    base_model_params,
    output_dimensionality,
    num_heads,
    dropping=False,
    attn_layers=[512, 256],
    drop_layers=[512, 128, 256],
    progress_head=False,  # NEW PARAMETER
):
    super(MultiProngAttDropoutModel, self).__init__()
    # ... existing code ...

    # NEW: Progress head
    if progress_head:
        self.progress_head = nn.Sequential(
            nn.Linear(output_dimensionality, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
```

**2. Modify `forward`** (model_multiprong.py:179-202):

```python
def forward(self, videos):
    general_features = self.base_model(videos)['outputs']
    prong_outputs = [prong(general_features) for prong in self.head_models]
    prong_outputs = list(map(list, zip(*prong_outputs)))

    outputs = []
    attentions = []
    dropouts = []
    progresses = []  # NEW

    for prong_output in prong_outputs:
        # ... existing attention computation ...
        combined_embedding = weighted_combination.sum(dim=2).T
        outputs.append(combined_embedding)
        attentions.append(attention_weights)

        if self.dropping:
            dout = self.dropout(combined_embedding).mean(dim=0)
            dropouts.append(dout)

        # NEW: Progress prediction
        if hasattr(self, 'progress_head'):
            progress_out = self.progress_head(combined_embedding)  # [T, 1]
            progresses.append(progress_out.squeeze(-1))  # [T]

    result = {'outputs': outputs, 'attentions': attentions}
    if self.dropping:
        result['dropouts'] = dropouts
    if hasattr(self, 'progress_head'):
        result['progress'] = progresses  # NEW
    return result
```

**3. Add Progress Loss** (utils/loss_entry.py):

```python
def progress_loss(progress_preds, time_dicts):
    """Supervised progress loss using ground truth action segments."""
    total_loss = 0
    for pred, tdict in zip(progress_preds, time_dicts):
        true_progress = get_trueprogress(tdict).to(pred.device)
        total_loss += F.mse_loss(pred, true_progress)
    return total_loss / len(progress_preds)
```

---

### Option B: Extend Dropout Network with Shared Features

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               OPTION B: BRANCHED DROPOUT NETWORK                             │
│                  (Shared intermediate features)                              │
└─────────────────────────────────────────────────────────────────────────────┘

    Input: List[(T_i, 2048)]
              │
              ▼
    ┌─────────────────────┐
    │     Base Model      │
    └─────────────────────┘
              │
              ▼
    ┌─────────────────────────────────────────────┐
    │    Heads → Attention → Combined Embedding    │
    │                 [T, 128]                     │
    └─────────────────────────────────────────────┘
              │
              ├─────────────────────────────────────┐
              │                                     │
              ▼                                     ▼
    ┌───────────────┐            ┌─────────────────────────────────────┐
    │    outputs    │            │    ★ EXTENDED DROPOUT NETWORK       │
    │   [T, 128]    │            │                                     │
    └───────────────┘            │  ┌─────────────────────────────┐    │
                                 │  │     Shared Trunk            │    │
                                 │  │  Linear(128,256) + ReLU     │    │
                                 │  │  Linear(256,1024) + ReLU    │    │
                                 │  │  Linear(1024,512) + ReLU    │    │
                                 │  └──────────────┬──────────────┘    │
                                 │                 │                   │
                                 │        ┌────────┴────────┐          │
                                 │        │                 │          │
                                 │        ▼                 ▼          │
                                 │  ┌───────────┐   ┌─────────────┐    │
                                 │  │ Dropout   │   │ ★ Progress  │    │
                                 │  │   Head    │   │    Head     │    │
                                 │  │           │   │             │    │
                                 │  │ Lin(512,  │   │ Lin(512,    │    │
                                 │  │     256)  │   │     128)    │    │
                                 │  │ Lin(256,  │   │ Lin(128,1)  │    │
                                 │  │     129)  │   │ + Sigmoid   │    │
                                 │  │           │   │             │    │
                                 │  │mean(dim=0)│   │ (per-frame) │    │
                                 │  └─────┬─────┘   └──────┬──────┘    │
                                 │        │                │           │
                                 └────────┼────────────────┼───────────┘
                                          │                │
                                          ▼                ▼
                                       [129]             [T, 1]
                                          │                │
                                          ▼                ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         Output Dictionary                            │
    │  {                                                                  │
    │    'outputs': [T, 128],                                             │
    │    'attentions': [T, N],                                            │
    │    'dropouts': [129],           ← Existing dropout weights          │
    │    'progress': [T]              ← NEW: Shares features with dropout │
    │  }                                                                  │
    └─────────────────────────────────────────────────────────────────────┘
```

**Advantages:**
- Shared feature computation between dropout and progress
- Progress can benefit from curriculum learning features
- Potentially more parameter efficient

**Disadvantages:**
- More complex implementation
- Tighter coupling between dropout and progress heads
- Harder to train them with different schedules

**Code Changes Required:**

```python
class MultiProngAttDropoutModel(nn.Module):
    def __init__(self, ..., dropping=False, progress_head=False):
        # ...

        if dropping or progress_head:
            # Shared trunk
            self.shared_dropout_trunk = nn.Sequential(
                nn.Linear(output_dimensionality, 256),
                nn.ReLU(),
                nn.Linear(256, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
            )

        if dropping:
            # Dropout-specific head
            self.dropout_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, output_dimensionality + 1)
            )

        if progress_head:
            # Progress-specific head
            self.progress_branch = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

    def forward(self, videos):
        # ... attention computation ...

        for prong_output in prong_outputs:
            combined_embedding = ...  # [T, 128]

            if hasattr(self, 'shared_dropout_trunk'):
                shared_features = self.shared_dropout_trunk(combined_embedding)  # [T, 512]

                if self.dropping:
                    dout = self.dropout_head(shared_features).mean(dim=0)  # [129]
                    dropouts.append(dout)

                if hasattr(self, 'progress_branch'):
                    prog = self.progress_branch(shared_features).squeeze(-1)  # [T]
                    progresses.append(prog)
```

---

### Option C: Separate Progress Model (External)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                OPTION C: SEPARATE PROGRESS MODEL                             │
│                    (Decoupled architecture)                                  │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │                    MAIN MODEL (Unchanged)                            │
    │                                                                     │
    │  Input → Base → Heads → Attention → Combined Embedding              │
    │                                           │                         │
    │                              ┌────────────┴────────────┐            │
    │                              │                         │            │
    │                              ▼                         ▼            │
    │                         outputs                   Dropout           │
    │                        [T, 128]                   [129]             │
    │                              │                         │            │
    │                              ▼                         ▼            │
    │  Output: {'outputs': [...], 'dropouts': [...], 'attentions': [...]} │
    └─────────────────────────────────────────────────────────────────────┘
                                   │
                                   │  outputs: [T, 128]
                                   ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │               ★ SEPARATE PROGRESS MODEL (New Module)                 │
    │                                                                     │
    │     class ProgressHead(nn.Module):                                  │
    │                                                                     │
    │         ┌─────────────────────────────────┐                         │
    │         │       Input: [T, 128]           │                         │
    │         └───────────────┬─────────────────┘                         │
    │                         │                                           │
    │                         ▼                                           │
    │         ┌─────────────────────────────────┐                         │
    │         │    Linear(128, 256) + ReLU      │                         │
    │         └───────────────┬─────────────────┘                         │
    │                         │                                           │
    │                         ▼                                           │
    │         ┌─────────────────────────────────┐                         │
    │         │    Linear(256, 128) + ReLU      │                         │
    │         └───────────────┬─────────────────┘                         │
    │                         │                                           │
    │                         ▼                                           │
    │         ┌─────────────────────────────────┐                         │
    │         │  Linear(128, 1) + Sigmoid       │                         │
    │         └───────────────┬─────────────────┘                         │
    │                         │                                           │
    │                         ▼                                           │
    │         ┌─────────────────────────────────┐                         │
    │         │      Output: [T] (progress)     │                         │
    │         └─────────────────────────────────┘                         │
    └─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                      TRAINING SCRIPT                                 │
    │                                                                     │
    │  # Two separate models                                              │
    │  main_model = MultiProngAttDropoutModel(...)                        │
    │  progress_model = ProgressHead(embedding_dim=128)                   │
    │                                                                     │
    │  # Forward pass                                                     │
    │  output_dict = main_model(videos)                                   │
    │  embeddings = output_dict['outputs']                                │
    │  progress_preds = [progress_model(emb) for emb in embeddings]       │
    │                                                                     │
    │  # Separate optimizers possible                                     │
    │  optimizer_main = Adam(main_model.parameters())                     │
    │  optimizer_prog = Adam(progress_model.parameters())                 │
    └─────────────────────────────────────────────────────────────────────┘
```

**Advantages:**
- Complete decoupling - can train main model first, then add progress
- Can freeze main model weights while training progress head
- Easy to swap different progress head architectures
- Can use different learning rates/schedules

**Disadvantages:**
- Requires modifying training script, not just model
- No gradient flow from progress loss to main model (if frozen)
- Two models to manage and checkpoint

**Code Implementation:**

```python
# models/progress_head.py (NEW FILE)
class ProgressHead(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, embeddings):
        """
        Args:
            embeddings: [T, 128] frame embeddings
        Returns:
            progress: [T] progress values in [0, 1]
        """
        return self.layers(embeddings).squeeze(-1)
```

**Usage in training script:**
```python
# Initialize models
main_model = MultiProngAttDropoutModel(...)
progress_head = ProgressHead(embedding_dim=128)

# Option 1: Joint training
optimizer = Adam(list(main_model.parameters()) + list(progress_head.parameters()))

# Option 2: Separate training (freeze main model)
main_model.eval()
for p in main_model.parameters():
    p.requires_grad = False
optimizer = Adam(progress_head.parameters())

# Forward pass
output_dict = main_model(videos)
embeddings = output_dict['outputs']
progress_preds = [progress_head(emb) for emb in embeddings]

# Compute progress loss
progress_loss = compute_progress_loss(progress_preds, time_dicts)
```

---

### Comparison Summary

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      OPTIONS COMPARISON TABLE                               │
├──────────────┬──────────────┬───────────────┬──────────────────────────────┤
│   Aspect     │   Option A   │   Option B    │         Option C             │
│              │  (Parallel)  │  (Branched)   │        (Separate)            │
├──────────────┼──────────────┼───────────────┼──────────────────────────────┤
│ Complexity   │    Low       │    Medium     │         Low                  │
├──────────────┼──────────────┼───────────────┼──────────────────────────────┤
│ Code Changes │  model only  │  model only   │  model + training script     │
├──────────────┼──────────────┼───────────────┼──────────────────────────────┤
│ Parameter    │  Additional  │  Shared trunk │  Fully separate              │
│ Sharing      │  params only │  with dropout │                              │
├──────────────┼──────────────┼───────────────┼──────────────────────────────┤
│ Training     │  Joint       │  Joint        │  Joint or separate           │
│ Flexibility  │              │               │  (can freeze main)           │
├──────────────┼──────────────┼───────────────┼──────────────────────────────┤
│ Gradient     │  Yes         │  Yes (shared) │  Optional (if joint)         │
│ to Main Model│              │               │                              │
├──────────────┼──────────────┼───────────────┼──────────────────────────────┤
│ Best For     │  Standard    │  If dropout & │  Two-stage training or       │
│              │  use case    │  progress are │  pretrained alignment        │
│              │              │  related      │  models                      │
└──────────────┴──────────────┴───────────────┴──────────────────────────────┘
```

---

### Training Considerations

1. **Loss combination**: Add progress loss to total loss with a coefficient
   ```python
   total_loss = alignment_loss + lambda_progress * progress_loss
   ```

2. **Ground truth**: Use `get_trueprogress()` from `utils/tensorops.py` for supervision

3. **Curriculum**: Consider training alignment first, then adding progress head (especially with Option C)

4. **Evaluation**: Compare against `OnlineGeoProgressError` metric (utils/evaluation.py:222-308)

---

## 10. Configuration Reference

### Key Config Options (generic_config.py)

```python
CONFIG.ARCHITECTURE = {
    'MCN': True/False,           # Enable multi-head architecture
    'num_heads': None or int,    # Number of heads (None = len(tasks))
    'attn_layers': [512, 1024, 512, 512],  # Attention network layers
    'drop_layers': [256, 1024, 512, 256],  # Dropout network layers
}

CONFIG.BASEARCH.ARCHITECTURE = 'StackingEncoder'  # or 'Resnet50Encoder', 'NaiveEncoder'

CONFIG.BASEARCH.TEMPORAL_STACKING_ARCH = {
    'temporal_depth': 2,         # Conv1d kernel size
    'conv_num_channels': 256,    # Starting channels
    'output_dimensions': 128,    # Final embedding dim (set via OUTPUT_DIMENSIONALITY)
    'input_dimensions': 2048,    # Input features (ResNet)
    'drop_layers': [256, 1024, 512, 256],  # Dropout layers for non-MCN mode
}

CONFIG.LOSS_TYPE = {
    'GTCC': True/False,          # Enables dropout network
    'tcc': True/False,           # Original TCC loss
    'LAV': True/False,           # LAV with SoftDTW
    'VAVA': True/False,          # VAVA with optimal transport
}

CONFIG.GTCC_PARAMS = {
    'softmax_temp': .1,          # Softmax temperature for similarities
    'max_gmm_iters': 8,          # L-BFGS iterations for GMM
    'n_components': None,        # Number of GMM components
    'delta': None,               # Margin for stochastic identity matrix
    'gamma': None,               # Curriculum decay rate
    'alignment_variance': 0,     # Variance regularization term
}

CONFIG.OUTPUT_DIMENSIONALITY = 128  # Final embedding dimension
CONFIG.SKIP_RATE = None             # Frame subsampling rate
CONFIG.MULTITASK = False            # Multitask training flag
CONFIG.BATCH_SIZE = 4               # Batch size per GPU
CONFIG.LEARNING_RATE = 1e-4         # Learning rate
CONFIG.NUM_EPOCHS = 50              # Training epochs
```

### Loss-Specific Parameters

```python
CONFIG.TCC_ORIGINAL_PARAMS = {
    'softmax_temp': .1,          # Softmax temperature
    'alignment_variance': 0.001  # Variance regularization
}

CONFIG.LAV_PARAMS = {
    'min_temp': .1,              # SoftDTW gamma parameter
}

CONFIG.VAVA_PARAMS = {
    # Uses defaults from VAVA_loss function
}
```

---

## Quick Reference: Key Files

| Purpose | File |
|---------|------|
| Multi-head model | `models/model_multiprong.py` |
| Base encoders | `models/model_singleprong.py` |
| Loss functions | `utils/loss_functions.py` |
| Loss factory | `utils/loss_entry.py` |
| GMM fitting | `utils/tensorops.py` |
| Training loop | `models/alignment_training_loop.py` |
| Multitask entry | `multitask_train.py` |
| Configuration | `configs/generic_config.py` |
| Evaluation | `utils/evaluation.py` |

---

## Appendix: Helper Functions

### get_linear_layers_w_activations (utils/model_util.py:4-15)

```python
def get_linear_layers_w_activations(layers, activation=nn.ReLU(), activation_at_end=False):
    """Generate linear layers with activations between them.

    Example: layers=[512, 128, 256] produces:
        activation, Linear(512→128), activation, Linear(128→256)
        (+ final activation if activation_at_end=True)
    """
    if len(layers) == 0:
        return []
    output = []
    for i in range(len(layers)-1):
        output.append(activation)
        output.append(nn.Linear(layers[i], layers[i+1]))
    if activation_at_end:
        output.append(activation)
    return output
```

### get_trueprogress (utils/tensorops.py:298-310)

```python
def get_trueprogress(time_dict):
    """Compute ground truth progress from action segment annotations.

    Returns: Tensor[N] where N = total frames
             Progress increases linearly within each action segment,
             stays constant during SIL (silence/background) segments.
    """
```
