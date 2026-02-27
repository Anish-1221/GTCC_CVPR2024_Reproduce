# Progress Prediction Improvements: Transformer & Dilated Conv Architectures

## Overview

This plan details two new ProgressHead architectures to replace the current GRU-based implementation. Both solve the core problem of predicting action progress (0-1) without contradictory position signals.

**Key Constraints:**
- Causal/online inference only (frame t can only see frames 0 to t-1)
- Input: segment embeddings of shape `(T, 128)` where T varies
- Output: single scalar progress value [0, 1]
- Drop-in replacement for current ProgressHead

---

## Problem Recap

The current GRU-based ProgressHead faces a fundamental dilemma:
- **Without position encoding**: Outputs stuck around 0.5-0.6 (no temporal awareness)
- **With position encoding**: Same position (0.8 in 5-frame segment) maps to different targets:
  - Short action (5 frames total): target = 1.0
  - Long action (45 frames, 5 seen): target = 0.11

**Solution**: Use architectures that provide implicit temporal awareness without explicit position tokens.

---

## Architecture 1: TransformerProgressHead (with ALiBi)

### Concept

ALiBi (Attention with Linear Biases) adds a linear penalty to attention scores based on distance between tokens:
- Provides **relative positional awareness** without explicit position tokens
- Creates natural **recency bias** (recent frames get higher attention)
- No contradictory signals because no position value is provided

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    TransformerProgressHead                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: segment_embeddings (T, 128)                             │
│         ↓                                                        │
│  ┌──────────────────────┐                                       │
│  │ Input Projection     │  Linear(128 → 64) + LayerNorm         │
│  └──────────────────────┘                                       │
│         ↓ (T, 64)                                               │
│  ┌──────────────────────────────────────────────────────┐       │
│  │         Transformer Block × 2                         │       │
│  │  ┌────────────────────────────────────────────────┐  │       │
│  │  │  Causal Multi-Head Attention (4 heads)         │  │       │
│  │  │  ┌─────────────────────────────────────────┐   │  │       │
│  │  │  │ Q, K, V = Linear(64 → 64) each          │   │  │       │
│  │  │  │                                          │   │  │       │
│  │  │  │ Attention = softmax(QK^T/√d + ALiBi)    │   │  │       │
│  │  │  │                                          │   │  │       │
│  │  │  │ ALiBi_bias[i,j] = -m × (i-j) for j≤i    │   │  │       │
│  │  │  │ slopes m = [1, 0.5, 0.25, 0.125]        │   │  │       │
│  │  │  │                                          │   │  │       │
│  │  │  │ Causal mask: j > i → -∞                 │   │  │       │
│  │  │  └─────────────────────────────────────────┘   │  │       │
│  │  │  + Residual + LayerNorm                        │  │       │
│  │  └────────────────────────────────────────────────┘  │       │
│  │  ┌────────────────────────────────────────────────┐  │       │
│  │  │  Feed-Forward Network                          │  │       │
│  │  │  Linear(64 → 128) → GELU → Linear(128 → 64)   │  │       │
│  │  │  + Dropout(0.1) + Residual + LayerNorm        │  │       │
│  │  └────────────────────────────────────────────────┘  │       │
│  └──────────────────────────────────────────────────────┘       │
│         ↓ (T, 64)                                               │
│  ┌──────────────────────┐                                       │
│  │ Take Last Token      │  x[-1, :] → (64,)                     │
│  │ (Causal Aggregation) │  (encodes full past context)          │
│  └──────────────────────┘                                       │
│         ↓ (64,)                                                 │
│  ┌──────────────────────┐                                       │
│  │ Output MLP           │  Linear(64→32)→ReLU→Linear(32→1)     │
│  │                      │  → Sigmoid                            │
│  └──────────────────────┘                                       │
│         ↓                                                        │
│  Output: progress scalar [0, 1]                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### How ALiBi Works

#### The Problem with Standard Position Encoding

In standard transformers, we add a position value to each embedding:
```
frame_0_input = embedding_0 + position_encoding(0)
frame_1_input = embedding_1 + position_encoding(1)
...
```

This tells the model "you are at position X" but creates the contradiction we discussed - position 0.8 in a 5-frame segment means different things for different action lengths.

#### ALiBi: A Different Approach

ALiBi doesn't modify the input embeddings at all. Instead, it modifies the **attention scores** by adding a distance-based penalty.

```
Standard Attention:
  score[i,j] = Q[i] · K[j] / √d

ALiBi Attention:
  score[i,j] = Q[i] · K[j] / √d  -  m × (i - j)
               \_____________/     \__________/
                content-based       distance penalty
                similarity          (ALiBi bias)

Where m is a head-specific slope (e.g., 1, 0.5, 0.25, 0.125 for 4 heads)

Effect: Recent frames get higher attention, distant frames get penalized
        But NO explicit position value that could cause contradictions
```

#### Visual Example: Computing Attention for Frame 4

Say we have 5 frames and are computing attention for frame 4 (the last one):

```
Frame indices:     0      1      2      3      4
                   │      │      │      │      │
                   ▼      ▼      ▼      ▼      ▼

Standard attention scores (content similarity only):
Q[4] · K[j]:      0.3    0.5    0.2    0.8    0.6

ALiBi penalty (m=1, distance from frame 4):
-m × (4-j):       -4     -3     -2     -1      0

Final ALiBi scores:
                  -3.7   -2.5   -1.8   -0.2    0.6
                   │      │      │      │      │
                   ▼      ▼      ▼      ▼      ▼
After softmax:    0.01   0.03   0.06   0.30   0.60
                  └──────────────────────────────┘
                  Recent frames get more attention!
```

#### Multi-Head with Different Slopes

Each attention head uses a different slope `m`, creating diverse temporal patterns:

```
Head 1 (m=1.0):   Strong recency bias
  Frame 4 attending to frame 0: penalty = -4
  → Focuses on very recent frames (last 1-2)

Head 2 (m=0.5):   Medium recency bias
  Frame 4 attending to frame 0: penalty = -2
  → Balances recent and slightly older frames

Head 3 (m=0.25):  Weak recency bias
  Frame 4 attending to frame 0: penalty = -1
  → Can integrate information from moderate history

Head 4 (m=0.125): Very weak recency bias
  Frame 4 attending to frame 0: penalty = -0.5
  → Can attend to the entire sequence history
```

This allows the model to simultaneously:
- **Head 1**: Focus on immediate context (what just happened?)
- **Head 4**: Integrate information from the entire history (overall action pattern)

### Why ALiBi Solves the Contradiction

| Approach | What it tells the model | Problem |
|----------|------------------------|---------|
| Position encoding | "You're at position 0.8" | Ambiguous: 0.8 of what? |
| ALiBi | "Frame t-1 is 1 step away, t-5 is 5 steps away" | Relative, not absolute |

The model learns to combine **content patterns** with **relative distances** to infer progress.

### Forward Pass Pseudocode

```python
def forward(self, segment_embeddings):
    # segment_embeddings: (T, 128)

    # 1. Project to model dimension
    x = self.input_proj(segment_embeddings)  # (T, 64)
    x = x.unsqueeze(0)  # (1, T, 64)

    # 2. Create causal mask + ALiBi bias
    T = x.shape[1]
    causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()

    # ALiBi: distance-based penalty per head
    positions = torch.arange(T)
    distance = positions.unsqueeze(0) - positions.unsqueeze(1)  # (T, T)
    alibi_bias = torch.stack([
        -slope * torch.clamp(-distance, min=0)
        for slope in [1.0, 0.5, 0.25, 0.125]
    ])  # (4, T, T)

    # 3. Transformer layers with ALiBi attention
    for layer in self.layers:
        Q, K, V = layer.qkv(x)  # Each (1, T, 64)
        attn = (Q @ K.T / sqrt(d)) + alibi_bias  # Add ALiBi
        attn = attn.masked_fill(causal_mask, -inf)
        attn = softmax(attn) @ V
        x = layer.norm1(x + attn)
        x = layer.norm2(x + layer.ffn(x))

    # 4. Take last token (causal aggregation)
    final = x[0, -1, :]  # (64,)

    # 5. Output MLP
    return sigmoid(self.output_mlp(final))  # scalar [0,1]
```

### Hyperparameters

```python
TransformerProgressHead_CONFIG = {
    'input_dim': 128,
    'd_model': 64,
    'num_heads': 4,
    'num_layers': 2,
    'ffn_dim': 128,
    'dropout': 0.1,
    'alibi_slopes': [1.0, 0.5, 0.25, 0.125],
}
# ~76K parameters
```

### Potential Issues & Mitigations

| Issue | Mitigation |
|-------|------------|
| O(T²) attention complexity | T typically <100; chunk for T>200 |
| Needs more data than GRU | Small model (2 layers), dropout, weight decay |
| Cold start (T < 3) | Minimum segment length check in loss |

---

## Architecture 2: DilatedConvProgressHead

### Concept

Multi-scale dilated convolutions capture hierarchical temporal patterns:
- **Local (dilation=1)**: Frame-to-frame transitions
- **Medium (dilation=2,4)**: Short-term dynamics
- **Long (dilation=8,16,32)**: Action-level structure

Progress is inferred from **pattern of activations across scales**, not position.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                   DilatedConvProgressHead                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: segment_embeddings (T, 128)                             │
│         ↓                                                        │
│  ┌──────────────────────┐                                       │
│  │ Input Projection     │  Linear(128 → 64) + ReLU              │
│  └──────────────────────┘                                       │
│         ↓ (T, 64)                                               │
│  ┌──────────────────────┐                                       │
│  │ Transpose for Conv1d │  (T, 64) → (1, 64, T)                 │
│  └──────────────────────┘                                       │
│         ↓ (1, 64, T)                                            │
│  ┌──────────────────────────────────────────────────────┐       │
│  │      Dilated Residual Blocks × 6                      │       │
│  │                                                        │       │
│  │   Block 1: dilation=1   [receptive field: 3]          │       │
│  │   Block 2: dilation=2   [receptive field: 7]          │       │
│  │   Block 3: dilation=4   [receptive field: 15]         │       │
│  │   Block 4: dilation=8   [receptive field: 31]         │       │
│  │   Block 5: dilation=16  [receptive field: 63]         │       │
│  │   Block 6: dilation=32  [receptive field: 127]        │       │
│  │                                                        │       │
│  │   Each block:                                          │       │
│  │   ┌────────────────────────────────────────────────┐  │       │
│  │   │  Causal Pad (left only): (k-1) × dilation     │  │       │
│  │   │         ↓                                      │  │       │
│  │   │  Conv1d(64, 64, k=3, dilation=d)              │  │       │
│  │   │         ↓                                      │  │       │
│  │   │  BatchNorm → ReLU → Conv1d(64, 64, k=1)       │  │       │
│  │   │         ↓                                      │  │       │
│  │   │  Dropout(0.1)                                  │  │       │
│  │   │         ↓                                      │  │       │
│  │   │  + Residual Connection                         │  │       │
│  │   └────────────────────────────────────────────────┘  │       │
│  └──────────────────────────────────────────────────────┘       │
│         ↓ (1, 64, T)                                            │
│  ┌──────────────────────┐                                       │
│  │ Take Last Position   │  x[:, :, -1] → (64,)                  │
│  │ (Causal Pooling)     │  (has seen all past via dilations)    │
│  └──────────────────────┘                                       │
│         ↓ (64,)                                                 │
│  ┌──────────────────────┐                                       │
│  │ Output MLP           │  Linear(64→32)→ReLU→Linear(32→1)     │
│  │                      │  → Sigmoid                            │
│  └──────────────────────┘                                       │
│         ↓                                                        │
│  Output: progress scalar [0, 1]                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Causal Dilated Convolution Detail

```
Kernel size k=3, Dilation d=4:

Standard conv:     looks at positions [t-1, t, t+1]
Dilated conv:      looks at positions [t-4, t, t+4]  (NOT causal!)
Causal dilated:    looks at positions [t-8, t-4, t]  (only past)

Implementation:
  1. Pad LEFT by (k-1) × d = 2 × 4 = 8 zeros
  2. Apply conv (no padding in conv layer)
  3. Output length = input length (causality preserved)

┌─────────────────────────────────────────────────────────┐
│  Receptive Field Growth (Causal)                        │
│                                                          │
│  d=1:   [t-2, t-1, t]                    RF = 3         │
│  d=2:   [t-4, t-2, t]                    RF = 5         │
│  d=4:   [t-8, t-4, t]                    RF = 9         │
│  d=8:   [t-16, t-8, t]                   RF = 17        │
│  d=16:  [t-32, t-16, t]                  RF = 33        │
│  d=32:  [t-64, t-32, t]                  RF = 65        │
│                                                          │
│  Stacked: Effective RF = 127 frames                     │
└─────────────────────────────────────────────────────────┘
```

### Why Dilated Convs Avoid Position Contradiction

The model learns **completion patterns** at different temporal scales:
- Early in action: only local patterns filled, long-range convolutions see padding/zeros
- Late in action: all scales have rich activations from accumulated context

**No explicit position** → no contradictory mapping.

### Forward Pass Pseudocode

```python
def forward(self, segment_embeddings):
    # segment_embeddings: (T, 128)

    # 1. Project to hidden dim
    x = relu(self.input_proj(segment_embeddings))  # (T, 64)

    # 2. Transpose for Conv1d
    x = x.T.unsqueeze(0)  # (1, 64, T)

    # 3. Dilated residual blocks
    for dilation, block in zip([1,2,4,8,16,32], self.blocks):
        residual = x

        # Causal padding (left only)
        causal_pad = 2 * dilation  # (k-1) * d for k=3
        x_padded = F.pad(x, (causal_pad, 0))

        # Dilated conv + BatchNorm + ReLU
        out = relu(block.bn(block.conv(x_padded)))
        out = out[..., :x.shape[-1]]  # Trim to original length

        # 1x1 conv + dropout + residual
        out = block.dropout(block.conv1x1(out))
        x = residual + out

    # 4. Take last position (causal)
    final = x[0, :, -1]  # (64,)

    # 5. Output MLP
    return sigmoid(self.output_mlp(final))  # scalar [0,1]
```

### Hyperparameters

```python
DilatedConvProgressHead_CONFIG = {
    'input_dim': 128,
    'hidden_dim': 64,
    'kernel_size': 3,
    'dilations': [1, 2, 4, 8, 16, 32],
    'dropout': 0.1,
    'use_batch_norm': True,
}
# ~109K parameters
```

### Potential Issues & Mitigations

| Issue | Mitigation |
|-------|------------|
| Very short sequences | Padding handles gracefully |
| Large dilation instability | BatchNorm + residual connections |
| Gradient flow through 6 layers | Residual connections |

---

## Comparison Table

| Aspect | TransformerProgressHead | DilatedConvProgressHead |
|--------|------------------------|-------------------------|
| **Position handling** | ALiBi (relative distance) | Implicit via receptive fields |
| **Complexity** | O(T²) | O(T) |
| **Parameters** | ~76K | ~109K |
| **Interpretability** | Attention weights visible | Less interpretable |
| **Long sequences** | May need chunking for T>200 | Handles efficiently |
| **Similar to existing** | New pattern | Similar to ProTAS |
| **Training stability** | Pre-norm helps | BatchNorm + residual |

---

## Short Input Behavior (1-4 Frames)

Both architectures are designed to handle variable-length inputs, including very short sequences at the start of an action.

### The Challenge

For very short inputs (1-4 frames), both models face the same fundamental challenge:

**Target for frame 1 of an action:**
- 5-frame action: target = 1/5 = 0.20
- 50-frame action: target = 1/50 = 0.02

**What information is available?**
1. The **content** of the embeddings (what action is this?)
2. Very limited **temporal context** (not much has happened yet)

### TransformerProgressHead with Short Inputs

**1 frame:**
```
Attention matrix: [1×1] - just self-attention
                  ┌─────┐
                  │ 1.0 │  (softmax of single element = 1.0)
                  └─────┘
Output = single frame passed through FFN + MLP
```
Works mechanically, but limited information available.

**2 frames:**
```
Attention matrix: [2×2] with causal mask
Frame 0: [1.0,  -  ]     (can only see itself)
Frame 1: [0.3, 0.7]      (attends to both, prefers recent due to ALiBi)

Last token (frame 1) encodes: "I've seen 2 frames, frame 1 is most relevant"
```

**3-4 frames:** Similar pattern, attention becomes more meaningful with more context.

### DilatedConvProgressHead with Short Inputs

**1 frame:**
```
Input shape: (1, 64, 1)  - single time step

Block 1 (d=1): causal_pad=2, padded shape (1, 64, 3)
               Conv sees: [0, 0, frame_0] → outputs 1 value

Block 2 (d=2): causal_pad=4, padded shape (1, 64, 5)
               Conv sees: [0, 0, 0, 0, prev_output] → outputs 1 value

... (all blocks see mostly zeros + the single frame)

Final: Takes last position (the only position)
```
Works mechanically - padding ensures valid output.

**2-4 frames:**
```
Input shape: (1, 64, T) where T=2,3,4

Block 1 (d=1): Sees actual frame relationships
Block 2+ (d=2,4,8,16,32): Mostly sees padding for short sequences
```

### Summary: How Each Architecture Handles Short Sequences

| Frames | TransformerProgressHead | DilatedConvProgressHead |
|--------|------------------------|-------------------------|
| 1 | Works but essentially just MLP on single embedding | Works but essentially just MLP on single embedding |
| 2 | Minimal attention context, should work | Only d=1 block sees real data, others see padding |
| 3-4 | Attention starts to be meaningful | d=1,2 blocks see real data |
| 5+ | Full architecture utilized | More blocks see real data |

### Why This Is Acceptable

1. **Training uses `min_segment_len=3`**: Very short sequences are rare in training samples
2. **Boundary loss handles first frame**: Explicit supervision for T=1 case via boundary loss
3. **Online inference is cumulative**: The model sees frames 1, then 1-2, then 1-2-3... so short sequences only occur at the very beginning of each action
4. **Residual connections help**: Information flows directly through residuals even when conv layers see mostly padding

### Optional: Single-Frame Fallback

If particularly concerned about T=1 performance, consider a fallback:

```python
def forward(self, segment_embeddings):
    T = segment_embeddings.shape[0]

    if T == 1:
        # Fallback: simple MLP for single frame
        return self.single_frame_mlp(segment_embeddings[0])

    # Normal processing for T >= 2
    ...
```

This ensures the single-frame case uses a dedicated pathway optimized for that scenario.

---

## Files to Modify

### 1. `models/model_multiprong.py`

Add new classes:
- `TransformerProgressHead`
- `DilatedConvProgressHead`
- `CausalDilatedResidualBlock`
- `ALiBiAttention`

Modify `MultiProngAttDropoutModel.__init__`:
```python
if use_progress_head and progress_head_config is not None:
    arch = progress_head_config.get('architecture', 'gru')
    if arch == 'transformer':
        self.progress_head = TransformerProgressHead(...)
    elif arch == 'dilated_conv':
        self.progress_head = DilatedConvProgressHead(...)
    else:  # 'gru' (default)
        self.progress_head = ProgressHead(...)
```

### 2. `configs/generic_config.py`

Add to `CONFIG.PROGRESS_LOSS['learnable']`:
```python
'architecture': 'gru',  # 'gru', 'transformer', or 'dilated_conv'
'transformer_config': {
    'd_model': 64,
    'num_heads': 4,
    'num_layers': 2,
    'ffn_dim': 128,
    'dropout': 0.1,
},
'dilated_conv_config': {
    'hidden_dim': 64,
    'kernel_size': 3,
    'dilations': [1, 2, 4, 8, 16, 32],
    'dropout': 0.1,
},
```

### 3. `utils/train_util.py`

Update checkpoint loading to detect architecture type from saved weights.

---

## Implementation Steps

### Phase 1: TransformerProgressHead
1. Implement `ALiBiAttention` module
2. Implement `TransformerProgressHead` class
3. Add to model_multiprong.py
4. Test forward pass with dummy input

### Phase 2: DilatedConvProgressHead
1. Implement `CausalDilatedResidualBlock` (reference ProTAS)
2. Implement `DilatedConvProgressHead` class
3. Add to model_multiprong.py
4. Test forward pass with dummy input

### Phase 3: Config & Integration
1. Add architecture selection to config
2. Update `MultiProngAttDropoutModel` to use new heads
3. Update checkpoint save/load for architecture detection
4. Backward compatibility with existing GRU checkpoints

### Phase 4: Training & Evaluation
1. Train both architectures: `--progress_arch transformer` or `--progress_arch dilated_conv`
2. Compare OGPE metrics
3. Analyze progress curves with `extract_progress.py`

---

## Verification Commands

```bash
# Train with Transformer head
python multitask_train.py --version 5 --dataset egoprocel --loss_type GTCC \
    --progress_loss learnable --progress_lambda 1000000 \
    --progress_arch transformer

# Train with Dilated Conv head
python multitask_train.py --version 5 --dataset egoprocel --loss_type GTCC \
    --progress_loss learnable --progress_lambda 1000000 \
    --progress_arch dilated_conv

# Extract progress values
python extract_progress.py -f <output_folder> --max_videos 5

# Evaluate OGPE
python eval.py -f <output_folder> --level action
```

## Expected Improvements

1. Progress values start closer to 0 (not 0.5)
2. Progress values reach 1.0 at action end
3. Smoother monotonic increase through action
4. Lower OGPE (Online Grounded Progress Error)

---

## Implementation Complete

The following changes have been implemented and tested:

### Files Modified

| File | Changes |
|------|---------|
| `models/model_multiprong.py` | Added `create_progress_head()` factory, `TransformerProgressHead`, `TransformerBlockWithALiBi`, `DilatedConvProgressHead`, `CausalDilatedResidualBlock`. **Original `ProgressHead` class unchanged.** |
| `configs/generic_config.py` | Added `architecture` option, `transformer_config`, `dilated_conv_config` sections |
| `utils/parser_util.py` | Added `--progress_arch` argument (choices: `gru`, `transformer`, `dilated_conv`) |
| `configs/entry_config.py` | Passes `progress_arch` to config |
| `utils/train_util.py` | Auto-detects architecture type from checkpoint weights |

### Verification Results

| Test | Result |
|------|--------|
| GRU ProgressHead unchanged | ✓ |
| All architectures same interface `(T, 128) → scalar` | ✓ |
| Loss function identical for all architectures | ✓ |
| Default is GRU (backward compatible) | ✓ |
| Parser accepts `--progress_arch` | ✓ |
| Full model integration works | ✓ |
| Single frame (T=1) handling works | ✓ |

### Parameter Counts

| Architecture | Parameters |
|--------------|------------|
| GRU (default) | ~39K |
| Transformer (ALiBi) | ~77K |
| DilatedConv | ~110K |

### Usage Examples

```bash
# Default GRU (same as before, backward compatible)
python multitask_train.py 5 --GTCC --temporal_stacking --egoprocel \
    --progress_loss learnable --progress_lambda 1000000

# Transformer with ALiBi attention
python multitask_train.py 5 --GTCC --temporal_stacking --egoprocel \
    --progress_loss learnable --progress_lambda 1000000 \
    --progress_arch transformer

# Dilated Convolutions (similar to ProTAS)
python multitask_train.py 5 --GTCC --temporal_stacking --egoprocel \
    --progress_loss learnable --progress_lambda 1000000 \
    --progress_arch dilated_conv
```

### Key Design Decisions

1. **Factory Pattern**: `create_progress_head()` selects architecture based on config
2. **Same Interface**: All heads take `(T, 128)` input and return scalar `[0, 1]`
3. **Backward Compatible**: Default is `'gru'`, existing checkpoints load correctly
4. **Auto-Detection**: Checkpoint loading detects architecture from weight keys
5. **Loss Unchanged**: `loss_entry.py` works identically for all architectures
