# Fix ProgressHead ~0.5 First-Frame Prediction Issue

## Problem Summary
All learnable progress head models (V3 GRU, V4 Transformer, V5 DilatedConv) predict ~0.5 for early frames instead of starting near 0. This persists despite:
- Weighted loss (20x on early frames)
- Stratified sampling
- Boundary loss
- Learnable h0
- High lambda (1M)

## Critical Finding: Position Encoding is DISABLED

**Current config** (`configs/generic_config.py` line 78):
```python
'use_position_encoding': False,  # V4+: disabled by default
```

This means:
- **GRU**: Receives ONLY raw embeddings - no position, no frame count
- **Transformer**: Has ALiBi (relative attention) but no explicit frame count
- **DilatedConv**: Has implicit position via causal conv but no explicit frame count

**None of the architectures know the segment length T!**

## Root Cause
Without knowing T (segment length), the model cannot distinguish:
- T=1 (should predict ~0.01)
- T=3 (should predict ~0.03)
- T=100 (should predict ~1.0)

The GRU/Transformer/DilatedConv only see the embedding sequence. They have no information about "how many frames do I have?" which is the KEY signal for predicting progress.

With no distinguishing features, the sigmoid output defaults to ~0.5 (maximum entropy / safest prediction).

---

## EgoProceL Dataset Statistics (1fps)

Before choosing a normalization factor, here are the actual dataset statistics:

| Metric | Value |
|--------|-------|
| Mean action segment length | ~22 frames |
| Median action segment length | ~11 frames |
| Action segment range | 1 to 239 frames (typical) |
| Outliers | Up to 1500 frames |
| 70% of actions | Under 20 frames |
| Average video length | ~448 frames |
| Max video length | ~3300 frames |

**Note**: During online evaluation, T can grow from 1 to full video length.

---

## Important Q&A: Will Frame Count Actually Help?

### The Concern

For different action lengths, the same frame count T produces very different GT progress:
- **Short action (5 frames)**: At T=5, GT progress = 1.0 (action complete)
- **Long action (100 frames)**: At T=5, GT progress = 0.05 (just started)

So how can the model learn to predict correctly with just frame count?

### The Answer

**Frame count alone is NOT sufficient.** The model needs to learn:

```
progress = f(embeddings, T)
```

Where the **embeddings** carry information about:
1. **What action** is being performed (cutting tomato vs assembling tent)
2. **Visual appearance** (how the scene looks at this stage)
3. **Temporal patterns** (how embeddings evolve over time)

### What Frame Count Provides

**Without frame count:**
```
Input: [emb_1, emb_2, ..., emb_5]  # Just embeddings
Model thinks: "I see some embeddings... I guess 0.5?" (no reference point)
```

**With frame count:**
```
Input: [emb_1, ..., emb_5] + frame_count=0.31 (log scale for T=5)
Model can learn:
  - "These embeddings look like 'cut tomato' + T=5 → usually ~10 frames → predict ~0.5"
  - "These embeddings look like 'assemble tent' + T=5 → usually ~100 frames → predict ~0.05"
```

### The Key Insight

Frame count doesn't tell the model the GT progress directly. Instead, it gives the model a **reference point** to combine with embedding patterns.

The model must learn:
- "Short actions have certain embedding patterns"
- "Long actions have different embedding patterns"
- "Combined with T, I can estimate progress"

### Why Current Models Fail

Without frame count, the model has NO signal about "how many frames have I seen?" When it sees 1 frame vs 5 frames with similar initial embeddings, it can't distinguish them → outputs ~0.5 (safe default).

### Will It Work?

**It depends on how discriminative your embeddings are.**

If embeddings for "frame 5 of a 5-frame action" look different from "frame 5 of a 100-frame action" (which they should - the visual content differs), the model can learn this relationship.

**Sanity check needed**: Analyze if embeddings are discriminative for short vs long actions before implementing.

---

## Embedding Discriminability Analysis Results

**Experiment**: Compare embeddings at early frames (~20% into action) grouped by action length.

```
================================================================================
EMBEDDING DISCRIMINABILITY ANALYSIS
Question: Are embeddings at frame T different for short vs long actions?
================================================================================

GTCC:
  Samples: short=303, medium=315, long=74
  Inter-group dist: short-med=0.000, short-long=0.000, med-long=0.000
  Intra-group var:  short=0.001, medium=0.001, long=0.001
  RATIO (inter/intra): 0.363 (>1 = discriminative)

LAV:
  Samples: short=303, medium=315, long=74
  Inter-group dist: short-med=4.321, short-long=7.997, med-long=4.243
  Intra-group var:  short=23.306, medium=26.852, long=25.949
  RATIO (inter/intra): 0.218 (>1 = discriminative)

TCC:
  Samples: short=303, medium=315, long=74
  Inter-group dist: short-med=0.000, short-long=0.000, med-long=0.000
  Intra-group var:  short=0.000, medium=0.000, long=0.000
  RATIO (inter/intra): 0.225 (>1 = discriminative)

VAVA:
  Samples: short=303, medium=315, long=74
  Inter-group dist: short-med=0.000, short-long=0.000, med-long=0.000
  Intra-group var:  short=0.000, medium=0.000, long=0.000
  RATIO (inter/intra): 0.116 (>1 = discriminative)
================================================================================
```

### Critical Finding: Embeddings Are NOT Discriminative!

**ALL models have ratio < 1**, meaning:
- Inter-group distance (between short/medium/long actions) is **SMALLER** than intra-group variance
- The model **CANNOT** distinguish "frame 5 of a short action" from "frame 5 of a long action" based on embeddings alone
- Embeddings cluster by **action type** (cooking, assembly), not by **action duration**

### Implications

1. **Frame count is MANDATORY** - Without it, the model has no signal about action duration
2. **Embeddings alone cannot solve this** - The alignment loss optimizes for action similarity, not duration awareness
3. **The current ~0.5 prediction makes sense** - With no duration signal, the model defaults to maximum entropy (0.5)

### Why This Happens

The alignment losses (GTCC, TCC, LAV, VAVA) are designed to:
- Make same-action frames similar across videos
- Make different-action frames dissimilar

They are **NOT** designed to:
- Encode action duration information
- Make short vs long actions distinguishable

### Conclusion

**Adding frame count as an explicit input feature is not just helpful - it's NECESSARY.**

The model fundamentally cannot predict progress without knowing T because:
1. Embeddings don't encode duration
2. The same embedding patterns appear in both short and long actions
3. The only differentiating signal is T (frame count)

---

## Solution: Add Frame Count Feature + Bias Initialization

The key insight: **Frame count (T) is the most important missing feature**. The model needs to know "I have T frames" to predict that T frames represents T/action_length progress.

### Frame Count Normalization Options

Choose ONE of these normalization approaches:

#### Option A: Simple Division (Recommended for Simplicity)
```python
frame_count = T / 200.0  # Normalized by max expected action length
```
- **Pros**: Simple, interpretable
- **Cons**: Values can exceed 1.0 for very long sequences
- **When to use**: If most actions are under 200 frames (true for EgoProceL)

#### Option B: Log Scale (Recommended for Robustness)
```python
import math
frame_count = math.log1p(T) / math.log1p(300)  # log(1+T) / log(1+max_T)
```
- **Pros**: Compresses large values, robust to outliers, smooth gradient
- **Cons**: Slightly less interpretable
- **When to use**: Variable action lengths, want values always in [0, ~1.2]

Example values for log scale with max_T=300:
| T (frames) | frame_count |
|------------|-------------|
| 1 | 0.12 |
| 5 | 0.31 |
| 10 | 0.42 |
| 20 | 0.53 |
| 50 | 0.69 |
| 100 | 0.81 |
| 200 | 0.93 |
| 300 | 1.00 |

#### Option C: Clamped Division
```python
frame_count = min(T / 100.0, 2.0)  # Capped at 2.0
```
- **Pros**: Bounded output, simple
- **Cons**: Loses information for very long sequences
- **When to use**: If you want strict bounds

#### Option D: Adaptive (Dataset-Aware)
```python
# During training, track max_T seen; use that for normalization
frame_count = T / self.max_T_observed  # Learned/tracked
```
- **Pros**: Adapts to actual data
- **Cons**: More complex, needs state tracking

### Recommendation
**Use Option B (Log Scale)** with `max_T=300` for EgoProceL:
- Handles the 1-239 frame range well
- Robust to outliers (1500 frame actions)
- Values stay in reasonable [0, ~1.2] range
- Smooth gradients for learning

---

## Implementation

### Changes Required

#### 1. Modify All ProgressHead Architectures
**File**: `/vision/anishn/GTCC_CVPR2024/models/model_multiprong.py`

**For GRU ProgressHead** (~line 202):
```python
import math

class ProgressHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, use_gru=True,
                 use_position_encoding=False, use_frame_count=True,
                 frame_count_max=300.0):  # NEW params
        super(ProgressHead, self).__init__()
        self.use_gru = use_gru
        self.input_dim = input_dim
        self.use_position_encoding = use_position_encoding
        self.use_frame_count = use_frame_count
        self.frame_count_max = frame_count_max

        # Calculate input dim: +1 for position (if enabled), +1 for frame count (if enabled)
        extra_dims = 0
        if use_position_encoding:
            extra_dims += 1
        if use_frame_count:
            extra_dims += 1
        gru_input_dim = input_dim + extra_dims

        if use_gru:
            self.gru = nn.GRU(gru_input_dim, hidden_dim, num_layers=1,
                              batch_first=True, bidirectional=False)
            if use_position_encoding:
                self.h0 = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            # NEW: Initialize final bias toward low values
            with torch.no_grad():
                self.fc[-2].bias.fill_(-2.0)  # sigmoid(-2) ~ 0.12
        else:
            self.fc = nn.Sequential(
                nn.Linear(gru_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
            with torch.no_grad():
                self.fc[-2].bias.fill_(-2.0)

    def forward(self, segment_embeddings):
        T = segment_embeddings.shape[0]
        device = segment_embeddings.device

        features_to_concat = [segment_embeddings]

        if self.use_position_encoding:
            # Relative position within segment (0 to ~1)
            positions = torch.arange(T, device=device, dtype=torch.float32) / max(T, 1)
            positions = positions.unsqueeze(1)  # (T, 1)
            features_to_concat.append(positions)

        if self.use_frame_count:
            # KEY FEATURE: Log-normalized frame count (same value for all frames)
            # Option B: Log scale (recommended)
            fc_value = math.log1p(T) / math.log1p(self.frame_count_max)
            frame_count = torch.full((T, 1), fc_value, device=device, dtype=torch.float32)
            features_to_concat.append(frame_count)

        x = torch.cat(features_to_concat, dim=1)  # (T, D + extras)

        if self.use_gru:
            x = x.unsqueeze(0)  # (1, T, D+extras)
            if self.use_position_encoding:
                _, h_n = self.gru(x, self.h0)
            else:
                _, h_n = self.gru(x)
            progress = self.fc(h_n.squeeze())
        else:
            x = x.mean(dim=0)
            progress = self.fc(x)

        return progress.squeeze()
```

**Why frame_count is critical:**
- T=1 → frame_count=0.12 → "I have 1 frame, predict very low"
- T=10 → frame_count=0.42 → "I have 10 frames, predict ~10/action_len"
- T=100 → frame_count=0.81 → "I have 100 frames, predict higher"
- T=300 → frame_count=1.0 → "I have full action, predict ~1.0"

**For TransformerProgressHead** (~line 277):

**Changes:**
1. Add frame count feature
2. **Upgrade to PyTorch 2.0 Flash Attention** (faster, memory efficient)
3. Bias initialization

```python
class TransformerProgressHead(nn.Module):
    def __init__(self, input_dim=128, d_model=64, num_heads=4, num_layers=2,
                 ffn_dim=128, dropout=0.1, use_frame_count=True, frame_count_max=300.0):
        super(TransformerProgressHead, self).__init__()
        self.use_frame_count = use_frame_count
        self.frame_count_max = frame_count_max
        self.dropout_p = dropout

        proj_input_dim = input_dim + (1 if use_frame_count else 0)

        self.input_proj = nn.Sequential(
            nn.Linear(proj_input_dim, d_model),
            nn.LayerNorm(d_model),
        )
        # ... Q, K, V projections unchanged ...

        # Initialize output bias
        with torch.no_grad():
            self.output_mlp[-2].bias.fill_(-2.0)

    def forward(self, segment_embeddings):
        T = segment_embeddings.shape[0]
        device = segment_embeddings.device

        if self.use_frame_count:
            fc_value = math.log1p(T) / math.log1p(self.frame_count_max)
            frame_count = torch.full((T, 1), fc_value, device=device, dtype=torch.float32)
            segment_embeddings = torch.cat([segment_embeddings, frame_count], dim=1)

        # Input projection
        x = self.input_proj(segment_embeddings).unsqueeze(0)  # (1, T, d_model)

        # Apply transformer layers with Flash Attention
        for layer in self.layers:
            x = layer(x)

        # Take last token
        return self.output_mlp(x[0, -1, :]).squeeze()
```

**Flash Attention upgrade for TransformerBlockWithALiBi:**
```python
class TransformerBlockWithALiBi(nn.Module):
    def forward(self, x):
        B, T, D = x.shape

        # Pre-norm + Q, K, V
        x_norm = self.norm1(x)
        Q = self.q_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x_norm).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # PyTorch 2.0 Flash Attention with native causal masking
        # NOTE: ALiBi bias requires custom handling (see below)
        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            is_causal=True,  # Native causal masking - replaces manual mask
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.out_proj(attn_out)
        x = x + self.ffn(self.norm2(x))
        return x
```

**Flash Attention Benefits:**
- ~2-4x faster training
- O(T) memory instead of O(T²)
- Native causal masking (no need for manual mask creation)
- PyTorch 2.0.1 already available in your env

**Note on ALiBi:** PyTorch's `scaled_dot_product_attention` doesn't directly support ALiBi bias with Flash kernel. Options:
1. Drop ALiBi (simpler, still works with causal mask)
2. Use `attn_mask` parameter (falls back to non-Flash path)
3. Keep frame_count feature as primary position signal (recommended)

---

**For DilatedConvProgressHead** (~line 471):
```python
class DilatedConvProgressHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, kernel_size=3,
                 dilations=None, dropout=0.1, use_frame_count=True, frame_count_max=300.0):
        super(DilatedConvProgressHead, self).__init__()
        self.use_frame_count = use_frame_count
        self.frame_count_max = frame_count_max

        proj_input_dim = input_dim + (1 if use_frame_count else 0)

        self.input_proj = nn.Sequential(
            nn.Linear(proj_input_dim, hidden_dim),  # Changed from input_dim
            nn.ReLU(),
        )
        # ... rest of init unchanged ...

        # Initialize output bias
        with torch.no_grad():
            self.output_mlp[-2].bias.fill_(-2.0)

    def forward(self, segment_embeddings):
        T = segment_embeddings.shape[0]
        device = segment_embeddings.device

        if self.use_frame_count:
            fc_value = math.log1p(T) / math.log1p(self.frame_count_max)
            frame_count = torch.full((T, 1), fc_value, device=device, dtype=torch.float32)
            segment_embeddings = torch.cat([segment_embeddings, frame_count], dim=1)

        # ... rest of forward unchanged (causal conv provides implicit position) ...
```

#### 2. Update Factory Function
**File**: `/vision/anishn/GTCC_CVPR2024/models/model_multiprong.py` (~line 150)

```python
def create_progress_head(input_dim, config):
    """Factory function to create the appropriate ProgressHead based on config."""
    architecture = config.get('architecture', 'gru')
    use_frame_count = config.get('use_frame_count', True)  # NEW: default True
    frame_count_max = config.get('frame_count_max', 300.0)  # NEW

    if architecture == 'transformer':
        transformer_config = config.get('transformer_config', {})
        return TransformerProgressHead(
            input_dim=input_dim,
            d_model=transformer_config.get('d_model', 64),
            num_heads=transformer_config.get('num_heads', 4),
            num_layers=transformer_config.get('num_layers', 2),
            ffn_dim=transformer_config.get('ffn_dim', 128),
            dropout=transformer_config.get('dropout', 0.1),
            use_frame_count=use_frame_count,  # NEW
            frame_count_max=frame_count_max,  # NEW
        )
    elif architecture == 'dilated_conv':
        dilated_config = config.get('dilated_conv_config', {})
        return DilatedConvProgressHead(
            input_dim=input_dim,
            hidden_dim=dilated_config.get('hidden_dim', 64),
            kernel_size=dilated_config.get('kernel_size', 3),
            dilations=dilated_config.get('dilations', [1, 2, 4, 8, 16, 32]),
            dropout=dilated_config.get('dropout', 0.1),
            use_frame_count=use_frame_count,  # NEW
            frame_count_max=frame_count_max,  # NEW
        )
    else:
        # Default: GRU-based ProgressHead
        return ProgressHead(
            input_dim=input_dim,
            hidden_dim=config.get('hidden_dim', 64),
            use_gru=config.get('use_gru', True),
            use_position_encoding=config.get('use_position_encoding', False),
            use_frame_count=use_frame_count,  # NEW
            frame_count_max=frame_count_max,  # NEW
        )
```

#### 3. Add Config Options
**File**: `/vision/anishn/GTCC_CVPR2024/configs/generic_config.py`

```python
'learnable': {
    # ... existing options ...
    'use_frame_count': True,  # NEW: Add frame count as input feature
    'frame_count_max': 300.0,  # NEW: Max T for log normalization (based on EgoProceL stats)
    'frame_count_method': 'log',  # NEW: 'log', 'linear', or 'clamped'
}
```

---

## Implementation Order

1. **Phase 1: Quick Test** - Just add bias initialization (-2.0) to existing code
   - Modify `model_multiprong.py` to init bias in `__init__`
   - Load existing checkpoint, fine-tune briefly
   - Test if predictions shift lower
   - **Expected**: Predictions shift from ~0.5 to ~0.3-0.4 (partial fix)

2. **Phase 2: Frame Count Feature** - Full architecture change
   - Add `use_frame_count` and `frame_count_max` parameters
   - Implement log-scale frame count in all three architectures
   - Update factory function and config
   - Retrain from scratch
   - **Expected**: Predictions start near 0 and increase to ~1

3. **Phase 3 (Optional): Raw Features** - If still needed
   - Modify training to include raw frame features
   - Increase input dimension accordingly
   - **Expected**: Further improvement in edge cases

---

## Files to Modify

| File | Changes |
|------|---------|
| `models/model_multiprong.py` | Add `use_frame_count`, `frame_count_max` params; implement log-scale frame count; bias init |
| `configs/generic_config.py` | Add `use_frame_count`, `frame_count_max`, `frame_count_method` options |
| (Optional) `utils/loss_entry.py` | Support raw features during training |

---

## Verification

1. After modification, run quick training (5-10 epochs)
2. Visualize progress predictions on sample videos:
   ```bash
   python extract_progress.py --exp_folder <path> --visualize
   ```
3. Check that first frame predictions are < 0.2 (not ~0.5)
4. Run full evaluation:
   ```bash
   python eval_protas_action_level.py --exp_folder <path>
   ```
5. Compare OGPE before vs after

---

## Summary: What's Missing vs What You Have

| Feature | Current Status | Impact |
|---------|---------------|--------|
| Weighted loss (20x early) | Enabled | Helps, but can't fix missing input |
| Stratified sampling | Enabled | Helps, but can't fix missing input |
| Boundary loss | Enabled | Helps, but 1-frame input has no info |
| Learnable h0 | Enabled (GRU) | Minor impact |
| Position encoding | **DISABLED** | No relative position info |
| Frame count | **MISSING** | **ROOT CAUSE** - model doesn't know T |
| Bias initialization | Default (~0.5) | Makes problem worse |

**The fix**: Add frame count feature (log-scale) + bias init (-2.0) → Model knows T → Can predict correctly

---

## Quick Test (Before Full Retrain)

You can test the bias initialization alone with existing checkpoint:
```python
# In evaluation script, after loading model:
if hasattr(model, 'progress_head'):
    # For GRU/simple heads
    if hasattr(model.progress_head, 'fc'):
        with torch.no_grad():
            model.progress_head.fc[-2].bias.fill_(-2.0)
    # For Transformer
    elif hasattr(model.progress_head, 'output_mlp'):
        with torch.no_grad():
            model.progress_head.output_mlp[-2].bias.fill_(-2.0)
```
This won't fully fix the issue (still missing frame count) but will show if bias helps shift predictions lower.

---

## Why This Should Work

1. **Frame count = T** explicitly tells the model "you have T frames"
2. **Log scale** ensures:
   - T=1 → 0.12 (low, model learns "very early")
   - T=100 → 0.81 (high, model learns "late in action")
   - T=300 → 1.0 (full, model learns "complete")
3. **Bias init = -2.0** starts predictions at sigmoid(-2) ≈ 0.12 instead of 0.5
4. Combined with existing improvements (weighted loss, boundary loss), should produce 0→1 range

---

## Alternative: Position Encoding Instead of Frame Count

If you want to try enabling position encoding first (simpler change):

```python
# In config:
'use_position_encoding': True,  # Change from False
```

This adds relative position (0 to 1) but **still doesn't tell the model T**.

- Position encoding tells: "You're at position 0.5 within current segment"
- Frame count tells: "You have T=50 frames total"

**Frame count is more informative** because it directly relates to progress.
