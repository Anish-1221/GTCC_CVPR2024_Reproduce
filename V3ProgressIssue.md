# V3 Progress Head Issue: Contradictory Training Signals

## Problem Summary

The V3 ProgressHead outputs values stuck around 0.5-0.6, failing to predict progress correctly at early frames (outputs ~0.53 when it should be ~0.03) or late frames (outputs ~0.6 when it should be ~1.0).

## Root Cause: Position Encoding Contradiction

The position encoding gives "position relative to what you've seen" but the target is "progress relative to total action length." These are fundamentally different things.

---

## Step 1: What happens in ProgressHead forward pass

```python
def forward(self, segment_embeddings):
    T = segment_embeddings.shape[0]  # e.g., 5 frames

    # Create position encoding
    positions = torch.arange(T) / T
    # For T=5: [0.0, 0.2, 0.4, 0.6, 0.8]

    # Concatenate to each frame's embedding
    # Frame 0: [emb_0, 0.0]
    # Frame 1: [emb_1, 0.2]
    # Frame 2: [emb_2, 0.4]
    # Frame 3: [emb_3, 0.6]
    # Frame 4: [emb_4, 0.8]  ← Last frame position is always ~1.0

    x = torch.cat([segment_embeddings, positions], dim=1)

    # Feed through GRU sequentially
    _, h_n = self.gru(x, self.h0)

    # Final hidden state → FC → Sigmoid → output
    progress = self.fc(h_n)  # Single value between 0-1
```

## Step 2: How GRU processes this

```
Time →

Frame 0: [emb_0, 0.0] → GRU → h1
Frame 1: [emb_1, 0.2] → GRU(h1) → h2
Frame 2: [emb_2, 0.4] → GRU(h2) → h3
Frame 3: [emb_3, 0.6] → GRU(h3) → h4
Frame 4: [emb_4, 0.8] → GRU(h4) → h5 ← Final hidden state
                                   ↓
                              FC + Sigmoid
                                   ↓
                            Output: 0.53
```

The GRU sees each frame with its position, updates its hidden state, and the FINAL hidden state is used to predict progress.

## Step 3: Training example - TWO different scenarios

### Scenario A: Short action (5 frames total)

```
Action: "pick up egg" (frames 0-4, total length = 5)
Training sample: frames 0-4 (all 5 frames)

ProgressHead receives:
- Frame 0: [emb_0, 0.0]
- Frame 1: [emb_1, 0.2]
- Frame 2: [emb_2, 0.4]
- Frame 3: [emb_3, 0.6]
- Frame 4: [emb_4, 0.8]  ← position 0.8

Ground truth: 5/5 = 1.0 (we've seen the whole action)

Model learns: "position 0.8 at last frame → output 1.0"
```

### Scenario B: Long action (45 frames total)

```
Action: "mix brownie batter" (frames 0-44, total length = 45)
Training sample: frames 0-4 (first 5 frames only)

ProgressHead receives:
- Frame 0: [emb_0, 0.0]
- Frame 1: [emb_1, 0.2]
- Frame 2: [emb_2, 0.4]
- Frame 3: [emb_3, 0.6]
- Frame 4: [emb_4, 0.8]  ← SAME position 0.8!

Ground truth: 5/45 = 0.11 (we've seen only 11% of action)

Model learns: "position 0.8 at last frame → output 0.11"
```

## The Contradiction

| Scenario | Position at last frame | Target |
|----------|----------------------|--------|
| A (short action) | 0.8 | 1.0 |
| B (long action) | 0.8 | 0.11 |

**Same input (position 0.8) → Different targets (1.0 vs 0.11)**

The model is being told:
- "When you see position 0.8, output 1.0"
- "When you see position 0.8, output 0.11"

## What the model does

It can't satisfy both, so it learns a compromise:
- "Position 0.8 → output ~0.5" (average of contradictory signals)

This is why you see outputs stuck around 0.5-0.6!

## The only difference between scenarios

The **embeddings** (emb_0, emb_1, etc.) are different:
- Scenario A: embeddings of "pick up egg"
- Scenario B: embeddings of "mix brownie batter"

So the model COULD learn:
- "Position 0.8 + egg embeddings → 1.0"
- "Position 0.8 + brownie embeddings → 0.11"

But this requires the model to:
1. Recognize the action type from embeddings
2. Know typical duration for that action
3. Combine this with position to estimate progress

This is **much harder** than a simple position → progress mapping.

## Summary

The position encoding tells the model "you're 80% through WHAT YOU'VE SEEN" but the target is "you're X% through THE TOTAL ACTION." These are fundamentally different things, causing contradictory training signals.

---

## Possible Solutions

### Option 1: Remove position encoding
Let the model rely purely on embeddings to infer progress. The embeddings might contain action-specific cues about "how done" the action looks.

### Option 2: Provide total action length during training AND inference
Change the position encoding to use total action length:
```python
positions = torch.arange(T) / total_action_length
```
But this requires knowing action length at inference time (not truly online).

### Option 3: Different task formulation
Instead of predicting absolute progress, predict something else:
- "Time remaining" estimation
- "Action phase" classification (early/mid/late)
- Relative progress cues

### Option 4: Action-conditioned progress
1. First classify the action type
2. Look up typical duration for that action
3. Use this to inform progress prediction

---

## Lessons from ProTAS Architecture

### How ProTAS Predicts Progress Differently

#### 1. Per-Class Progress (Not Scalar)

**ProTAS outputs:** `[batch, num_classes, T]` - progress value for EACH action class at EACH frame

**GTCC outputs:** Single scalar (0-1) for the whole segment

```python
# ProTAS
progress_out = self.conv_app(out)  # Shape: [B, num_classes, T]

# GTCC
progress = self.fc(h_n)  # Shape: scalar
```

#### 2. No Position Encoding - Trusts GRU Hidden State

ProTAS does **NOT** use explicit position encoding. It trusts the GRU's hidden state to learn temporal context through:
- Dilated convolutions (receptive field grows exponentially)
- GRU hidden state accumulation
- Multi-stage refinement

#### 3. Ground Truth Progress Labels (Supervised)

ProTAS generates progress labels from GT segmentation:

```python
# For each action segment
progress = (np.arange(segment_length) + 1) / segment_length
# e.g., 45-frame action: [1/45, 2/45, ..., 45/45]
```

**Key:** Progress is normalized **per-segment**, not per-video.

#### 4. Per-Class Progress Heads

ProTAS predicts progress for ALL action classes simultaneously:
- Doesn't need to know which action is happening at inference
- Action classification head tells you which action's progress to use

---

## Dilated Convolutions Explained

**Normal convolution:** Looks at adjacent frames (e.g., frames 0,1,2)

**Dilated convolution:** Skips frames based on dilation rate
- Dilation=1: frames 0,1,2 (normal)
- Dilation=2: frames 0,2,4
- Dilation=4: frames 0,4,8

```
Dilation=1:  [x] [x] [x] [ ] [ ] [ ] [ ] [ ]
Dilation=2:  [x] [ ] [x] [ ] [x] [ ] [ ] [ ]
Dilation=4:  [x] [ ] [ ] [ ] [x] [ ] [ ] [ ] [x]
```

**ProTAS stacks them exponentially:** dilation = 2^0, 2^1, 2^2, ..., 2^9

This creates a **hierarchical receptive field**:
- Layer 1 (d=1): sees 3 frames
- Layer 2 (d=2): sees 7 frames
- Layer 3 (d=4): sees 15 frames
- ...
- Layer 10 (d=512): sees 1000+ frames

**Why dilated convolutions help:**
1. **Captures long-range dependencies** without huge parameter count
2. **Multi-scale temporal patterns** - short actions AND long actions
3. **Efficient** - same compute as regular conv but much larger receptive field
4. **No position encoding needed** - the network learns temporal patterns from the structure itself

### Would Dilated Convolutions Help GTCC?

**Potentially yes**, but there's a key difference:
- **ProTAS:** Works on raw features, needs to build temporal understanding
- **GTCC:** Already has aligned embeddings from TCC/GTCC loss

The alignment losses already encourage temporal consistency. But dilated convolutions before the ProgressHead could help capture "how the action is evolving" over longer time spans.

---

## Recommended Changes for GTCC ProgressHead

### Change 1: Remove Position Encoding
Position encoding gives contradictory signals (same position maps to different progress values). Remove it entirely and trust the GRU to learn from embeddings.

### Change 2: Trust Embeddings + GRU Hidden State
The embeddings from TCC/GTCC/LAV/VAVA losses should contain information about action progression. Let the GRU learn to extract this without explicit position hints.

### Change 3 (Optional): Add Dilated Conv Layer Before GRU
```python
self.dilated_conv = nn.Conv1d(input_dim, input_dim, kernel_size=3,
                               dilation=4, padding=4)  # Sees ~9 frames
```
This could help capture longer-range temporal patterns.

### Change 4 (Optional): Per-Class Progress
Instead of predicting a single scalar, predict progress for each action class. Requires action labels during training but provides richer supervision.

---

## Changes Implemented (V4)

The following changes were implemented to address the issues above:

### 1. Removed Position Encoding as Default

**File:** `models/model_multiprong.py`

```python
# Before
def __init__(self, ..., use_position_encoding=True):

# After (V4)
def __init__(self, ..., use_position_encoding=False):
```

**Rationale:** Position encoding was giving contradictory signals - same position (0.8) mapped to different targets (0.11 vs 1.0) depending on action length.

### 2. Added Boundary Loss

**File:** `utils/loss_entry.py`

Explicitly supervises first and last frames of each action:

```python
# First frame: predict with just 1 frame, target ≈ 1/action_len
first_emb = vid_emb[start:start+1]
pred_first = progress_head(first_emb)
gt_first = 1.0 / action_len
boundary_loss += boundary_weight * torch.abs(pred_first - gt_first)

# Last frame: predict with full segment, target = 1.0
full_emb = vid_emb[start:end+1]
pred_last = progress_head(full_emb)
gt_last = 1.0
boundary_loss += boundary_weight * torch.abs(pred_last - gt_last)
```

**Rationale:** Forces the model to learn that:
- First frame of any action → progress ≈ 0
- Last frame of any action → progress = 1.0

### 3. Increased Weight Cap

**File:** `utils/loss_entry.py` and `configs/generic_config.py`

```python
# Before
weight_cap = 10.0

# After (V4)
weight_cap = 20.0
```

**Rationale:** Early frames now get up to 20x weight instead of 10x, further emphasizing correct early-frame predictions.

### 4. New Config Options

**File:** `configs/generic_config.py`

```python
CONFIG.PROGRESS_LOSS = edict({
    'enabled': False,
    'method': 'cumulative_l2',      # 'cumulative_l2' or 'learnable'
    'lambda_fixed': 0.1,
    'learnable': {
        'hidden_dim': 64,
        'use_gru': True,
        'min_segment_len': 3,
        'samples_per_video': 10,
        'frames_per_segment': 5,
        'stratified_sampling': True,
        'weighted_loss': True,
        'weight_cap': 20.0,          # Increased from 10
        'boundary_loss': True,       # NEW: explicit boundary supervision
        'boundary_weight': 5.0,      # NEW: weight for boundary loss
    },
})
```

### 5. Backward Compatibility

**File:** `utils/train_util.py`

Checkpoint loading auto-detects position encoding from GRU weight dimensions:
- GRU input dim = 128 → no position encoding (v2, v4+)
- GRU input dim = 129 → has position encoding (v3)

---

## Training V4 Models

```bash
python multitask_train.py --version 4 --dataset egoprocel --loss_type GTCC \
    --progress_loss learnable --progress_lambda 1000000
```

## Expected Improvements

1. **No contradictory signals** - model learns from embeddings only
2. **Better early-frame predictions** - boundary loss explicitly supervises first frames
3. **Better late-frame predictions** - boundary loss explicitly supervises last frames
4. **Stronger early-frame weighting** - 20x weight cap instead of 10x
