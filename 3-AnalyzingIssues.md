# Next Experiment Plan: Fix Progress Head Early Saturation + Add Disambiguation

## Context

### The Problem
Every learnable progress head version (v1-v8) has failed to learn actual temporal progression:
- **v1**: Predicted near-constant ~0.5 (got OGPE 0.2583 by accident, not by learning)
- **v2**: Same, but worse OGPE (0.3034) due to lambda=1M killing gradients
- **v3**: Position encoding created contradictory signals (same pos → different targets)
- **v4**: Removed position encoding → stuck at 0.5 without temporal signal
- **v5-v7**: Transformer/DilatedConv + frame_count → mode collapse, step-functions [0.67, 1.0, 1.0...]
- **v8**: Dense supervision → erratic, no improvement

**128-d aligned features** failed because alignment makes all videos temporally identical → progress head learns ONE curve for everything.

**2048-d raw features** were tried (confirmed in `output_testing/`) but hit **early saturation**:
- **V6 (GRU + raw 2048-d + dense + progress-only)**: Rises to 0.73 in 3 frames → dead plateau with deltas=0.0 for hundreds of remaining frames. Predictions: `[0.16, 0.45, 0.70, 0.73, 0.73, 0.73, 0.73, ...]`
- **V7 (Transformer + raw 2048-d + dense + progress-only)**: Complete degeneration — jumps to 1.0 on frame 2, stays there forever. Predictions: `[0.23, 1.0, 1.0, 1.0, 1.0, ...]`

Root cause: GRU hidden_dim=64 overwhelmed by 32:1 compression ratio (2048→64). Transformer likely hit same issue differently.

### What 2-AnalyzingIssues.md Recommends (Section 7 + 8)

**Priority 1**: Fix early saturation (blocks everything else)
- Input projection (2048→128) + wider GRU (128) + replace sigmoid with clamped linear + per-frame running count

**Then combine**:
- **Alternative A**: Action-class conditioning (strongest disambiguation signal)
- **Alternative B**: Auxiliary length estimation head
- **Alternative C**: Rate-of-change features (cheap supplement)
- **Alternative D**: Better metrics for evaluation narrative

---

## Step 0: Git Housekeeping

1. Commit and push all current changes to `main`
2. Create new branch `progress_changes` for all subsequent work

---

## Step 1: Fix Early Saturation Architecture (Priority 1)

All changes in `models/model_multiprong.py` and `configs/generic_config.py`.

### 1a. Input Projection Layer (2048→128)
Add `nn.Linear(2048, 128)` before the GRU. Reduces compression ratio from 32:1 to 1:1. The GRU processes semantically compressed features, not raw high-dimensional noise.

### 1b. Wider GRU Hidden State (64→128)
Doubles the "notebook space" for tracking temporal evolution. Balanced with 128-d projected input.

### 1c. Replace Sigmoid with Clamped Linear
Remove `nn.Sigmoid()` from FC output. Apply `torch.clamp(output, 0, 1)` in forward instead. Removes the compression penalty that makes progress >0.80 disproportionately hard to achieve. Equal gradient flow at all output levels.

### 1d. Per-Frame Running Count (for GRU anti-saturation only)
Change frame_count from broadcast `log(1+T)/log(1+300)` (same for all frames) to per-frame `log(1+i)/log(1+300)` (unique per frame). This breaks GRU update gate saturation within a forward pass. Document honestly that this does NOT resolve length ambiguity.

**Files**:
- `models/model_multiprong.py` — ProgressHead class: add input_proj, change hidden_dim, replace sigmoid, modify frame_count
- `configs/generic_config.py` — add `input_projection: true`, `hidden_dim: 128`, `output_activation: 'clamp'`

---

## Step 2: Action-Class Conditioning (Alternative A — Strongest Signal)

The most impactful addition per the analysis. If the model knows "this is chop_vegetables", it can learn that chopping typically takes 15-40 frames → seeing 5 frames → "probably 12-33% through."

### Implementation
- Add `nn.Embedding(num_actions, 16)` to ProgressHead
- Concatenate 16-d action embedding to each frame's features
- During training: use GT action labels from `hdl_actions` in `dset_jsons/egoprocel.json`
- During eval: use GT action labels (OGPE evaluation already uses GT segmentation boundaries — using GT action class is defensible and consistent)

### Data Flow
- `dset_jsons/egoprocel.json` already has `hdl_actions` per video → action names per segment
- Need to build action-to-index mapping (one-time)
- Pass action index through `loss_entry.py` to `progress_head.forward(segment_emb, action_idx)`

**Files**:
- `models/model_multiprong.py` — add action embedding to ProgressHead
- `utils/loss_entry.py` — pass action label when calling progress_head
- `models/json_dataset.py` — ensure action labels available in batch
- `extract_progress.py` — pass action label during inference

---

## Step 3: Auxiliary Length Estimation Head (Alternative B)

Add second output neuron: predict both progress AND estimated total action length.

- `total_loss = progress_loss + 0.1 * MSE(pred_length, actual_length)`
- Progress output: clamped linear [0, 1]
- Length output: softplus → positive value
- Forces model to build "how long is this action?" internal representation

**Files**:
- `models/model_multiprong.py` — modify FC output from 1 → 2, split in forward
- `utils/loss_entry.py` — add length supervision term

---

## Step 4: Rate-of-Change Features (Alternative C — Cheap Supplement)

Concatenate frame-to-frame L2 distance as extra scalar input. Actions have characteristic velocity profiles (fast at transitions, slow in middle).

```python
diffs = torch.zeros(T, 1, device=device)
diffs[1:] = torch.norm(segment_emb[1:] - segment_emb[:-1], dim=1, keepdim=True)
features_to_concat.append(diffs)
```

**Files**: `models/model_multiprong.py` — 3 lines in ProgressHead.forward()

---

## Step 5: Comprehensive Evaluation Metrics (Alternative D — No Training)

Implement better metrics that reveal temporal understanding even if OGPE is marginal:
- Spearman rank correlation (constant 0.5 → rho=0, any temporal model → rho>>0)
- R-squared (constant 0.5 → R²=0)
- Monotonicity score
- Per-length-bin OGPE (1-5, 6-10, 11-20, 21-50, 50+ frames)
- Normalized OGPE (improvement over constant 0.5)

**Files**: New `eval_comprehensive_metrics.py`

---

## Execution Order

### Round 1: Architecture fixes only (Step 1)
1. **Git**: commit+push current changes to `main` → create `progress_changes` branch
2. **Implement Step 1** (all 4 sub-fixes together): input projection, wider GRU, clamped linear, per-frame running count
3. **Train**: Progress-only on frozen encoder with dense supervision + raw 2048-d features
4. **Evaluate**: Extract progress, check if saturation is resolved (gradual 0→1 curves, not plateau at 0.73)

```bash
CUDA_VISIBLE_DEVICES=X python multitask_train.py 1 --gtcc --egoprocel --resnet --mcn \
  --progress_loss learnable --progress_lambda 500000.0 \
  --train_progress_only --reinit_progress_head \
  --alignment_checkpoint output_learnable_progress_v7/multi-task-setting_val/V1___GTCC_egoprocel/ckpt/best_model.pt \
  --progress_lr 0.001 --progress_epochs 50 \
  --progress_loss_mode dense --progress_features raw
```

### Round 2: Add disambiguation (only if Round 1 fixes saturation but progress still not good)
5. **Step 2**: Action-class conditioning
6. **Step 4**: Rate-of-change features (trivial, do alongside)
7. **Re-train & evaluate**

### Round 3: Additional refinement (only if needed)
8. **Step 3**: Auxiliary length estimation head
9. **Step 5**: Comprehensive evaluation metrics

---

## Files Summary

| File | Changes |
|------|---------|
| `models/model_multiprong.py` | Input projection, wider GRU, clamped linear, per-frame count, action embedding, aux length output, rate-of-change |
| `configs/generic_config.py` | New config fields for all above |
| `configs/entry_config.py` | Propagate new flags |
| `utils/loss_entry.py` | Pass action label, add length loss |
| `utils/parser_util.py` | New CLI args |
| `models/json_dataset.py` | Action label data flow |
| `extract_progress.py` | Action label + new head at inference |
| New: `eval_comprehensive_metrics.py` | Spearman, R², monotonicity, per-bin |

## Verification

1. **Saturation fixed**: Progress predictions show gradual 0→1 curves (not plateau at 0.80 by frame 7)
2. **Video-specific**: Different videos produce DIFFERENT progress curves (not one universal curve)
3. **Action-class helps**: OGPE with action conditioning < OGPE without
4. **Metrics tell the story**: Spearman rho >> 0 for model vs rho=0 for constant 0.5
5. **Per-length-bin**: Model dominates on medium/long actions (6+ frames)
