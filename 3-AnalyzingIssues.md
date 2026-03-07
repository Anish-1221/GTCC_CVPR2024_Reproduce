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

## Step 0: Git Housekeeping -- DONE

1. Commit and push all current changes to `main`
2. Create new branch `progress_changes` for all subsequent work

---

## Step 1: Fix Early Saturation Architecture (Priority 1) -- V9 CHANGES -- DONE

All changes in `models/model_multiprong.py`, `configs/generic_config.py`, `configs/entry_config.py`, `utils/parser_util.py`.

### 1a. Input Projection Layer (2048→128) -- DONE
Add `nn.Linear(2048, 128)` before the GRU. Reduces compression ratio from 32:1 to 1:1. The GRU processes semantically compressed features, not raw high-dimensional noise.
CLI flag: `--use_input_projection --projection_dim 128`

### 1b. Wider GRU Hidden State (64→128) -- DONE
Doubles the "notebook space" for tracking temporal evolution. Balanced with 128-d projected input.
CLI flag: `--progress_hidden_dim 128`

### 1c. Replace Sigmoid with Clamped Linear -- DONE
Remove `nn.Sigmoid()` from FC output. Apply `torch.clamp(output, 0, 1)` in forward instead. Removes the compression penalty that makes progress >0.80 disproportionately hard to achieve. Equal gradient flow at all output levels.
CLI flag: `--output_activation clamp`

### 1d. Per-Frame Running Count (for GRU anti-saturation only) -- DONE
Change frame_count from broadcast `log(1+T)/log(1+300)` (same for all frames) to per-frame `log(1+i)/log(1+300)` (unique per frame). This breaks GRU update gate saturation within a forward pass. Document honestly that this does NOT resolve length ambiguity.
CLI flag: `--per_frame_count`

### 1e. Skip Alignment Checkpoint for Raw Features -- DONE
When `--progress_features raw`, the 2048-d features come from disk and the encoder is not in the progress forward path. Removed the requirement for `--alignment_checkpoint` in this case.

**Files modified**:
- `models/model_multiprong.py` — ProgressHead class: input_proj, hidden_dim, output_activation, per_frame_count; create_progress_head() factory updated
- `configs/generic_config.py` — new defaults: use_input_projection, projection_dim, output_activation, per_frame_count
- `configs/entry_config.py` — propagate new V9 flags to CONFIG
- `utils/parser_util.py` — new CLI args: --use_input_projection, --projection_dim, --progress_hidden_dim, --output_activation, --per_frame_count
- `multitask_train.py` — skip checkpoint requirement when --progress_features raw
- `models/alignment_training_loop.py` — improved error logging (show full error message)
- `extract_progress.py` — added V9 config keys (use_input_projection, projection_dim, output_activation, per_frame_count) to load_model_for_extraction() GRU branch so checkpoint loading reconstructs correct architecture

**Commits**:
- `048b71e` — V9 anti-saturation architecture: input projection, wider GRU, clamped linear, per-frame count
- `dfb3076` — Skip alignment checkpoint requirement when using raw features

---

## Step 2: Action-Class Conditioning (Alternative A — Strongest Signal) -- V10 CHANGES -- DONE

### Why Action-Class Conditioning?

The core problem V9 still has — even after fixing saturation — is **disambiguation**. The progress head sees a sequence of visual features and must predict "how far through this action are we?" But without knowing *which* action is happening, the model cannot answer this question:

**The ambiguity problem:** Consider two actions with identical visual features at frame 5:
- "crack_egg" (typically 8 frames total) → frame 5 = 62.5% done
- "stir_pot" (typically 120 frames total) → frame 5 = 4.2% done

Without action identity, the GRU can only learn an **average** progress curve across all action types. This average curve is dominated by the action length distribution in the training data, which is why:
1. **Early jump to ~0.7**: The model learns that most actions are short (median ~10 frames), so seeing 3+ frames means "probably almost done" on average
2. **Ceiling below 1.0**: The model hedges — it can't confidently predict 1.0 because some actions with similar features at frame N could have 3x more frames remaining
3. **Same curve for everything**: V6 and V9 both produce nearly identical curves for all segments regardless of action type or length

**How action conditioning fixes this:** By telling the model "this is action class 42 (e.g., chop_vegetables)", the embedding gives the GRU a **lookup key** for action-specific duration statistics learned during training:
- "Action 42 typically takes 15-40 frames" → seeing 5 frames → "~25% through" (not 70%)
- "Action 7 typically takes 5-8 frames" → seeing 5 frames → "~75% through"
- The model can now learn separate progress curves per action class, with proper 0→1 ramps

**Why GT labels are defensible:** OGPE evaluation already uses GT segmentation boundaries (the model doesn't detect action boundaries, it receives them). Using GT action labels is consistent — we're measuring progress *within* known segments, not detecting actions. This matches the ProTAS/GTCC evaluation protocol.

### Implementation -- DONE
- Added `nn.Embedding(num_actions=116, action_embed_dim=16)` to ProgressHead
- Concatenate 16-d action embedding to each frame's features before GRU
- During training: action labels extracted from `vid_times['step']` in loss functions, converted to int via `_action_label_to_idx()`
- During eval/extraction: action labels from `times_dict['step']`, passed as `action_idx` to progress_head
- Backward compatible: `action_idx=None` falls back to index 0 (unknown/background)
- CLI flags: `--use_action_conditioning`, `--action_embed_dim 16`

### Data Flow -- DONE
- Action labels already present in `times_dict['step']` (no dataset changes needed)
- `_action_label_to_idx(step)` helper in `loss_entry.py`: converts string labels to int, guards against "SIL"/"background"
- All 5 loss modes updated to pass `action_idx` to `progress_head()` calls
- `sample_action_segment_with_multiple_frames()` extended with `return_action_name=True` option (backward compatible)
- `extract_progress.py` passes `action_idx` during frame-by-frame inference

**Files modified**:
- `models/model_multiprong.py` — ProgressHead: action_embedding, rate-of-change; create_progress_head() factory updated
- `utils/loss_entry.py` — `_action_label_to_idx()` helper, all 5 loss modes pass action_idx
- `utils/tensorops.py` — `sample_action_segment_with_multiple_frames()` optional `return_action_name` param
- `utils/parser_util.py` — new CLI args: --use_action_conditioning, --action_embed_dim, --use_rate_of_change
- `configs/generic_config.py` — new defaults for V10 flags
- `configs/entry_config.py` — propagate V10 flags to CONFIG
- `extract_progress.py` — pass action_idx during inference, V10 config keys in load_model_for_extraction()
- `utils/train_util.py` — GRU checkpoint loading reads V9+V10 config from stored checkpoint

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

## Step 4: Rate-of-Change Features (Alternative C — Cheap Supplement) -- V10 CHANGES -- DONE

Concatenate frame-to-frame L2 distance as extra scalar input. Actions have characteristic velocity profiles (fast at transitions, slow in middle). Computed on projected features (post input_proj, 128-d), not raw 2048-d.

CLI flag: `--use_rate_of_change`

**Files**: `models/model_multiprong.py` — ProgressHead.__init__ and forward()

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

### Round 1: Architecture fixes only (Step 1) -- DONE
1. **Git**: commit+push current changes to `main` → create `progress_changes` branch -- DONE
2. **Implement Step 1** (all 4 sub-fixes together): input projection, wider GRU, clamped linear, per-frame running count -- DONE
3. **Train**: Progress-only on frozen encoder with dense supervision + raw 2048-d features -- DONE (50/50 epochs, completed 2026-03-07)
4. **Evaluate**: Extract progress, compare V9 vs V6 -- DONE

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

### V9 Final Results (Epoch 50/50, 2026-03-07)

Training completed at ~05:00 UTC. Extraction via `extract_progress.py` completed at ~12:55 UTC. Compared V9 against V6 across 5 overlapping test videos (2 BaconAndEggs, 5 Sandwich), 10 segments with length >= 15 frames.

**Summary Table:**

| Metric | V6 | V9 (epoch 50) | Change |
|--------|-----|---------------|--------|
| Avg gap-to-1.0 (final pred) | 0.1748 | 0.1441 | **17.6% closer to 1.0** |
| Segments exceeding 0.90 max | 0/10 | 3/10 | **Ceiling partially broken** |
| 2nd-half MAD (avg) | baseline | 2.44x higher | **Model keeps adjusting** |
| Last-20% MAD (avg) | baseline | 7.38x higher | **More active but oscillatory** |
| Max value range | 0.80-0.84 | 0.82-0.90 | **Higher ceilings** |

**Best improvements on long segments:**
- S07 Sandwich (138 frames): V6 max=0.83, V9 max=**0.90** (gap reduced by 0.073)
- S18 Sandwich (149 frames): V6 max=0.83, V9 max=**0.90** (gap reduced by 0.061)
- OP01 BaconAndEggs (521 frames): V6 max=0.83, V9 max=**0.89** (gap reduced by 0.046)

**What V9 fixes:**
- Saturation mechanism: model keeps adjusting in 2nd half (2.44x higher MAD), not frozen like V6
- Ceiling raised: 3/10 segments now exceed 0.90 (V6 never exceeded 0.84)
- Longer segments benefit most (gap-to-1.0 reduced by 0.05-0.07 for 138-521 frame segments)

**What V9 does NOT fix:**
- **Early jump persists**: ALL 10/10 segments still jump to 0.7-0.87 in first 1-3 frames
- **Ceiling still below 1.0**: Most segments cap at 0.84-0.87, only 3 reach 0.90+
- **Increased oscillation**: Last-20% MAD is 7.38x higher than V6 (instability)
- **Curve shape still wrong**: Should be gradual 0→1 ramp, still 0→0.7 jump + slow crawl

**Example (521-frame segment, BaconAndEggs OP01-R03):**
- V6: first 3 preds=[0.26, 0.65, 0.73], last=0.8259, MAD last 20%=0.0007
- V9: first 3 preds=[0.30, 0.61, 0.69], last=0.8716, MAD last 20%=0.0051 (7x more active)
- V9 reaches 0.8869 max but neither approaches 1.0

**Conclusion:** V9 partially fixes the saturation problem (model stays active, ceilings raised to 0.90 on long segments) but the curve shape is fundamentally unchanged — early jump dominates, and the model lacks disambiguation signal to know "how far through" an action it is. This confirms **Round 2 (action-class conditioning)** is the critical next step.

### Round 2: Add disambiguation -- IN PROGRESS
5. **Step 2**: Action-class conditioning (nn.Embedding for action types)
6. **Step 4**: Rate-of-change features (frame-to-frame L2 distance)
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
