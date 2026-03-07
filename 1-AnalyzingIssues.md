# Analysis of OGPE Issues & Proposed Solutions

## Part 1: Why Naive Baselines Appear to Beat Cumulative L2 on OGPE

### All Numbers Side by Side

| Method | OGPE |
|--------|------|
| **Naive Baselines** | |
| Constant 0.0 | 0.5591 |
| Constant 0.25 | 0.3522 |
| Constant 0.50 | 0.2625 |
| Constant 0.75 | 0.2931 |
| Constant 1.0 | 0.4409 |
| Random [0,1] | 0.3433 ± 0.009 |
| **Cumulative L2** | |
| GTCC (output_l2_progress) | 0.6961 |
| TCC | 0.5940 |
| LAV | 0.5783 |
| VAVA | 0.6598 |
| GTCC (output_l2_progress_v2) | 0.5137 |
| TCC (v2) | 0.5103 |
| LAV (v2) | 0.5899 |
| VAVA (v2) | 0.5490 |
| **Learnable Progress** | |
| GTCC (output_learnable_progress) | 0.2583 |
| TCC | 0.2575 |
| LAV | 0.2529 |
| VAVA | 0.2613 |
| GTCC (v2) | 0.3034 |

### Root Cause 1: OGPE (MAE) Has a Very Low Trivial Floor (~0.25)

Ground truth per action is a linear ramp: `[1/L, 2/L, ..., 1.0]` for action length L.

MAE of constant 0.5 against a uniform ramp on [0,1]:
- `∫₀¹ |0.5 - x| dx = 0.25`

So **any model must beat 0.25 to outperform "just say 0.5"**. The trivial baseline is already 75% of the way to perfection. The maximum possible improvement is only 0.25 (from 0.25 → 0.0).

For different action lengths:
- L=1: error = 0.500 | L=2: 0.250 | L=3: 0.278 | L=5: 0.260 | L=10: 0.252 | L→∞: 0.250

The observed 0.2625 matches this math given the action length distribution (28% ≤5 frames pushes average slightly above 0.25).

### Root Cause 2: Cumulative L2 Has Unbounded Scale

The cumulative L2 pipeline:
1. `get_cum_matrix(segment)` → `[0, d₁, d₁+d₂, ...]`
2. Divide by `action_means[action_name]['mean']` (training-set average)

**Problems:**
- **Predictions NOT bounded to [0,1]**: If test cumulative distance is 2× training mean → predictions reach 2.0. No clipping applied.
- **Training mean is a poor normalizer**: L2 distances vary enormously across videos due to lighting, viewpoint, speed differences.
- **First frame always 0, but GT starts at 1/L**: For 28% of actions (≤5 frames), this alone contributes 0.2+ error on frame 0.
- **Train-eval mismatch** (documented in `FixEval.md`): L2 progress models train with `cum/max_val` (always ends at 1.0) but evaluate with `cum/action_mean` (can be anything).

### Root Cause 3: Qualitative vs Quantitative Disconnect

Cumulative L2 produces the **right shape** — monotonically increasing within each action. Plots look correct. But OGPE measures **pointwise MAE**, not shape. A curve 0→2.0 instead of 0→1.0 has correct trend but ~0.5 MAE. Meanwhile, constant 0.5 is always within 0.5 of any GT value, and on average within 0.25.

### Root Cause 4: Learnable Models Barely Beat Constant 0.5

Learnable progress (0.253-0.261) vs constant 0.5 (0.262) — marginal improvement. This is NOT because learnable is bad:
- The theoretical floor is already 0.25
- Short actions (28% ≤5 frames at 1fps) give minimal temporal context
- Any prediction noise eats into the narrow 0.25 margin

### Action Length Distribution (Test Set, 1fps)

| Length | Count | % |
|--------|-------|---|
| 1 frame | 11 | 3.4% |
| 2 frames | 13 | 4.1% |
| 3 frames | 26 | 8.1% |
| 4 frames | 21 | 6.5% |
| 5 frames | 20 | 6.2% |
| **≤5 total** | **91** | **28.4%** |
| 6-10 | 93 | 29.0% |
| 11-20 | 58 | 18.1% |
| 21-50 | 59 | 18.4% |
| 50+ | 20 | 6.2% |

Mean: 17.5 frames, Median: 9.0. Original dataset: 24fps. Currently using 1fps (24× downsampled).

---

## Part 2: Proposed Solutions

### Solution A: Higher FPS (4fps or higher)

**Why it helps:**
At 1fps, a real-world 3-second action is 3 frames. At 4fps it's 12 frames. At 24fps (native) it's 72 frames.

More frames per action means:
- More temporal context for the model to distinguish early/mid/late
- Constant 0.5 baseline stays at ~0.25 regardless of FPS (the math doesn't change)
- But a good model can exploit the finer temporal resolution to get closer to 0.0
- The 28% of actions that are ≤5 frames at 1fps become ≤20 frames at 4fps — much more tractable

**Already supported:** The codebase already has 4fps data (`GTCC_Data_Processed_4fps/`, `ProTAS/data_4fps/`), 4fps training configs, and 4fps evaluation paths.

**But this alone doesn't fix the metric problem.** Constant 0.5 still achieves ~0.25 at any FPS.

### Solution B: Complementary Shape-Aware Metrics

OGPE only measures pointwise error. We need metrics that capture what we actually care about: **does the predicted curve look like correct progress?**

**B1. Monotonicity Score**
- What fraction of consecutive frame pairs have non-decreasing predicted progress within each action?
- Perfect model: 100%. Constant baseline: 100% (trivially — all values equal). Random: ~50%.
- This directly measures "does progress go up over time?"
- Note: constant baseline trivially achieves 100% since equal counts as non-decreasing. Need to pair with other metrics.

**B2. Rank Correlation (Spearman ρ or Kendall τ)**
- Within each action, compute rank correlation between predicted and GT progress.
- Measures ordinal agreement: "do frames predicted as further along actually occur later?"
- Perfect model: 1.0. Constant: undefined/0.0 (no variance → no correlation). Random: ~0.0.
- **This is the key metric that separates constant baselines from real models.** Constant 0.5 gets ρ≈0, while cumulative L2 should get ρ close to 1.0.
- Note: requires actions with ≥3 frames. For 1-2 frame actions, skip or handle separately.

**B3. Endpoint Accuracy**
- Error at first frame (should predict ~0/small) and last frame (should predict ~1.0) of each action.
- Captures whether the model knows when an action starts and ends.
- Constant 0.5: first frame error ≈ 0.5×(1/L), last frame error = 0.5. Cumulative L2: first frame always 0 (good), last frame depends on normalization.

**B4. R² (Coefficient of Determination) per action**
- Measures how much variance in GT progress the model explains.
- Perfect: 1.0. Constant: 0.0 (by definition — it IS the mean). Worse than constant: negative.
- Directly answers "is this model better than predicting the mean?"
- Note: need sufficient frames per action for this to be meaningful (≥5 recommended).

**B5. Normalized OGPE**
- `1 - (model_OGPE / constant_0.5_OGPE)` → improvement ratio over trivial baseline
- Makes the comparison explicit and FPS-independent.
- Learnable GTCC: `1 - 0.2583/0.2625 = 0.016` (1.6% improvement). This is honest about how small the margin is.

### Solution C: Per-Action-Length-Bin Analysis

Instead of a single OGPE number, report separately for:
- **Short actions (1-5 frames)**: Models are inherently limited here
- **Medium actions (6-20 frames)**: Models should show clear advantage
- **Long actions (20+ frames)**: Models should dominate constant baselines

This reveals WHERE models actually help, rather than averaging across regimes where constant baselines are hard to beat.

### Solution D: Fix Cumulative L2 Evaluation

**D1. Clip predictions to [0,1]**: `pred = torch.clamp(pred, 0, 1)` — prevents overshoot from inflating MAE.

**D2. Self-normalize per action**: Instead of dividing by training mean, divide each action's cumulative distance by its own max: `pred = cum / cum[-1]`. This guarantees the curve goes 0→1 and removes the normalization mismatch entirely. However, this changes the method from "online" (causal) to non-causal since you need the full action to know the max.

**D3. Shift start to match GT**: GT starts at 1/L, cumulative L2 starts at 0. Could linearly map cum L2 from [0, max] → [1/L, 1.0] instead. But this also requires knowing L (action length) which may not be available at inference time.

### Solution E: Weighted OGPE by Action Length

Weight each action's contribution to the mean by its frame count. Currently all actions contribute equally regardless of length, so a 1-frame action (where constant 0.5 error = 0.5) counts the same as a 100-frame action (where constant 0.5 error ≈ 0.25 but a good model might achieve 0.05).

Length-weighted OGPE would better reflect the actual user experience of watching progress unfold — you spend more time watching long actions.

### Solution F: Dense Evaluation at Higher FPS (without retraining)

If a model is trained at 1fps, evaluate at 4fps by:
1. Loading the 4fps features
2. Running the same encoder to get 4fps embeddings
3. Evaluating OGPE at 4fps (where actions have 4× more frames)

This tests temporal generalization without retraining and gives more frames to discriminate model quality.

---

## Part 3: Recommended Action Plan

### Immediate (no retraining needed):
1. **Add Spearman rank correlation** to eval — this single metric will clearly separate constant baselines (ρ≈0) from real models (ρ≈1)
2. **Add per-length-bin OGPE** to see where models actually help
3. **Add monotonicity score** — trivial to compute, very informative
4. **Report normalized OGPE** (improvement over constant 0.5)

### Short-term:
5. **Fix cumulative L2 eval** — add clipping to [0,1] or self-normalization
6. **Run naive baselines on all new metrics** to establish proper baselines

### Medium-term:
7. **Train and evaluate at 4fps** — already have the data pipeline
8. **Re-evaluate whether OGPE alone is sufficient** for paper/reporting
