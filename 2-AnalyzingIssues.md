# Analysis of OGPE Issues & Next Steps (Part 2)

## Table of Contents

1. [Early Saturation with Raw 2048-d Features — What Next?](#1-early-saturation-with-raw-2048-d-features--what-next)
2. [Justifying Cumulative L2 vs Naive Baselines](#2-justifying-cumulative-l2-vs-naive-baselines)
3. [Making Cumulative L2 Quantitatively Better](#3-making-cumulative-l2-quantitatively-better)
4. [Lessons from ProTAS](#4-lessons-from-protas)
5. [Length-Based OGPE Analysis](#5-length-based-ogpe-analysis)
6. [Multiple Heads for Short/Medium/Long — Analysis](#6-multiple-heads-for-shortmediumlong--analysis)
7. [Prioritized Action Plan](#7-prioritized-action-plan)
8. [Critique of Fix 4 (Running Count) & Alternative Temporal Signals](#8-critique-of-fix-4-per-frame-running-count--alternative-temporal-signals)

---

## Current Numbers Reference

| Method | OGPE |
|--------|------|
| **Naive Baselines** | |
| Constant 0.0 | 0.5591 |
| Constant 0.25 | 0.3522 |
| Constant 0.50 | 0.2625 |
| Constant 0.75 | 0.2931 |
| Constant 1.0 | 0.4409 |
| Random [0,1] | 0.3433 +/- 0.009 |
| **Cumulative L2 (1fps)** | |
| GTCC (v2) | 0.5137 |
| TCC (v2) | 0.5103 |
| LAV (v2) | 0.5899 |
| VAVA (v2) | 0.5490 |
| **Cumulative L2 (4fps, 1fps-trained)** | |
| GTCC | 0.5069 |
| **Learnable Progress (v1)** | |
| GTCC | 0.2583 |
| TCC | 0.2575 |
| LAV | 0.2529 |
| VAVA | 0.2613 |
| **Learnable Progress (v2)** | |
| GTCC | 0.3034 |
| **ProTAS (oracle GT class, action-level)** | |
| Full dataset | 0.4792 |
| Test split v2 | 0.4097 |

---

## 1. Early Saturation with Raw 2048-d Features — What Next?

### What We Observe

With raw 2048-d pre-computed features (no alignment influence), the progress head now produces different predictions for different videos — the identical-prediction problem from aligned embeddings is solved. However, a new problem emerged: **early saturation**. For a 45-frame action, the model predicts:

```
Frame 1: 0.254 → Frame 5: 0.732 → Frame 7: 0.795 → Frame 10-45: ~0.80 (plateau)
```

The model rockets to ~0.80 in the first 7 frames, then effectively stops progressing for the remaining 38 frames. The curve has the right general direction but the wrong dynamics — it should be a gradual ramp from 0 to 1, not a sharp rise followed by a plateau.

### Why This Happens — Three Compounding Factors

**Factor A: The GRU is overwhelmed by the input dimensionality.**

Think of the GRU's 64-dimensional hidden state as a small notebook trying to summarize an encyclopedia. Each frame brings 2048 numbers, but the GRU can only maintain 64 numbers of "memory." That's a 32:1 compression ratio. After processing just a few frames, the notebook is "full" — the hidden state has captured the general gist ("I've seen several frames of someone cooking") but has no room to track fine-grained changes ("this is frame 7 vs frame 38 of chopping").

**Concrete example**: Imagine you're watching someone chop vegetables. Frames 1-7 show dramatically different content: picking up knife, positioning, first cut, second cut. The GRU's hidden state changes rapidly. But frames 8-45 show visually similar content: repeated chopping motions. The 2048-d features for frames 8, 20, and 40 are very similar (all show chopping), so the GRU's hidden state barely updates. The model has no way to distinguish "early chopping" from "late chopping" because the features look the same.

With a 64-d hidden state processing 2048-d input, the GRU's update gate quickly converges to near-zero — new frames cause negligible state changes. This is the classic GRU saturation phenomenon.

**Factor B: The model has no concept of "how long is this action?"**

The `frame_count` feature was disabled in dense mode to prevent a length shortcut (where the model just memorizes "if I've seen T frames, predict T/total"). But without ANY temporal signal, the model cannot distinguish:
- "I've seen 7 of 10 frames" (should predict ~0.7 — makes sense!)
- "I've seen 7 of 45 frames" (should predict ~0.16 — very different!)

Both situations present the same 7 embeddings to the GRU, so the GRU produces the same output. The model converges to a compromise: predict ~0.80 after seeing "enough" frames, because that's roughly correct for medium-length actions and not catastrophically wrong for either short or long ones.

**Factor C: The sigmoid activation compresses the upper range.**

The output goes through `nn.Sigmoid()`, which maps real numbers to (0, 1). The issue is that sigmoid is highly nonlinear at the extremes:
- To output 0.25, the pre-sigmoid value needs to be -1.1 (easy to learn)
- To output 0.50, the pre-sigmoid value needs to be 0.0 (easy)
- To output 0.80, the pre-sigmoid value needs to be 1.4 (moderate)
- To output 0.90, the pre-sigmoid value needs to be 2.2 (requires 57% more than 0.80)
- To output 0.95, the pre-sigmoid value needs to be 2.9 (requires 107% more than 0.80)
- To output 0.99, the pre-sigmoid value needs to be 4.6 (requires 229% more than 0.80)

So going from 0.80 to 1.0 requires the pre-activation to increase by 3.2 units, while going from 0.25 to 0.80 only required an increase of 2.5 units. The sigmoid makes it disproportionately hard to push predictions from "good" (0.80) to "great" (0.95+). The GRU's already-saturated hidden state has to produce ever-larger pre-activation values to make even tiny progress above 0.80 — a losing battle.

### Recommended Fixes

**Fix 1: Input Projection (addresses Factor A)**

Add a learned linear layer that compresses 2048-d to 128-d before the GRU sees it. This is exactly what ProTAS does (Conv1d 2048 to 64 as its very first operation).

**Why this helps**: Instead of the GRU trying to simultaneously compress and remember 2048-d features, a dedicated projection layer learns "which 128 dimensions of these 2048 features actually matter for progress prediction." The GRU then processes 128-d features with its 128-d hidden state — a 1:1 ratio instead of 32:1. The GRU has much more capacity to track subtle frame-to-frame changes.

**Example analogy**: It's like giving a student a well-organized summary of a textbook chapter (128 key points) vs. the entire raw chapter (2048 paragraphs). The student (GRU) can track how the story evolves much better from the summary.

File: `models/model_multiprong.py` — add to all three head variants (GRU, Transformer, DilatedConv)

**Fix 2: Wider GRU Hidden State (addresses Factor A)**

Increase GRU hidden dimension from 64 to 128 when using raw features.

**Why this helps**: A 128-d hidden state gives the GRU twice the "notebook space" to track temporal evolution. More importantly, with the input projection reducing input to 128-d, the compression ratio goes from 32:1 (2048 to 64) to 1:1 (128 to 128). The GRU can now dedicate different hidden dimensions to tracking different aspects of the evolving action without running out of capacity.

File: `configs/generic_config.py`

**Fix 3: Replace Sigmoid with Clamped Linear (addresses Factor C)**

Remove `nn.Sigmoid()` and use `torch.clamp(output, 0, 1)` instead.

**Why this helps**: With a clamped linear output, the relationship between pre-activation and output is linear within [0, 1]. Going from 0.80 to 0.95 requires the same pre-activation increase as going from 0.25 to 0.40. There's no compression penalty at the upper end. The gradient flows equally well at all output levels, so the model can learn to push predictions from 0.80 toward 1.0 just as easily as it learned to push from 0.25 toward 0.80.

**Example**: Think of sigmoid as trying to fill a bottle with a narrowing neck — the last 20% takes disproportionate effort. Clamped linear is like filling a straight cylinder — uniform effort throughout.

File: `models/model_multiprong.py`

**Fix 4: Per-Frame Running Count (addresses Factor B)**

Instead of a single broadcast frame_count value (which was disabled because it was a shortcut), provide a per-frame running index: frame 1 gets 1/300, frame 2 gets 2/300, frame 50 gets 50/300, etc.

**Why this helps and why it's not a shortcut**: The old `frame_count` broadcast the SAME value (total action length T) to ALL frames. In dense mode, every frame saw the same T, so the model could trivially learn "T=45 means predict linearly from 0 to 1 in 45 steps" without looking at embeddings at all. That's the shortcut.

The per-frame running count is DIFFERENT at each frame position but does NOT reveal the total action length. Frame 5 gets value 5/300 = 0.017 whether the action is 10 frames or 100 frames long. The model knows "I've seen 5 frames" but NOT "this action is 10 frames total." It must still use the embedding content to estimate how far along it is. But the running count breaks the GRU saturation problem because each frame has a unique temporal identifier — the GRU's update gate can condition on "this is a new frame" rather than "this looks identical to the last frame."

**Example**: A runner on a track knows they've completed 5 laps (running count) but doesn't know if the race is 10 laps or 50 laps (total length). They must observe the course context (features) to estimate their progress, but the lap counter helps them track that time is actually passing.

File: `models/model_multiprong.py`

---

## 2. Justifying Cumulative L2 vs Naive Baselines

### The Core Paradox

Cumulative L2 produces progress curves that look visually correct — they monotonically increase within each action, following the general shape of how an action unfolds over time. A human looking at the plots would say "yes, this captures progress." Yet on OGPE (mean absolute error), constant 0.5 wins (0.2625 vs 0.51+). How can a temporally-aware model lose to a method with zero temporal awareness?

### Why OGPE Fails to Capture the Truth

OGPE measures **pointwise error** — at each frame, how far is the prediction from the ground truth? It doesn't care about ordering, shape, or trends. Consider this concrete example:

**Action: 10 frames of "chop vegetables"**
```
Ground truth progress: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

Cumulative L2 (good shape, wrong scale):
  Prediction:           [0.0, 0.2, 0.5, 0.8, 1.1, 1.3, 1.5, 1.7, 1.8, 2.0]
  Per-frame error:      [0.1, 0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.9, 1.0]
  MAE = 0.56

Constant 0.5 (no shape, lucky value):
  Prediction:           [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
  Per-frame error:      [0.4, 0.3, 0.2, 0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
  MAE = 0.25
```

The cumulative L2 model correctly identified that the person is near the start at frame 1 and near the end at frame 10. It understands temporal progression. But because its scale is 2x too large, every prediction in the second half overshoots badly. Meanwhile, constant 0.5 knows nothing about time — it predicts the same value for the start and end of every action — but it happens to be the mathematical optimum for minimizing MAE against a uniform ramp.

This is analogous to a student who understands the concept but makes arithmetic errors losing to a student who just writes "C" for every multiple-choice answer on a test where "C" happens to be correct 25% of the time.

### Shape-Aware Metrics That Tell the Real Story

**Spearman Rank Correlation (most important)**

This metric asks: "Ignoring absolute values, do the predictions and ground truth agree on the ORDERING of frames?" It ranks all frames by predicted progress and by GT progress, then correlates the ranks.

- **Constant 0.5**: All frames get the same prediction, so they all have the same rank. There is zero correlation with any ordering. Spearman rho = 0.0 (or undefined due to no variance).
- **Cumulative L2**: Predictions increase monotonically, exactly matching the GT ordering. Spearman rho ~ 1.0 (near-perfect).
- **Random [0,1]**: Random ordering. Spearman rho ~ 0.0.

**Example**: For the 10-frame chop action above:
```
Cumulative L2 ranks:  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
GT ranks:             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
-> Spearman rho = 1.0  (perfect rank agreement)

Constant 0.5 ranks:   [5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5]  (all tied)
GT ranks:             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
-> Spearman rho = 0.0  (no temporal awareness)
```

This single metric obliterates the narrative that "constant 0.5 is better." It demonstrates that cumulative L2 has genuine temporal understanding while constant 0.5 has none.

**R-squared (Coefficient of Determination)**

R-squared answers: "How much of the variance in GT progress does this model explain?" By definition, predicting the mean (which is approximately constant 0.5) gives R-squared = 0.0. Any model that actually tracks progress gets R-squared > 0. Negative R-squared means worse than predicting the mean.

- **Constant 0.5**: R-squared = 0.0 exactly (it IS the mean prediction)
- **Cumulative L2 (clipped to [0,1])**: R-squared should be 0.6-0.9 (explains most variance)
- **Cumulative L2 (unclipped, scale 2x)**: R-squared may be negative (overshoot introduces more variance than it explains)

This metric cleanly separates "learned something" (R-squared > 0) from "learned nothing" (R-squared = 0) from "made things worse" (R-squared < 0).

**Monotonicity Score**

What fraction of consecutive frame pairs within an action have non-decreasing predicted progress?

- **Constant 0.5**: 100% (trivially — all values equal)
- **Cumulative L2**: 100% (by construction — cumulative sum can only increase)
- **Random [0,1]**: ~50% (random ordering)
- **Learnable heads**: Should be >90% if working correctly

Alone, this metric doesn't separate constant from cumulative L2 (both get 100%). But combined with Spearman, the picture is clear: cumulative L2 has both perfect monotonicity AND perfect rank correlation, while constant has perfect monotonicity but zero rank correlation. The combined story: cumulative L2 understands temporal progression, constant doesn't.

**Normalized OGPE**

`normalized_OGPE = 1 - (model_OGPE / constant_0.5_OGPE)`

This frames results relative to the trivial baseline:
- Value > 0: better than constant 0.5
- Value = 0: same as constant 0.5
- Value < 0: worse than constant 0.5

Examples from actual results:
- Cumulative L2 GTCC (unclipped): `1 - 0.5137/0.2625 = -0.96` -> 96% worse than constant
- Learnable V1 GTCC: `1 - 0.2583/0.2625 = +0.016` -> 1.6% better
- Cumulative L2 (with self-normalization, estimated): `1 - 0.17/0.2625 = +0.35` -> 35% better

This makes the comparison transparent and honest.

**Implementation**: Create `eval_comprehensive_metrics.py` or extend `utils/evaluation_action_level.py`

---

## 3. Making Cumulative L2 Quantitatively Better

### Why Cumulative L2 Currently Fails on OGPE

The cumulative L2 pipeline works as follows:
1. For each action, compute pairwise L2 distances between consecutive frame embeddings
2. Take the cumulative sum -> monotonically increasing curve
3. Divide by the training-set mean cumulative distance for this action class (the "action mean")

The problem is step 3. The action mean is computed from training videos, but test videos can have very different total cumulative distances due to:
- Different video lengths for the same action class
- Different viewpoints, lighting, or execution speed
- Embedding space variance across videos

**Concrete example**: Suppose training videos of "chop vegetables" have cumulative L2 distances averaging 5.0. But a test video has cumulative distance 12.0 (perhaps the person chops more slowly or the camera angle is different). The predictions become:
```
Cumulative L2:    [0, 1.2, 2.4, 3.6, 4.8, 6.0, 7.2, 8.4, 9.6, 10.8, 12.0]
Divide by 5.0:    [0, 0.24, 0.48, 0.72, 0.96, 1.20, 1.44, 1.68, 1.92, 2.16, 2.40]
GT progress:      [0.09, 0.18, 0.27, 0.36, 0.45, 0.55, 0.64, 0.73, 0.82, 0.91, 1.00]
```

Predictions reach 2.40 at the end instead of 1.0. Frames 5-11 all have predictions >1.0, contributing massive error. This single normalization mismatch is why cumulative L2 gets 0.51 OGPE.

### Fix A: Clip Predictions to [0, 1] (Immediate, No Retraining)

Simply cap predictions: `pred = clamp(pred, 0, 1)`.

**Why this helps**: In the example above, after clipping:
```
Clipped predictions: [0, 0.24, 0.48, 0.72, 0.96, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
GT progress:         [0.09, 0.18, 0.27, 0.36, 0.45, 0.55, 0.64, 0.73, 0.82, 0.91, 1.00]
Per-frame error:     [0.09, 0.06, 0.21, 0.36, 0.51, 0.45, 0.36, 0.27, 0.18, 0.09, 0.00]
MAE = 0.23 (down from 0.88 without clipping!)
```

The first half of the action is predicted well (errors <0.4). The second half is capped at 1.0, which is the correct endpoint — so frames near the end have near-zero error. Only the middle frames suffer from "premature completion" (predicting 1.0 too early), but this is far better than unbounded overshoot.

Files: `utils/evaluation_action_level.py` (~line 426), `eval_4fps_cumulative.py` (~line 196)

### Fix B: Shift First Frame to Match GT (Immediate, No Retraining)

Ground truth starts at 1/L (not 0), but cumulative L2 always starts at 0 (before any distance is accumulated).

**Why this matters**: For a 5-frame action, GT starts at 0.20 but cumulative L2 starts at 0.00 — that's a 0.20 error on the very first frame. For short actions (which are 28% of the test set), this systematic error is a significant chunk of the MAE.

The fix: map predictions from [0, max_pred] to [1/L, 1.0] using `pred = (1 - 1/L) * (cum / max_cum) + 1/L`. This requires knowing the action length L, which is available during evaluation from the GT segmentation.

**Example for L=5 with self-normalization**:
```
Before shift: [0.00, 0.25, 0.50, 0.75, 1.00]
After shift:  [0.20, 0.40, 0.60, 0.80, 1.00]  (maps [0,1] -> [1/5, 1])
GT:           [0.20, 0.40, 0.60, 0.80, 1.00]
MAE = 0.00 (perfect match for a linear cumulative curve!)
```

### Fix C: Self-Normalization Per Action (Non-Causal, Evaluation Only)

Instead of dividing by the training mean, divide each action's cumulative distance by its own maximum: `pred = cum / cum[-1]`.

**Why this is powerful**: This guarantees predictions go from 0 to exactly 1.0 for every action, every video. No normalization mismatch is possible. The only remaining error comes from the SHAPE of the cumulative curve vs the linear GT ramp.

**Example**: If cumulative L2 produces a slightly concave curve (common — early frames often have larger distances as the action starts):
```
Self-normalized: [0.00, 0.15, 0.28, 0.39, 0.50, 0.60, 0.69, 0.77, 0.85, 0.92, 1.00]
GT (linear):     [0.09, 0.18, 0.27, 0.36, 0.45, 0.55, 0.64, 0.73, 0.82, 0.91, 1.00]
Per-frame error:  very small — max error ~0.09
MAE ~ 0.05-0.10 (dramatically better than 0.26!)
```

**Trade-off**: This is non-causal — you need to see the entire action to know `cum[-1]`. But for EVALUATION purposes, action boundaries come from GT segmentation, so this is valid. For online/causal deployment, you'd need to estimate the normalizer, but evaluation results still demonstrate the model's potential.

**Expected impact**: Self-normalization should bring cumulative L2 OGPE to approximately 0.05-0.17, dramatically beating constant 0.5 (0.26). The exact value depends on how linear the cumulative distance curve is within each action.

### Fix D: Percentile Normalization

Use the 90th percentile of training distances instead of the mean. The mean is sensitive to outliers — one very long or very different training video can skew the mean. The 90th percentile is more robust and slightly over-estimates, which combined with clipping, prevents most overshoot.

**Why percentile is better than mean**: Consider an action class where 9 training videos have cumulative distance ~5.0 and 1 outlier has distance ~50.0. The mean is ~9.5 (inflated by the outlier). But the 90th percentile is ~5.0 (reflects the typical case). Test videos are likely to be ~5.0, so the percentile normalizer gives much better predictions.

### Recommended Order

Apply Fix A (clipping) + Fix B (shift) immediately and re-evaluate. If still worse than constant 0.5, apply Fix C (self-normalization) which should definitively beat it. Fix D is optional refinement.

---

## 4. Lessons from ProTAS

### How ProTAS Works

ProTAS (Procedure-aware Task-graph Segmentation) is a multi-stage temporal convolutional network designed for **action segmentation** — predicting which action is happening at each frame. It was NOT designed primarily for progress prediction. Progress is an auxiliary signal.

**Architecture walkthrough**:
1. **Input**: 2048-d ResNet50 features, shape (2048, T)
2. **Dimensionality reduction**: Conv1d(2048 -> 64) — compresses features to 64-d
3. **Temporal feature extraction**: 10 dilated residual convolutional layers with exponentially increasing dilation (1, 2, 4, 8, ..., 512). This gives each frame a receptive field of 1024 frames — it can "see" far into the past and future.
4. **Split into two paths**:
   - **Classification path**: Conv1d(64 -> num_classes) -> predicts action class per frame
   - **Progress path**: GRU(64 -> 64, bidirectional) -> Conv1d(64 -> num_classes) -> predicts progress per action class per frame
5. **Fusion**: Concatenates classification + progress outputs -> Conv1d -> refined classification
6. **Multi-stage refinement**: This whole process repeats 4 times, with each stage refining the previous stage's softmax predictions

**Key insight: Progress serves classification, not the other way around.** The progress predictions help the task graph decide "has action A been completed yet? If so, action B can begin." Progress accuracy only matters insofar as it helps classify the correct action — there's no reward for getting progress exactly right.

### Why ProTAS Gets Bad OGPE (0.41-0.48)

Even with oracle GT action class labels (which is generous — at test time you wouldn't have these), ProTAS gets action-level OGPE of 0.41-0.48. This is WORSE than constant 0.5 (0.26). Why?

1. **Progress is not the optimization target.** ProTAS's total loss is: `classification_CE + 0.15 * temporal_smoothness_MSE + progress_MSE + 0.1 * graph_loss`. The progress MSE is just one of four terms, and the primary optimization pressure is on classification accuracy. Progress predictions are "good enough to help the task graph" but not calibrated for standalone progress evaluation.

2. **Per-class progress channels create sparse supervision.** ProTAS outputs (num_classes, T) — separate progress curves for each action class. At any given frame, only ONE class has ground truth progress > 0 (the active action). All other classes should predict 0. So for 10 action classes, 90% of the progress output is trained to predict 0 at any frame. The model spends most of its progress capacity learning to predict zeros for inactive classes.

3. **The GRU is bidirectional.** ProTAS's progress GRU sees both past AND future context. This is fine for offline segmentation but means the progress predictions are not causal — frame 5's progress prediction already "knows" what happens at frame 45. This makes the predictions less representative of genuine temporal understanding and more of a post-hoc smoothing.

### What IS Transferable to GTCC

**Idea A: Dimensionality Reduction Before Temporal Processing**

ProTAS's very first operation is `Conv1d(2048 -> 64)`. It NEVER feeds raw 2048-d features to a recurrent network. This is the most important lesson.

**Why it matters for GTCC**: Our early saturation problem (Section 1) stems directly from the GRU trying to process 2048-d features with a 64-d hidden state. ProTAS avoids this entirely by projecting down first. The temporal processing (dilated convs + GRU) operates entirely in 64-d space, where the GRU can track meaningful temporal evolution.

**Analogy**: ProTAS first "translates" the raw visual features into a compact task-relevant representation, THEN reasons about temporal progression. GTCC currently asks the GRU to simultaneously translate AND reason — it's like asking someone to summarize a book while simultaneously reading it in a foreign language.

**Idea B: Dilated Convolutions for Multi-Scale Temporal Context**

ProTAS uses 10 dilated residual layers (dilations: 1, 2, 4, ..., 512) before the GRU. This means each frame's 64-d feature already encodes information about what happened 512 frames ago and what will happen 512 frames in the future (in the non-causal case).

**Why this helps progress prediction**: Progress estimation inherently requires multi-scale temporal reasoning. To predict "I'm 30% through chopping," you need to know:
- Local context (frame-to-frame): "the knife is currently moving downward" (scale: 1-2 frames)
- Medium context: "we started chopping 5 seconds ago" (scale: 5-20 frames)
- Global context: "the entire chopping action takes about 15 seconds" (scale: 15-50 frames)

A single-layer GRU can only capture this through sequential processing, which leads to the saturation problem. Dilated convolutions capture all scales simultaneously and in parallel.

**For GTCC**: We can add 4-6 causal dilated residual blocks (we already have `CausalDilatedResidualBlock` in the codebase in `models/model_multiprong.py`) between the input projection and the GRU. This gives the GRU temporally-enriched features instead of raw frame features.

**Idea C: Simple Dense MSE Works (with the right architecture)**

ProTAS uses the simplest possible progress loss: `MSE(pred, gt).mean()` on all frames. No fancy weighting, no sampling, no boundary losses, no monotonicity penalties. This works because the architecture (dilated convs + GRU on 64-d features) is well-suited to the task.

**Lesson**: Our dense mode (`_progress_loss_dense()`) already implements this. The loss was never the problem — the architecture was. Once we fix the input projection and temporal context (Ideas A & B), the simple dense MSE should work.

### What NOT to Transfer

- **Per-class progress channels**: GTCC has no action classification component. Adding one would be a fundamental architecture change with unclear benefit for standalone progress prediction.
- **Task graph**: Highly specific to ProTAS's joint classification+progress objective. The predecessor/successor matrices encode action ordering constraints that only make sense when you're classifying actions.
- **Multi-stage refinement**: Each stage of ProTAS refines the previous stage's classification softmax. Without a classification component, there's nothing to refine. The added complexity is not justified for standalone progress prediction.

---

## 5. Length-Based OGPE Analysis

### Mathematical Foundation

For an action of length L, the ground truth progress is a linear ramp: `GT = [1/L, 2/L, ..., L/L]`. The MAE of constant prediction c = 0.5 against this ramp is:

```
MAE(0.5, L) = (1/L) * sum_{i=1}^{L} |i/L - 0.5|
```

**Worked example for L = 5**:
```
GT:      [0.2, 0.4, 0.6, 0.8, 1.0]
Pred:    [0.5, 0.5, 0.5, 0.5, 0.5]
Errors:  [0.3, 0.1, 0.1, 0.3, 0.5]
MAE = (0.3 + 0.1 + 0.1 + 0.3 + 0.5) / 5 = 0.26
```

**Worked example for L = 20**:
```
GT:      [0.05, 0.10, ..., 0.95, 1.00]
Pred:    [0.50, 0.50, ..., 0.50, 0.50]
Errors:  [0.45, 0.40, ..., 0.45, 0.50]
```
The errors form a "V" shape centered at frame 10 (where GT = 0.5), with the deepest point being 0 error and the endpoints being 0.45-0.50 error. MAE ~ 0.251.

**Complete table**:

| Action Length (L) | Constant 0.5 MAE | Interpretation |
|---|---|---|
| 1 | 0.500 | Only frame has GT=1.0, pred=0.5. Terrible. |
| 2 | 0.250 | GT=[0.5, 1.0], errors=[0.0, 0.5], MAE=0.25 |
| 3 | 0.278 | Slightly worse than L=2 due to asymmetry |
| 5 | 0.260 | |
| 10 | 0.252 | Approaching the floor |
| 20 | 0.251 | |
| 50 | 0.250 | Essentially at floor |
| infinity | 0.250 | Continuous limit: integral from 0 to 1 of |x - 0.5| dx = 0.25 |

**Key observations**:
1. For L >= 10, constant 0.5 achieves essentially the theoretical minimum (0.25). No model can improve by more than 0.25 on these actions.
2. For L = 1, constant 0.5 has MAE = 0.50 — the only length where there's significant room for improvement, but a 1-frame action is fundamentally unpredictable.
3. The observed overall OGPE of 0.2625 is explained by the 28.4% of actions with L <= 5, which pull the average up from 0.25.

### Test Set Action Length Distribution (1fps)

| Length | Count | % |
|--------|-------|---|
| 1 frame | 11 | 3.4% |
| 2 frames | 13 | 4.1% |
| 3 frames | 26 | 8.1% |
| 4 frames | 21 | 6.5% |
| 5 frames | 20 | 6.2% |
| **<= 5 total** | **91** | **28.4%** |
| 6-10 | 93 | 29.0% |
| 11-20 | 58 | 18.1% |
| 21-50 | 59 | 18.4% |
| 50+ | 20 | 6.2% |

Mean: 17.5 frames, Median: 9.0. Original dataset: 24fps. Currently using 1fps (24x downsampled).

### Where Models SHOULD Outperform Baselines

**Short actions (L = 1-5, 28.4% of test set)**:
- Constant 0.5: MAE = 0.26-0.50
- Any model: Limited by minimal temporal context. 1-3 frames provide almost no distinguishing information.
- **Verdict**: Hard for ANY model to beat constant 0.5 here. The best strategy might be to simply predict 0.5 for these actions.

**Medium actions (L = 6-20, 47.1% of test set)**:
- Constant 0.5: MAE ~ 0.251-0.253
- Cumulative L2 (self-normalized): Should achieve MAE ~ 0.05-0.15 (the cumulative curve closely tracks the linear ramp over 6-20 frames)
- **Verdict**: This is where real models should clearly win. 6+ frames provide enough temporal context to track progression.

**Long actions (L = 21+, 24.6% of test set)**:
- Constant 0.5: MAE ~ 0.250
- Cumulative L2 (self-normalized): Should achieve MAE ~ 0.03-0.10 (very close to linear over long sequences)
- **Verdict**: Models should dominate here. More frames = better temporal signal.

### Proposed Evaluation: Per-Length-Bin Breakdown

Create `eval_per_length_bin.py` that reports OGPE per length bin for all methods. Expected output format:

```
Length Bin | N_actions | Const_0.5 | CumL2_clip | CumL2_selfnorm | Learnable_GRU
1-2       |    24     |  0.375    |   ~0.40    |    ~0.30        |    ~0.38
3-5       |    67     |  0.266    |   ~0.25    |    ~0.18        |    ~0.26
6-10      |    93     |  0.252    |   ~0.20    |    ~0.10        |    ~0.22
11-20     |    58     |  0.251    |   ~0.17    |    ~0.08        |    ~0.18
21-50     |    59     |  0.250    |   ~0.15    |    ~0.05        |    ~0.15
50+       |    20     |  0.250    |   ~0.14    |    ~0.04        |    ~0.12
ALL       |   321     |  0.2625   |   ~0.22    |    ~0.13        |    ~0.23
```

*(CumL2 estimates are approximate — actual values depend on embedding quality. Need to run to get real numbers.)*

**Why this table is powerful**: It immediately reveals that constant 0.5 is only competitive on short actions (where everything is hard), while real models dominate on medium and long actions. The overall OGPE average hides this because short actions are 28% of the test set and pull the average toward the constant baseline's level.

---

## 6. Multiple Heads for Short/Medium/Long — Analysis

### The Proposal

Train 3 separate progress heads specialized for different action duration ranges:
- Head A: Short actions (1-10 frames)
- Head B: Medium actions (11-30 frames)
- Head C: Long actions (30+ frames)

Each head would learn duration-specific patterns — Head A would learn to make quick predictions from minimal context, while Head C would learn gradual curves over many frames.

### Why This Fails for Online/Causal Prediction

**The fundamental chicken-and-egg problem**: To route to the correct head, you need to know the action length. But you don't know the action length until the action is over. At inference time, when you've seen 5 frames, you face an unresolvable ambiguity:

**Scenario 1**: This is a 7-frame action (short). You're at 71% progress. Head A should predict ~0.71.
**Scenario 2**: This is a 50-frame action (long). You're at 10% progress. Head C should predict ~0.10.

The features at frame 5 may look identical in both scenarios (both could show someone midway through a chopping motion). Without knowing the total length, you cannot route correctly, and the predictions from wrong heads would be catastrophically wrong (0.71 vs 0.10 — a 0.61 error).

### Why All Workarounds Are Flawed

**Workaround 1: Start with Head A, switch to Head B at frame 11, Head C at frame 31.**

Problem: Discontinuities at switch points. At frame 10, Head A predicts "nearly done" (e.g., 0.85 for a short action). At frame 11, Head B activates and predicts "just getting started" (e.g., 0.35 for a medium action). The progress prediction jumps from 0.85 to 0.35, violating monotonicity. This would look terrible visually and destroy rank correlation metrics.

**Workaround 2: Run all 3 heads in parallel, average their outputs.**

Problem: For a 50-frame action, Head A's predictions are meaningless (it was trained on 1-10 frame actions and has no idea what to do with 50 frames). Averaging a meaningful prediction (from Head C: 0.10) with a meaningless one (from Head A: 0.95) gives 0.53 — worse than either alone. Averaging defeats the purpose of specialization.

**Workaround 3: Predict the action length first, then route.**

Problem: Predicting total action length from a partial observation is essentially the same problem as predicting progress. If you could accurately predict "this action will be 50 frames long" from the first 5 frames, you could also just directly predict "I'm at 10% progress." The length estimator faces the same early saturation problem as the progress head.

### Better Alternative: Single Adaptive Head with Running Count

Instead of multiple heads, use a single head that naturally adapts its behavior based on how many frames it has seen. The per-frame running count (Fix 4 from Section 1) provides this:

- At frame 5 with count=5/300=0.017: The model learns "I've seen very few frames relative to the maximum possible. I'm likely early in a long action OR near the end of a very short action. Use the embedding content to disambiguate."
- At frame 50 with count=50/300=0.167: The model learns "I've seen a moderate number of frames. I'm somewhere in the middle of a medium-to-long action."

The running count gives the model continuous, graduated length information without discrete routing decisions. There are no discontinuities, no wrong-head catastrophes, and no averaging artifacts.

**Optional enhancement — Auxiliary length estimation**: Train the same head to predict two outputs: (1) progress and (2) estimated remaining frames. The auxiliary loss `MSE(pred_remaining, actual_remaining)` encourages the model to build an internal representation of temporal scale, which benefits the progress prediction. This is a soft, gradient-based version of "predict length then route," but without the hard routing decision.

### Comparison with ProTAS's Per-Class Approach

ProTAS uses per-action-CLASS channels (one progress output per action type). This works because:
- **Action type is observable**: You can infer "this is chopping" from the visual features at any frame
- **Routing is implicit**: The classification head determines which progress channel to read
- **No future information needed**: Action type is determined by the current activity, not its duration

Per-length routing is fundamentally different:
- **Action length is NOT observable**: You cannot determine "this is a 50-frame action" from frame 5
- **Routing requires future knowledge**: The total length is only known after the action ends
- **Misrouting is catastrophic**: Wrong length assumption leads to wrong progress prediction by a large margin

**The lesson**: Only route based on features that are observable at inference time. Action type is observable; action length is not.

---

## 7. Prioritized Action Plan

### Priority 1: Fix Early Saturation (Blocks all learnable progress work)

**Goal**: Get 2048-d raw feature progress head to produce gradual 0->1 curves instead of early saturation.

Changes:
- Input projection (2048->128) + wider GRU (128-d) + remove sigmoid + per-frame running count
- Files: `models/model_multiprong.py`, `configs/generic_config.py`

### Priority 2: Fix Cumulative L2 Evaluation (No retraining needed)

**Goal**: Make cumulative L2 competitive with constant 0.5 on OGPE.

Changes:
- Add clipping to [0,1] + first-frame shift + self-normalization option
- Files: `utils/evaluation_action_level.py`, `eval_4fps_cumulative.py`

### Priority 3: Comprehensive Metrics (No retraining needed)

**Goal**: Demonstrate cumulative L2 and learnable heads have real advantages despite similar/worse OGPE.

Changes:
- Add Spearman rank correlation, R-squared, monotonicity score, normalized OGPE, per-length-bin analysis
- Files: New `eval_comprehensive_metrics.py` and/or `eval_per_length_bin.py`

### Priority 4: ProTAS-Inspired Architecture (If Priority 1 insufficient)

**Goal**: Build a better progress head inspired by ProTAS's architecture.

Changes:
- Add 4-6 causal dilated conv layers between input projection and GRU
- Reuse existing `CausalDilatedResidualBlock` from `models/model_multiprong.py`
- File: `models/model_multiprong.py` — new `ProTASInspiredProgressHead` class

---

## 8. Critique of Fix 4 (Per-Frame Running Count) & Alternative Temporal Signals

### The Claimed Benefit vs Reality

Fix 4 in Section 1 proposed replacing the broadcast `frame_count` with a per-frame running count (`i/300` for frame i). The claim was that this breaks the GRU saturation problem AND helps the model estimate progress without creating a shortcut.

**The running count has the same fundamental ambiguity as the existing frame_count:**

| Scenario | Running Count at Frame 5 | GT Progress |
|----------|--------------------------|-------------|
| 10-frame action, frame 5 | 5/300 = 0.0167 | 5/10 = **0.50** |
| 50-frame action, frame 5 | 5/300 = 0.0167 | 5/50 = **0.10** |
| 100-frame action, frame 5 | 5/300 = 0.0167 | 5/100 = **0.05** |

Same input, three wildly different targets. If the embeddings also look similar (both show early-stage chopping), the model receives identical input with conflicting supervision. It will converge to `E[progress | seen 5 frames]` — a weighted average over the training distribution of action lengths — which is wrong for all individual cases.

### How the Existing `frame_count` Actually Works (Code-Level)

From `models/model_multiprong.py`, lines 140-145:

```python
T = segment_embeddings.shape[0]   # frames IN THIS FORWARD PASS
fc_value = math.log1p(T) / math.log1p(self.frame_count_max)  # log(1+T)/log(1+300)
frame_count = torch.full((T, 1), fc_value, ...)  # SAME value for ALL T frames
```

What `T` actually represents depends on the mode:

**Sparse training & evaluation mode**: The progress head receives a partial segment `vid_emb[action_start : target_frame + 1]`. So `T = number of frames seen so far` = current position within the action.

- Frame 5 of 10-frame action → fed frames [0..4], T=5, frame_count = log(6)/log(301) ≈ 0.31
- Frame 5 of 50-frame action → fed frames [0..4], T=5, frame_count = log(6)/log(301) ≈ 0.31
- **Both produce identical frame_count values** — same ambiguity as running count

**Dense training mode**: The full action segment is fed at once. `T = total action length`. Every frame sees the total length — this IS the shortcut that was disabled.

- 10-frame action → all frames see frame_count = log(11)/log(301) ≈ 0.42
- 50-frame action → all frames see frame_count = log(51)/log(301) ≈ 0.69
- The model can trivially learn `progress ≈ position / decode(frame_count)` without using embeddings

### Existing frame_count vs Proposed Running Count — Functional Comparison

| Property | Existing `frame_count` | Proposed Running Count |
|----------|----------------------|----------------------|
| Value at frame i | `log(1+T)/log(1+300)` (same for all frames) | `i/300` (different per frame) |
| Reveals current position? | Yes (via T = frames seen) | Yes (via i) |
| Reveals total action length? | No (sparse/eval), Yes (dense) | No |
| Unique per frame within segment? | **No** (broadcast) | **Yes** (unique) |
| Helps GRU saturation? | **No** (same input each step) | **Yes** (unique input per step) |
| Resolves length ambiguity? | **No** | **No** |

**Conclusion**: In sparse/eval mode, both signals encode essentially the same information ("you've seen i frames") in different formats. The running count has one genuine mechanical advantage — unique per-frame values prevent GRU update gate saturation within a forward pass. But neither resolves the fundamental ambiguity: **progress = position / total_length**, and neither provides total_length.

### The Information-Theoretic Limitation

Progress prediction is fundamentally:

```
progress = frames_seen / total_action_length
```

Any temporal signal that reveals `frames_seen` (running count, frame_count) but not `total_action_length` leaves the problem underspecified. The model must infer total length from the embedding content. This only works when:

1. The visual content at frame 5 of a 10-frame action looks genuinely different from frame 5 of a 50-frame action (action-phase is visually distinguishable)
2. The model has learned priors over action durations (e.g., "chopping usually takes 15-40 frames")

For many actions, especially repetitive ones (chopping, stirring, mixing), frame 5 looks identical regardless of total duration. The embeddings carry no length-distinguishing signal. In these cases, no amount of temporal encoding helps — the ambiguity is irreducible without additional information.

### Alternative Temporal Signals That Could Actually Help

#### Alternative A: Action-Class Conditioning (Fully Causal)

If the model knows "this is chop_vegetables," it can learn that chopping typically takes 15-40 frames at 1fps. Seeing 5 frames of chopping → "I'm probably 12-33% through." This narrows the ambiguity dramatically because different action classes have very different length distributions.

**Implementation**: Concatenate a learnable action-class embedding to each frame's features. During training, use GT action labels from the segmentation annotations. At inference, either:
- Use a lightweight classifier on the first few frames
- Use GTCC's nearest-neighbor alignment to infer the action class
- Use the GT segmentation (which is already available during OGPE evaluation anyway)

**Pros**: Most informative signal. Directly addresses the length ambiguity by conditioning on action type.
**Cons**: Requires action class at inference. May not generalize to unseen action classes. Evaluation already uses GT segmentation boundaries, so using GT action class during eval is defensible.

#### Alternative B: Auxiliary Length Estimation Head (Fully Causal)

Train the progress head to predict TWO outputs: (1) progress and (2) estimated total action length. The length prediction has its own auxiliary loss:

```
total_loss = progress_MSE + alpha * length_MSE
where length_MSE = MSE(pred_total_length, actual_total_length)
```

The model's internal representation is forced to build a "how long will this action be?" estimate. At frame 5, it might predict "total ≈ 45 frames" → progress ≈ 5/45 ≈ 0.11. The length estimate naturally improves as more frames are seen.

**This is fundamentally different from the multi-head routing approach (Section 6)** because:
- No hard routing decisions — it's a soft, gradient-based estimate
- No discontinuities — a single head produces both outputs
- The length estimate is a continuous auxiliary signal, not a discrete routing key
- Wrong length estimates degrade gracefully (slightly wrong progress) rather than catastrophically (completely wrong head)

**Implementation**: Add a second output neuron to the FC layers. At the last layer: `[progress_logit, length_logit]`. Apply sigmoid to progress, apply softplus or exp to length (to keep it positive).

**Pros**: Directly addresses the core ambiguity. No external labels needed. Gradient-based, so improves smoothly.
**Cons**: Length prediction from partial observation is inherently noisy early on. Adds a hyperparameter (alpha weighting). May need careful loss balancing.

#### Alternative C: Rate-of-Change Features (Fully Causal)

Instead of just raw embeddings, provide the GRU with the frame-to-frame L2 distance (how fast features are changing). Actions have characteristic velocity profiles — fast transitions at action start/end, slower in the middle.

**Implementation**: At frame t, concatenate `||emb_t - emb_{t-1}||_2` as an extra scalar feature (0 for frame 0). Optionally also provide a running average of recent velocities.

```python
diffs = torch.zeros(T, 1)
diffs[1:] = torch.norm(segment_embeddings[1:] - segment_embeddings[:-1], dim=1, keepdim=True)
features_to_concat.append(diffs)
```

**Pros**: Zero labels needed. Provides genuine phase information (velocity profile). Cheap to compute.
**Cons**: Velocity alone doesn't resolve length ambiguity. May be noisy with raw 2048-d features. Most useful as a supplementary signal, not a primary fix.

#### Alternative D: Accept the Ambiguity — Design Around It

The most honest approach: **acknowledge that causal progress prediction without action-class information has an irreducible error floor**. The model cannot distinguish "frame 5 of 10" from "frame 5 of 50" when the visual content is identical.

**Practical consequences**:
1. **For evaluation/benchmarking**: Use cumulative L2 with self-normalization (non-causal). This sidesteps the ambiguity entirely by seeing the whole action and normalizing to [0, 1]. Expected OGPE: 0.05-0.17, decisively beating constant 0.5.
2. **For the learnable head**: Focus on the architecture fixes (input projection 2048→128, wider GRU, replace sigmoid) which address the saturation problem. Accept that the learnable head's OGPE will have a floor determined by the action-length distribution.
3. **For the paper narrative**: Use shape-aware metrics (Spearman, R²) that demonstrate temporal understanding regardless of scale calibration. The learnable head should achieve near-perfect Spearman correlation even if OGPE is only marginally better than constant 0.5.

**Pros**: Intellectually honest. Avoids wasting effort on an impossible problem. Redirects effort toward achievable improvements.
**Cons**: May not satisfy reviewers who want OGPE improvements. Requires a strong narrative around metric limitations.

### Recommended Strategy

Combine approaches rather than picking one:

1. **Implement Alternatives A + B together**: Action-class conditioning provides the strongest disambiguation signal. Auxiliary length estimation provides a learning-based fallback. These are complementary — the action class helps the length estimator converge faster.

2. **Add Alternative C as a cheap supplement**: Rate-of-change features are trivial to implement and provide additional temporal texture at no training cost.

3. **Use Alternative D for the evaluation story**: Self-normalized cumulative L2 for the benchmark numbers, comprehensive metrics for the narrative, and the learnable head as evidence that the representations support progress prediction.

4. **The running count (Fix 4) should be retained solely for its GRU anti-saturation benefit** — not for disambiguation. Document it honestly as a mechanical fix, not a temporal signal.
