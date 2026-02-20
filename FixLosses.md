# Loss Function Fixes

**Branch:** `fix_loss`
**Date:** 2026-02-18

## Summary

Fixed semantic changes in `utils/loss_functions.py` that were introduced during memory optimization but inadvertently changed how the losses work.

---

## Issues Found and Fixed

### 1. IntraContrast_loss - `.mean()` vs `.sum()`

**Problem:** Changed from `.sum()` to `.mean()`, which reduces the contrastive loss magnitude by a factor of N².

**Fix:** Restored `.sum()` aggregation (line 77).

```python
# BEFORE (incorrect)
result = (outside_window + inside_window).mean()

# AFTER (correct - matches original)
result = (outside_window + inside_window).sum()
```

### 2. LAV_loss - Gradients blocked for contrastive terms

**Problem:** IntraContrast calls were wrapped in `torch.no_grad()`, completely disabling learning from the contrastive regularization.

**Fix:** Removed `torch.no_grad()` wrapper (lines 169-171).

```python
# BEFORE (incorrect - no gradient flow)
with torch.no_grad():
    x_cr_term = IntraContrast_loss(v1, idx_range_N, window=15)
    y_cr_term = IntraContrast_loss(v2, idx_range_M, window=15)

# AFTER (correct - gradients flow through)
x_cr_term = IntraContrast_loss(v1, idx_range_N, window=15)
y_cr_term = IntraContrast_loss(v2, idx_range_M, window=15)
```

### 3. VAVA_loss - Multiple semantic changes

#### 3a. Sinkhorn regularization parameter
**Problem:** Used `reg=l2` instead of `reg=N`.

**Fix:** Restored `reg=N` (line 298).

```python
# BEFORE
dist, T = sink(D, reg=l2, cuda=True, numItermax=maxIter)

# AFTER (correct)
dist, T = sink(D, reg=N, cuda=True, numItermax=maxIter)
```

#### 3b. C_XY term removed
**Problem:** The cross-sequence contrastive term was completely removed.

**Fix:** Restored C_XY term (lines 327-331).

```python
# BEFORE (missing C_XY)
cr_loss = 0.5 * (C_X + C_Y)

# AFTER (correct - includes C_XY)
A = get_A(T, N+1, M+1)
Abar = get_Abar(T, N+1, M+1)
C_XY = torch.sum(A * D - Abar * D)
cr_loss = C_X + C_Y + C_XY
```

#### 3c. Normalization removed
**Problem:** VAVA loss was normalized by `(N * M)`, not in original.

**Fix:** Removed normalization (line 320).

```python
# BEFORE (with normalization)
vava_dis = (dist - l1 * I_T + l2 * KL_T_P) / (N * M)

# AFTER (original formula)
vava_loss = dist - l1 * I_T + l2 * KL_T_P
```

---

## Functions Unchanged (Verified Correct)

- **TCC_loss** - Identical to original
- **GTCC_loss** - Identical to original

---

## Memory Optimizations Preserved

The following memory optimizations were kept as they don't affect loss semantics:

1. `torch.no_grad()` for W_matrix and margin_identity computation in IntraContrast
2. Explicit `del` statements and `torch.cuda.empty_cache()` calls
3. Pairwise threshold checks in `alignment_training_loop.py`

---

## Testing

Both LAV and VAVA were tested for 2 epochs:

```bash
# LAV (GPU 0)
CUDA_VISIBLE_DEVICES=0 python multitask_train.py testLavLoss1 --lav --ego --resnet50 --mcn --ep 2

# VAVA (GPU 1)
CUDA_VISIBLE_DEVICES=1 python multitask_train.py testVavaLoss1 --vava --ego --resnet50 --mcn --ep 2
```

**Results:**
- Both trained successfully without unexpected OOM errors
- Large batches are correctly skipped by the threshold checks in training loop
- Loss values are high (expected with `.sum()` aggregation) but training progresses

---

## Files Modified

- `utils/loss_functions.py` - Main fixes applied

---

## Git Diff Summary

```
- result = (outside_window + inside_window).mean()
+ result = (outside_window + inside_window).sum()

- with torch.no_grad():
-     x_cr_term = IntraContrast_loss(...)
-     y_cr_term = IntraContrast_loss(...)
+ x_cr_term = IntraContrast_loss(...)
+ y_cr_term = IntraContrast_loss(...)

- dist, T = sink(D, reg=l2, ...)
+ dist, T = sink(D, reg=N, ...)

- cr_loss = 0.5 * (C_X + C_Y)
+ A = get_A(T, N+1, M+1)
+ Abar = get_Abar(T, N+1, M+1)
+ C_XY = torch.sum(A * D - Abar * D)
+ cr_loss = C_X + C_Y + C_XY

- vava_dis = (dist - l1 * I_T + l2 * KL_T_P) / (N * M)
+ vava_loss = dist - l1 * I_T + l2 * KL_T_P
```
