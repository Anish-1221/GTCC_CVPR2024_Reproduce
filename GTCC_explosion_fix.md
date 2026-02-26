# GTCC Loss Explosion Fix

## Problem
Training crashes due to numerical instability in GTCC alignment loss that causes loss explosion and NaN propagation.

### Error Trace Analysis
```
Batch 46: Loss: 251,214  | Align: 41,338    ✓ Normal
Batch 47: Loss: 199,518  | Align: 13,998    ✓ Normal
Batch 48: Loss: 11,955,280 | Align: 11,774,341  ⚠️ SPIKE (300x normal!)
Batch 49: Loss: 1.33e18  | Align: 1.33e18   ❌ OVERFLOW
Batch 50: tensor(nan) → Exception in GMM    💥 CRASH
```

### What Happened
1. **Batch 48 (Salad.cmu)**: GTCC loss computed abnormally high alignment loss (11.7M instead of ~40k)
2. **Gradient explosion**: Even with gradient clipping, extreme loss corrupted model weights
3. **Batch 49 (Sandwich.cmu)**: With corrupted weights, loss overflowed to 1.33e18
4. **Batch 50 (PastaSalad.egtea)**: GMM fitting received NaN from corrupted weights → crash

### Root Cause
In `utils/loss_functions.py` GTCC_loss:
```python
ALPHA_exp = torch.exp(-cdist / softmax_temp)  # softmax_temp=0.1
```
With `softmax_temp=0.1`, the exponential is extremely sensitive. Occasional numerical issues cause extreme loss values.

---

## The Fix

### File Modified
`models/alignment_training_loop.py`

### Code Added (after line 367)
```python
loss = loss_dict['total_loss']

# [FIX] Skip batches with extreme loss to prevent weight corruption
# Normal GTCC loss is ~40k-250k. Values > 1e7 indicate numerical instability.
MAX_SAFE_LOSS = 1e7
if loss.item() > MAX_SAFE_LOSS:
    logger.warning(f"[EXTREME LOSS] {loss.item():.2e} > {MAX_SAFE_LOSS:.2e} - skipping batch to protect weights")
    del output_dict, loss_dict, loss
    gc.collect()
    torch.cuda.empty_cache()
    skipped_exception += 1
    continue
```

### Why This Works
By skipping batch 48 before `loss.backward()`, we prevent the weight corruption that caused the cascade failure. With healthy weights, subsequent batches compute normally.

---

## Verification
After applying fix, run:
```bash
CUDA_VISIBLE_DEVICES=4 python multitask_train.py 1 --gtcc --egoprocel --resnet --mcn --progress_loss learnable --progress_lambda 1000000.0
```

Watch for `[EXTREME LOSS]` warnings - these indicate batches that were safely skipped.

---

## Note
The progress loss with lambda=1M is working correctly (values ~0.18-0.21). The crash was unrelated to progress loss - it was caused by GTCC alignment loss numerical instability.
