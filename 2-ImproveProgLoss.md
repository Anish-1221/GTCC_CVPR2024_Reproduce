# Plan: Fix Progress Head — Dense Supervision + Remove Length Shortcut

## Context

All progress head variants (GRU V6, Transformer V7) produce identical step-function predictions across every video:
- V7: [0.67, 1.0, 1.0, 1.0, ...]
- V6: [0.74, 1.0, 1.0, 1.0, ...]

**Root cause**: The cumulative-segment architecture creates a sequence-length shortcut. The model learns "1 frame → ~0.7, 2+ frames → 1.0" without using embedding content. The `frame_count` feature explicitly hands it the length. No loss formulation can fix this.

**ProTAS comparison**: ProTAS works because of (1) dilated convs for multi-scale context, (2) dense per-frame MSE supervision, (3) joint action classification. In online mode (`predict_online`, line 250), ProTAS also feeds growing prefixes — same as us. The key difference is **supervision density**.

---

## Approach: Keep Current Architectures, Fix Supervision

### Change 1: Add `dense_output` mode to all three progress heads

Add a `dense_output=False` parameter to `forward()`. When True, apply the output MLP to **all timesteps** (not just the last). Since all three architectures are already causal, each position's output only depends on past frames.

#### ProgressHead (GRU) — `model_multiprong.py:127`

Currently uses only `h_n` (final hidden state). Change to use `outputs` (all hidden states):

```python
def forward(self, segment_embeddings, dense_output=False):
    # ... existing feature concat (minus frame_count) ...
    x = x.unsqueeze(0)  # (1, T, D)
    outputs, h_n = self.gru(x)  # outputs: (1, T, H), h_n: (1, 1, H)

    if dense_output:
        per_frame = self.fc(outputs.squeeze(0))  # (T, H) → (T, 1)
        return per_frame.squeeze(-1)              # (T,)
    else:
        progress = self.fc(h_n.squeeze())
        return progress.squeeze()
```

**Key**: GRU is forward-only (unidirectional). `outputs[t]` = hidden state after processing frames 0..t. Applying FC to `outputs[t]` gives the same result as feeding only frames 0..t and using `h_n`.

#### TransformerProgressHead — `model_multiprong.py:249`

Currently uses only last token `x[0, -1, :]`. Change to use all positions:

```python
def forward(self, segment_embeddings, dense_output=False):
    # ... existing code through transformer layers ...
    # x shape: (1, T, d_model) after all layers

    if dense_output:
        per_frame = self.output_mlp(x.squeeze(0))  # (T, d_model) → (T, 1)
        return per_frame.squeeze(-1)                 # (T,)
    else:
        final = x[0, -1, :]
        progress = self.output_mlp(final)
        return progress.squeeze()
```

**Key**: Causal mask (line 279: `torch.triu(..., diagonal=1).bool()`) ensures position t only attends to positions ≤ t. So `x[0, t, :]` after all layers is the same whether we feed T frames or just t+1 frames.

#### DilatedConvProgressHead — `model_multiprong.py:437`

Currently uses only last position `x[0, :, -1]`. Change to use all positions:

```python
def forward(self, segment_embeddings, dense_output=False):
    # ... existing code through dilated blocks ...
    # x shape: (1, hidden_dim, T) after all blocks

    if dense_output:
        per_frame = self.output_mlp(x.squeeze(0).transpose(0, 1))  # (T, H) → (T, 1)
        return per_frame.squeeze(-1)                                  # (T,)
    else:
        final = x[0, :, -1]
        progress = self.output_mlp(final)
        return progress.squeeze()
```

**Key**: `CausalDilatedResidualBlock` uses left-only padding (line 516: `F.pad(x, (self.causal_pad, 0))`). Output at position t only depends on frames 0..t. All T positions are already computed causally — we just need to read them all instead of just the last one.

### Change 2: Remove `frame_count` feature for dense mode

The `frame_count` broadcasts `log(1+T)/log(1+300)` to all frames — the explicit length shortcut. Disable it when using dense mode.

**All three heads**: When `progress_loss_mode == 'dense'`, set `use_frame_count=False` in config. The heads already support this via their `use_frame_count` parameter.

- `ProgressHead`: line 140-146 — skip frame_count concat
- `TransformerProgressHead`: line 261-267 — skip frame_count concat
- `DilatedConvProgressHead`: line 449-455 — skip frame_count concat

### Change 3: Add `_progress_loss_dense()` in loss_entry.py

New loss mode that iterates over ALL non-background actions and computes dense MSE:

```python
def _progress_loss_dense(output_dict, times, progress_head, cfg):
    """Dense per-frame MSE: feed full action, supervise every frame."""
    total_loss = 0.0
    num_actions = 0

    for vid_idx in range(len(output_dict['outputs'])):
        vid_emb = output_dict['outputs'][vid_idx]
        vid_times = times[vid_idx]
        T = vid_emb.shape[0]

        for step, start, end in zip(
            vid_times['step'], vid_times['start_frame'], vid_times['end_frame']
        ):
            if step in BACKGROUND_LABELS:
                continue
            start_c = max(0, min(start, T - 1))
            end_c = max(start_c, min(end, T - 1))
            action_emb = vid_emb[start_c : end_c + 1]
            L = action_emb.shape[0]
            if L < 2:
                continue

            pred = progress_head(action_emb, dense_output=True)  # (L,)
            gt = torch.arange(1, L+1, device=pred.device, dtype=pred.dtype) / L
            total_loss = total_loss + F.mse_loss(pred, gt)
            num_actions += 1

    if num_actions == 0:
        return 0.0, 0
    return total_loss / num_actions, num_actions
```

Add `'dense'` branch to `_compute_learnable_progress_loss()` dispatcher.

**No sampling, no monotonicity penalty, no endpoint loss, no weighting.** Dense supervision handles all of that naturally.

### Change 4: CLI + config updates

**`utils/parser_util.py`**: Add `'dense'` to `--progress_loss_mode` choices.

**`configs/generic_config.py`**: Add `'dense'` to documented valid loss modes.

**`configs/entry_config.py`**: When `progress_loss_mode == 'dense'`, override `use_frame_count = False`:
```python
if progress_loss_mode == 'dense':
    CONFIG.PROGRESS_LOSS['learnable']['use_frame_count'] = False
```

---

## Inference (extract_progress.py)

**No change needed.** Still feeds growing prefix frame-by-frame with `dense_output=False` (default):
```python
for t in range(start, end + 1):
    partial_segment = outputs[start:t+1]
    pred_t = progress_head(partial_segment)  # single scalar
```

This gives the same result as `dense_output=True` on the full segment because all three architectures are causal.

---

## Fallback: DenseProgressHead with Dilated Convolutions

If the existing architectures can't learn gradual curves even with dense supervision, implement a ProTAS-inspired head with dilated convs + GRU (combined). See `DenseProgressHead` design in the earlier plan. Reuse `DilatedResidualLayer` from `models/protas_model.py` (lines 107-126).

---

## Files to Modify

| File | Change |
|------|--------|
| `models/model_multiprong.py` | Add `dense_output` param to all 3 heads' `forward()` |
| `utils/loss_entry.py` | Add `_progress_loss_dense()`, add to dispatcher |
| `utils/parser_util.py` | Add `'dense'` to `--progress_loss_mode` choices |
| `configs/generic_config.py` | Document `'dense'` mode |
| `configs/entry_config.py` | Override `use_frame_count=False` for dense mode |

---

## Verification

1. Train (progress-head-only, frozen encoder):
   ```bash
   CUDA_VISIBLE_DEVICES=5 python multitask_train.py 1 --gtcc --egoprocel --resnet --mcn \
     --progress_loss learnable --progress_arch transformer --progress_lambda 500000.0 \
     --train_progress_only --reinit_progress_head \
     --alignment_checkpoint output_learnable_progress_v7/multi-task-setting_val/V1___GTCC_egoprocel/ckpt/best_model.pt \
     --progress_lr 0.001 --progress_epochs 50 \
     --progress_loss_mode dense
   ```

2. `extract_progress.py` — expect:
   - Smooth gradual 0→1 curves (not step functions)
   - **Different** values across different videos/actions
   - Monotonically increasing within each action

3. Sanity checks:
   - Log confirms `use_frame_count=False`
   - Training loss is MSE-scale (not L1-scale)
   - Different videos produce different progress curves
