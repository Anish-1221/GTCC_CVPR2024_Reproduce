# Naive Baseline OGPE Evaluation Script

## Context
We need naive baselines for action-level OGPE to compare against GTCC learned/cumulative progress models. The baselines predict a constant value (0, 0.25, 0.5, 0.75, 1.0) or random [0,1] for all action frames, then compute MAE against ground truth.

## Approach
Create a single new script `eval_naive_baselines.py` in `/vision/anishn/GTCC_CVPR2024/`. No model needed — just ground truth files and test split info.

### What to reuse (no modifications to existing files):
- `utils/tensorops.py`: `get_trueprogress_per_action()` for ground truth
- `data_splits.json`: fixed test split video names
- `dset_jsons/egoprocel.json`: task structure (via `data_json_labels_handles`)
- Ground truth parsing: same `parse_groundtruth_to_segments()` logic from `evaluation_action_level.py` (lines 267-295)
- ProTAS ground truth paths: `/vision/anishn/ProTAS/data_1fps/` with 5 egoprocel subsets (same as `evaluation_action_level.py` lines 239, 349-355)
- Video length: number of lines in ground truth `.txt` file

### Ground truth is ACTION-LEVEL (not video-level):
- Uses `get_trueprogress_per_action(tdict)` — each action gets independent 0→1 linear ramp
- Background/SIL frames are masked out (progress = 0, excluded from error)
- This matches `evaluation_action_level.py`'s OGPE computation exactly

### How it works:
1. Load test split video names from `data_splits.json`
2. Load task structure from `dset_jsons/egoprocel.json`
3. For each task, for each test video:
   - Find ground truth `.txt` in ProTAS subsets (same 5-subset search as evaluation_action_level.py)
   - Get video length = number of lines in .txt
   - Parse segments via `parse_groundtruth_to_segments()`
   - Build tdict, compute `get_trueprogress_per_action(tdict)`
   - Build action_mask (exclude SIL/background, same logic as evaluation_action_level.py lines 437-442)
   - For each baseline (0, 0.25, 0.5, 0.75, 1.0, random):
     - Set `pred_progress = constant` everywhere (or random uniform [0,1])
     - Compute `torch.abs(true_progress - pred_progress)[action_mask].mean()`
4. Report mean OGPE per baseline per task, and overall mean across tasks
5. Random baseline: run with multiple seeds, report mean ± std
6. Save results to `output_baselines/` folder — one `ogpe.csv` per baseline

### Output:
- Folder: `/vision/anishn/GTCC_CVPR2024/output_baselines/`
- Each baseline gets its own subfolder with `ogpe.csv`:
  - `output_baselines/constant_0.0/ogpe.csv`
  - `output_baselines/constant_0.25/ogpe.csv`
  - `output_baselines/constant_0.5/ogpe.csv`
  - `output_baselines/constant_0.75/ogpe.csv`
  - `output_baselines/constant_1.0/ogpe.csv`
  - `output_baselines/random/ogpe.csv`
- CSV format matches existing eval output: columns for task, ogpe, num_videos

### CLI:
```
python eval_naive_baselines.py
```
No arguments needed — everything is hardcoded to match the existing evaluation pipeline.

## Files to create:
- `/vision/anishn/GTCC_CVPR2024/eval_naive_baselines.py`

## Verification:
- Run the script and confirm it produces OGPE values for all 6+ baselines
- Check `output_baselines/*/ogpe.csv` files exist with correct values
- Sanity checks: constant 0.5 should give ~0.25, constant 0 and 1 should give ~0.5
