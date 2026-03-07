# Evaluate Cumulative L2 v2 Models at 4fps

## Context
We want to evaluate existing 1fps-trained cumulative L2 models (output_l2_progress_v2) using 4fps features to see if more frames per action improves OGPE. Must NOT overwrite existing 1fps results.

## The Problem
The existing pipeline (`generate_aligned_features.py` → `calculate_action_means.py` → `eval.py`) saves to fixed paths:
- `{exp_folder}/aligned_features/` (would overwrite 1fps)
- `{exp_folder}/action_means.json` (would overwrite 1fps)
- `{exp_folder}/EVAL_action_level/` (would overwrite 1fps)

## Approach
Create a single new script `eval_4fps_cumulative.py` that reuses existing code but saves to separate paths:
- `{exp_folder}/aligned_features_4fps/`
- `{exp_folder}/action_means_4fps.json`
- `{exp_folder}/EVAL_action_level_4fps/best_model/ogpe_4fps.csv`

### What the script does (3 phases):
1. **Generate 4fps aligned features**: Load 1fps model checkpoint, run on 4fps raw features → save to `aligned_features_4fps/`
2. **Calculate 4fps action means**: Compute geodesic means from 4fps aligned features using 4fps ProTAS ground truth → save to `action_means_4fps.json`
3. **Evaluate OGPE at 4fps**: Load 4fps aligned features, compute cumulative L2 normalized by 4fps action means, compare to 4fps ground truth → save to `ogpe_4fps.csv`

### Reuses from existing code:
- `generate_aligned_features.py`: `initialize_model()`, `extract_features()` logic
- `calculate_action_means.py`: `get_segments_from_txt()`, `calculate_geodesic_distance()` logic
- `utils/evaluation_action_level.py`: `parse_groundtruth_to_segments()`, `get_trueprogress_per_action()`, cumulative L2 OGPE logic (lines 407-434)
- `utils/tensorops.py`: `get_cum_matrix()`, `get_trueprogress_per_action()`
- `data_splits.json`: fixed test splits

### Paths:
- 4fps raw features: `/vision/anishn/GTCC_Data_Processed_4fps/egoprocel/frames/`
- 4fps ProTAS GT: `/vision/anishn/ProTAS/data_4fps/egoprocel_subset*/groundTruth/`
- Models: `/vision/anishn/GTCC_CVPR2024/output_l2_progress_v2/multi-task-setting_val/V1___*_egoprocel/`

### CLI:
```bash
conda activate /vision/anishn/envs/vidal
cd /vision/anishn/GTCC_CVPR2024

# Run for all 4 models
python eval_4fps_cumulative.py --exp_folder output_l2_progress_v2/multi-task-setting_val/V1___GTCC_egoprocel
python eval_4fps_cumulative.py --exp_folder output_l2_progress_v2/multi-task-setting_val/V1___tcc_egoprocel
python eval_4fps_cumulative.py --exp_folder output_l2_progress_v2/multi-task-setting_val/V1___LAV_egoprocel
python eval_4fps_cumulative.py --exp_folder output_l2_progress_v2/multi-task-setting_val/V1___VAVA_egoprocel
```

## File to create:
- `/vision/anishn/GTCC_CVPR2024/eval_4fps_cumulative.py`

## Verification:
- Check that `aligned_features_4fps/` has 325 .npy files (same as 1fps)
- Check that `action_means_4fps.json` exists and has reasonable values
- Check that `ogpe_4fps.csv` exists with per-task results
- Verify 1fps files are untouched (aligned_features/, action_means.json, EVAL_action_level/)
