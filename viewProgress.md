# Standalone Script: Extract Progress Values from Learnable Models

## Goal
Create a standalone script that extracts and saves progress arrays from learnable ProgressHead models, showing the improvement of progress values from one frame to the next.

## Output Format
For each video, save:
- `pred_progress`: Predicted progress values [0, 1] for each frame
- `true_progress`: Ground truth progress values [0, 1] for each frame
- `segments`: Action segment info (name, start, end)
- Frame-by-frame deltas to show improvement

## Script Location
`/vision/anishn/GTCC_CVPR2024/extract_progress.py`

## Usage
```bash
python extract_progress.py -f output_learnable_progress_v2/multi-task-setting_val/V1___tcc_egoprocel --output progress_output/
```

## Key Logic (from evaluation_action_level.py lines 396-406)
```python
# For each frame t in an action segment:
for t in range(start, end + 1):
    partial_segment = outputs[start:t+1]  # frames 1, then 1-2, then 1-2-3...
    pred_progress_t = progress_head(partial_segment)
```

## Output Files
```
progress_output/
├── brownie/
│   ├── video1.json       # Per-video progress arrays
│   ├── video2.json
│   └── ...
├── meccano/
│   └── ...
└── summary.json          # Aggregate statistics
```

## JSON Format per Video
```json
{
  "video_name": "P01_01",
  "segments": [
    {
      "action": "add_brownie_mix",
      "start": 0,
      "end": 45,
      "pred_progress": [0.02, 0.05, 0.08, ...],
      "true_progress": [0.0, 0.022, 0.044, ...],
      "deltas": [0.03, 0.03, 0.02, ...]  // frame-to-frame improvement
    }
  ]
}
```

## Dependencies
- Uses existing `ckpt_restore_mprong` for model loading (already has backward compat)
- Uses existing `data_splits.json` for test set
- Uses existing ground truth parsing from evaluation code
