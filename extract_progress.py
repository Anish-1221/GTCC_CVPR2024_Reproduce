#!/usr/bin/env python
"""
Extract Predicted Progress Values from Learnable ProgressHead Models

Usage:
    python extract_progress.py -f output_learnable_progress_v2/multi-task-setting_val/V1___tcc_egoprocel
    python extract_progress.py -f output_learnable_progress_v2/multi-task-setting_val/V1___tcc_egoprocel --max_videos 2
"""

import argparse
import json
import os
import torch
import numpy as np

from torch.utils.data import DataLoader
from models.json_dataset import jsondataset_from_splits, data_json_labels_handles
from utils.collate_functions import jsondataset_collate_fn
from utils.train_util import get_config_for_folder, get_data_subfolder_and_extension
from utils.ckpt_save import get_ckpt_for_eval
from utils.os_util import get_env_variable
from utils.plotter import validate_folder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_splits_from_json(splits_path='/vision/anishn/GTCC_CVPR2024/data_splits.json'):
    with open(splits_path, 'r') as f:
        splits_data = json.load(f)
    return splits_data['splits']


def extract_progress_for_video(model, video_data, times_dict):
    """Extract predicted progress values per action segment."""
    model.eval()

    with torch.no_grad():
        outputs_dict = model(video_data)

    outputs = outputs_dict['outputs'][0]  # First video in batch
    progress_head = outputs_dict.get('progress_head')

    if progress_head is None:
        return None

    num_frames = len(outputs)

    # Get action segments from times_dict
    actions = times_dict['step']
    start_frames = times_dict['start_frame']
    end_frames = times_dict['end_frame']

    segments = []
    for action, start, end in zip(actions, start_frames, end_frames):
        # Clamp to video length
        start = max(0, min(start, num_frames - 1))
        end = max(0, min(end, num_frames - 1))

        if action in ['SIL', 'background'] or start >= num_frames:
            continue

        # Extract progress frame by frame for this action segment
        pred_progress = []
        with torch.no_grad():
            for t in range(start, end + 1):
                # Online: use frames from action start to current frame
                partial_segment = outputs[start:t+1].to(device)
                pred_t = progress_head(partial_segment)
                pred_progress.append(round(float(pred_t.item()), 4))

        # Calculate deltas (frame-to-frame improvement)
        deltas = [pred_progress[0]] if pred_progress else []
        for i in range(1, len(pred_progress)):
            deltas.append(round(pred_progress[i] - pred_progress[i-1], 4))

        segments.append({
            'action': action,
            'start': start,
            'end': end,
            'length': end - start + 1,
            'pred_progress': pred_progress,
            'deltas': deltas
        })

    return {
        'num_frames': num_frames,
        'segments': segments
    }


def main():
    parser = argparse.ArgumentParser(description='Extract progress values from learnable models')
    parser.add_argument('-f', '--folder', required=True, help='Experiment folder path')
    parser.add_argument('--max_videos', type=int, default=None, help='Max videos per task (for testing)')
    args = parser.parse_args()

    folder_to_test = args.folder
    output_folder = os.path.join(folder_to_test, 'view_progress')
    validate_folder(output_folder)

    print(f"Loading model from: {folder_to_test}")
    print(f"Output folder: {output_folder}")

    # Load config
    config = get_config_for_folder(folder_to_test)

    # Load data structure
    dset_json_folder = get_env_variable('JSON_DPATH')
    data_structure = data_json_labels_handles(dset_json_folder, dset_name=config.DATASET_NAME)
    TASKS = list(data_structure.keys())

    # Load test splits
    splits_dict = load_splits_from_json()

    # Get data folder
    data_subfolder_name, datafile_extension = get_data_subfolder_and_extension(
        architecture=config.BASEARCH.ARCHITECTURE
    )
    data_folder = f'{config.DATAFOLDER}/{data_subfolder_name}'

    # Load model
    num_heads = config['ARCHITECTURE']['num_heads'] or len(TASKS)
    model, epoch, ckpt_handle = get_ckpt_for_eval(
        ckpt_parent_folder=folder_to_test,
        config=config,
        num_heads=num_heads,
        device=device
    )

    if model is None:
        print("[ERROR] Failed to load model")
        return

    model.eval()
    print(f"Model loaded: use_progress_head={model.use_progress_head}")

    if not model.use_progress_head:
        print("[ERROR] This model doesn't have a learnable ProgressHead")
        return

    # Process each task
    for task in TASKS:
        print(f"\n{'='*50}")
        print(f"Task: {task}")
        print(f"{'='*50}")

        # Create task subfolder
        task_folder = os.path.join(output_folder, task)
        validate_folder(task_folder)

        # Load test set
        test_set = jsondataset_from_splits(
            task=task,
            task_json=data_structure[task],
            data_folder=data_folder,
            splits_dict=splits_dict,
            split_type='test',
            extension=datafile_extension,
            lazy_loading=config.LAZY_LOAD
        )

        test_dl = DataLoader(
            test_set,
            batch_size=1,
            collate_fn=jsondataset_collate_fn,
            drop_last=False,
            shuffle=False
        )

        for idx, (video_data, video_info) in enumerate(test_dl):
            if args.max_videos and idx >= args.max_videos:
                break

            # Get video info (times_dict)
            times_dict = video_info[0] if isinstance(video_info, list) else video_info
            video_name = times_dict['name']

            # Load data if lazy loading (strings are file paths)
            loaded_data = []
            for v in video_data:
                if isinstance(v, str):
                    v = torch.from_numpy(np.load(v)).float()
                loaded_data.append(v.to(device))
            video_data = loaded_data

            # Extract progress
            result = extract_progress_for_video(model, video_data, times_dict)

            if result:
                result['video_name'] = video_name

                # Save to JSON
                output_path = os.path.join(task_folder, f'{video_name}.json')
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)

                # Print sample
                print(f"\n  {video_name}: {result['num_frames']} frames, {len(result['segments'])} actions")
                if result['segments']:
                    seg = result['segments'][0]
                    print(f"    First action: {seg['action']} (frames {seg['start']}-{seg['end']})")
                    print(f"    Progress (first 5): {seg['pred_progress'][:5]}")
                    print(f"    Deltas (first 5): {seg['deltas'][:5]}")

    print(f"\n{'='*50}")
    print(f"Done! Results saved to: {output_folder}")


if __name__ == '__main__':
    main()
