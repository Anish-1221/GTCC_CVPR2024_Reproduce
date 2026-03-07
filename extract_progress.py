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

import glob
from collections import OrderedDict
from torch.utils.data import DataLoader
from models.json_dataset import jsondataset_from_splits, data_json_labels_handles
from models.model_multiprong import MultiProngAttDropoutModel
from utils.collate_functions import jsondataset_collate_fn
from utils.train_util import get_config_for_folder, get_data_subfolder_and_extension, get_base_model_deets
from utils.os_util import get_env_variable
from utils.plotter import validate_folder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_splits_from_json(splits_path='/vision/anishn/GTCC_CVPR2024/data_splits.json'):
    with open(splits_path, 'r') as f:
        splits_data = json.load(f)
    return splits_data['splits']


def extract_progress_for_video(model, video_data, times_dict, raw_features_tensor=None):
    """Extract predicted progress values per action segment."""
    model.eval()

    with torch.no_grad():
        outputs_dict = model(video_data)

    # Use 2048-d raw features from disk if provided, else aligned (128-d)
    if raw_features_tensor is not None:
        outputs = raw_features_tensor
    elif 'raw_features' in outputs_dict:
        outputs = outputs_dict['raw_features'][0]
    else:
        outputs = outputs_dict['outputs'][0]
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

        # Convert action label to index for conditioning
        try:
            action_idx = int(action) if action not in ['0', 'SIL', 'background'] else 0
        except (ValueError, TypeError):
            action_idx = 0

        # Extract progress frame by frame for this action segment
        pred_progress = []
        with torch.no_grad():
            for t in range(start, end + 1):
                # Online: use frames from action start to current frame
                partial_segment = outputs[start:t+1].to(device)
                pred_t = progress_head(partial_segment, action_idx=action_idx)
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


def load_model_for_extraction(folder_to_test, config, num_heads, device, ckpt_filename='best_model.pt'):
    """
    Load model for progress extraction.
    Reads progress head architecture from the stored config (ground truth),
    bypassing the dimension-based auto-detection in ckpt_restore_mprong which
    cannot distinguish frame_count from position_encoding (both add 1 dim).
    """
    # Find checkpoint
    ckpt_path = os.path.join(folder_to_test, 'ckpt', ckpt_filename)
    ckpt_handle = ckpt_filename.replace('.pt', '')
    if not os.path.exists(ckpt_path):
        ckpts = glob.glob(os.path.join(folder_to_test, 'ckpt', 'epoch-*.pt'))
        if not ckpts:
            return None, None, None
        ckpts = sorted(ckpts, key=lambda x: int(x.split('epoch-')[-1].split('.')[0]))
        ckpt_path = ckpts[-1]
        ckpt_handle = os.path.basename(ckpt_path).replace('.pt', '')

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    config_obj = checkpoint['config']
    state_dict = checkpoint['model_state_dict']

    # Check if progress head exists in checkpoint
    has_progress_head = any(
        k.startswith('progress_head.') or k.startswith('module.progress_head.')
        for k in state_dict.keys()
    )

    # Build progress_head_config from stored config (not auto-detection from weight dims)
    progress_head_config = None
    if has_progress_head:
        try:
            learnable_cfg = config_obj['PROGRESS_LOSS']['learnable']
        except (KeyError, TypeError):
            learnable_cfg = {}

        architecture = learnable_cfg.get('architecture', 'gru')
        features_source = learnable_cfg.get('features', 'aligned')

        if architecture == 'transformer':
            print(f"[INFO] Loading TransformerProgressHead (from stored config, features={features_source})")
            progress_head_config = {
                'architecture': 'transformer',
                'use_frame_count': learnable_cfg.get('use_frame_count', True),
                'frame_count_max': learnable_cfg.get('frame_count_max', 300.0),
                'transformer_config': learnable_cfg.get('transformer_config', {}),
                'features': features_source,
            }
        elif architecture == 'dilated_conv':
            print(f"[INFO] Loading DilatedConvProgressHead (from stored config, features={features_source})")
            progress_head_config = {
                'architecture': 'dilated_conv',
                'use_frame_count': learnable_cfg.get('use_frame_count', True),
                'frame_count_max': learnable_cfg.get('frame_count_max', 300.0),
                'dilated_conv_config': learnable_cfg.get('dilated_conv_config', {}),
                'features': features_source,
            }
        else:
            use_pos_enc = learnable_cfg.get('use_position_encoding', False)
            use_frame_count = learnable_cfg.get('use_frame_count', True)
            print(f"[INFO] Loading GRU ProgressHead (pos_enc={use_pos_enc}, frame_count={use_frame_count}, features={features_source})")
            progress_head_config = {
                'architecture': 'gru',
                'hidden_dim': learnable_cfg.get('hidden_dim', 64),
                'use_gru': learnable_cfg.get('use_gru', True),
                'use_position_encoding': use_pos_enc,
                'use_frame_count': use_frame_count,
                'frame_count_max': learnable_cfg.get('frame_count_max', 300.0),
                'features': features_source,
                # V9 anti-saturation fields
                'use_input_projection': learnable_cfg.get('use_input_projection', False),
                'projection_dim': learnable_cfg.get('projection_dim', 128),
                'output_activation': learnable_cfg.get('output_activation', 'sigmoid'),
                'per_frame_count': learnable_cfg.get('per_frame_count', False),
                # V10 action conditioning + rate-of-change
                'use_action_conditioning': learnable_cfg.get('use_action_conditioning', False),
                'num_actions': learnable_cfg.get('num_actions', 116),
                'action_embed_dim': learnable_cfg.get('action_embed_dim', 16),
                'use_rate_of_change': learnable_cfg.get('use_rate_of_change', False),
            }

    # Build model
    base_model_class, base_model_params = get_base_model_deets(config_obj)
    arch = config_obj['ARCHITECTURE']
    if 'drop_layers' in arch:
        model = MultiProngAttDropoutModel(
            base_model_class=base_model_class,
            base_model_params=base_model_params,
            output_dimensionality=config_obj['OUTPUT_DIMENSIONALITY'],
            num_heads=num_heads,
            dropping=config_obj['LOSS_TYPE']['GTCC'],
            attn_layers=arch['attn_layers'],
            drop_layers=arch['drop_layers'],
            use_progress_head=has_progress_head,
            progress_head_config=progress_head_config,
        ).to(device)
    else:
        model = MultiProngAttDropoutModel(
            base_model_class=base_model_class,
            base_model_params=base_model_params,
            output_dimensionality=config_obj['OUTPUT_DIMENSIONALITY'],
            num_heads=num_heads,
            dropping=False,
            attn_layers=arch['attn_layers'],
            use_progress_head=has_progress_head,
            progress_head_config=progress_head_config,
        ).to(device)

    # Handle DDP checkpoint (remove 'module.' prefix)
    if list(state_dict.keys())[0].startswith('module.'):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint.get('epoch', 0), ckpt_handle


def main():
    parser = argparse.ArgumentParser(description='Extract progress values from learnable models')
    parser.add_argument('-f', '--folder', required=True, help='Experiment folder path')
    parser.add_argument('--max_videos', type=int, default=None, help='Max videos per task (for testing)')
    parser.add_argument('--ckpt', type=str, default='best_model.pt', help='Checkpoint filename (default: best_model.pt)')
    parser.add_argument('--raw_features_path', type=str, default=None,
        help='Path to folder containing 2048-d raw feature .npy files (overrides model raw_features)')
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
    model, epoch, ckpt_handle = load_model_for_extraction(
        folder_to_test=folder_to_test,
        config=config,
        num_heads=num_heads,
        device=device,
        ckpt_filename=args.ckpt
    )

    if model is None:
        print("[ERROR] Failed to load model")
        return

    model.eval()
    print(f"Model loaded: use_progress_head={model.use_progress_head}")

    if not model.use_progress_head:
        print("[ERROR] This model doesn't have a learnable ProgressHead")
        return

    # Resolve raw features path (from CLI flag or stored config)
    raw_features_path = args.raw_features_path
    if raw_features_path is None:
        raw_features_path = getattr(config, 'RAW_FEATURES_PATH', None)
    if raw_features_path:
        print(f"Using 2048-d raw features from: {raw_features_path}")

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

            # Load 2048-d raw features from disk if configured
            raw_feat_tensor = None
            if raw_features_path:
                feat_file = os.path.join(raw_features_path, f'{video_name}.npy')
                if os.path.exists(feat_file):
                    raw_feat_tensor = torch.from_numpy(np.load(feat_file)).float().to(device)

            # Extract progress
            result = extract_progress_for_video(model, video_data, times_dict, raw_features_tensor=raw_feat_tensor)

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
