#!/usr/bin/env python
"""
Unified Aligned Feature Generation Script - STORES IN EXPERIMENT FOLDER

Usage:
    python generate_aligned_features.py --exp_folder /path/to/model

Output: {exp_folder}/aligned_features/*.npy

Examples:
    # Just specify the experiment folder - FPS and model type auto-detected!
    python generate_aligned_features.py --exp_folder output_4fps_val/multi-task-setting_val/V1___GTCC_egoprocel.4fps

    # Output will be at: output_4fps_val/multi-task-setting_val/V1___GTCC_egoprocel.4fps/aligned_features/
"""
import argparse
import os
import torch
import numpy as np
import glob
from tqdm import tqdm

from utils.train_util import get_config_for_folder, ckpt_restore_mprong
from utils.logging import configure_logging_format

logger = configure_logging_format()


def get_input_path(fps: str):
    """Get input features path based on FPS"""
    return f'/vision/anishn/GTCC_Data_Processed_{fps}/egoprocel/frames/'


def detect_model_type(config):
    """Detect model type from config"""
    if config.LOSS_TYPE.get('GTCC', False):
        return 'gtcc'
    elif config.LOSS_TYPE.get('tcc', False):
        return 'tcc'
    elif config.LOSS_TYPE.get('VAVA', False):
        return 'vava'
    elif config.LOSS_TYPE.get('LAV', False):
        return 'lav'
    return 'gtcc'  # default


def initialize_model(exp_folder: str, device):
    """Initialize and load the trained model"""
    logger.info(f"[1/4] Loading config from {exp_folder}")
    config = get_config_for_folder(exp_folder)

    model_type = detect_model_type(config)
    logger.info(f"[DEBUG] Detected model type: {model_type.upper()}")
    logger.info(f"[DEBUG] Config LOSS_TYPE: {config.LOSS_TYPE}")

    # Find checkpoint
    ckpt_path = os.path.join(exp_folder, 'ckpt', 'best_model.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(exp_folder, 'ckpt', 'epoch-50.pt')
    if not os.path.exists(ckpt_path):
        logger.warning(f"[WARNING] No checkpoint found, searching for any...")
        ckpt_files = sorted(glob.glob(os.path.join(exp_folder, 'ckpt', 'epoch-*.pt')))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoints found in {exp_folder}/ckpt/")
        ckpt_path = ckpt_files[-1]

    logger.info(f"[INFO] Using checkpoint: {ckpt_path}")
    logger.info(f"[2/4] Loading checkpoint from {ckpt_path}")

    # Detect number of heads from checkpoint
    ckpt_data = torch.load(ckpt_path, map_location='cpu')
    model_state = ckpt_data.get('model', ckpt_data)
    head_keys = [k for k in model_state.keys() if k.startswith('head_models.')]
    num_heads = max([int(k.split('.')[1]) for k in head_keys]) + 1 if head_keys else 16

    logger.info(f"[DEBUG] Detected {num_heads} heads from checkpoint")

    # GTCC uses dropout layers, others don't
    use_dropout = (model_type == 'gtcc')

    logger.info(f"[3/4] Restoring model with {num_heads} heads (dropout={use_dropout})...")
    model, epoch, loss, _, _ = ckpt_restore_mprong(
        ckpt_path,
        num_heads=num_heads,
        dropout=use_dropout,
        device=device
    )

    logger.info(f"[DEBUG] Restored model from epoch {epoch}")
    model.eval()
    logger.info(f"[4/4] Model loaded and set to eval mode")
    return model, config, model_type


def extract_features(model, input_dir: str, output_dir: str, device):
    """Extract aligned features from raw features using trained model"""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Get list of feature files
    logger.info(f"Scanning for .npy files in {input_dir}")
    feat_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.npy')])
    logger.info(f"Found {len(feat_files)} feature files")

    if len(feat_files) == 0:
        logger.error(f"No .npy files found in {input_dir}")
        return 0, 0

    logger.info(f"First 3 files: {feat_files[:3]}")

    successful, failed = 0, 0

    with torch.no_grad():
        for idx, f_name in enumerate(tqdm(feat_files, desc="Extracting Aligned Features")):
            try:
                # Load raw features
                feat_path = os.path.join(input_dir, f_name)
                raw_feats = np.load(feat_path)

                if idx == 0:
                    logger.info(f"\n[DEBUG] First file: {f_name}")
                    logger.info(f"  Raw features shape: {raw_feats.shape}")

                # Convert to tensor and move to device
                feat_tensor = torch.from_numpy(raw_feats).float().to(device)

                # Model expects a LIST of tensors (batch of videos)
                out_dict = model([feat_tensor])

                # outputs is a LIST of tensors (one per video in batch)
                aligned_feats = out_dict['outputs'][0].cpu().numpy()

                if idx == 0:
                    logger.info(f"  Aligned features shape: {aligned_feats.shape}")

                # Save to output directory
                np.save(os.path.join(output_dir, f_name), aligned_feats)
                successful += 1

            except Exception as e:
                failed += 1
                logger.error(f"[ERROR] Failed to process {f_name}: {str(e)}")
                if idx == 0:
                    import traceback
                    traceback.print_exc()

    logger.info(f"\n{'='*80}")
    logger.info(f"Extraction complete!")
    logger.info(f"  Successful: {successful}/{len(feat_files)}")
    logger.info(f"  Failed: {failed}/{len(feat_files)}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"{'='*80}")

    return successful, failed


def main():
    parser = argparse.ArgumentParser(
        description='Generate aligned features (stored in experiment folder)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--exp_folder', type=str, required=True,
                        help='Path to experiment folder containing ckpt/')
    parser.add_argument('--fps', type=str, default=None,
                        help='FPS of the data (auto-detected from folder name if not specified)')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-detect FPS from folder name
    fps = args.fps
    if fps is None:
        fps = '4fps' if '4fps' in args.exp_folder else '1fps'

    input_dir = get_input_path(fps)
    output_dir = os.path.join(args.exp_folder, 'aligned_features')  # INSIDE experiment folder!

    logger.info("="*80)
    logger.info("Aligned Feature Generation (Experiment Folder Storage)")
    logger.info("="*80)
    logger.info(f"Using device: {device}")
    logger.info(f"FPS: {fps}")
    logger.info(f"Experiment folder: {args.exp_folder}")
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")

    # Initialize model
    model, config, model_type = initialize_model(args.exp_folder, device)

    # Extract features
    successful, failed = extract_features(model, input_dir, output_dir, device)

    if successful > 0:
        logger.info(f"\nAligned features saved to: {output_dir}")
        logger.info(f"Next step: python calculate_action_means.py --exp_folder {args.exp_folder}")


if __name__ == "__main__":
    main()
