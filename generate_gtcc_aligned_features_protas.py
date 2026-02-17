"""
Generate GTCC aligned features (128-d) for all ProTAS videos.
Uses the trained GTCC model to transform raw features (1024x14x14) to aligned embeddings (128-d).

Input:  /vision/anishn/GTCC_Data_Processed_1fps/GTCC_1fps_protas/frames/
Output: /vision/anishn/GTCC_Data_Processed_1fps/GTCC_1fps_protas/aligned_features/

Usage:
    python generate_gtcc_aligned_features_protas.py [--gpu GPU_ID]
"""

import os
import argparse

# Parse arguments before importing torch to set GPU
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU ID to use (default: 1)')
args = parser.parse_args()

# Set GPU before importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import torch
import numpy as np
import glob
from tqdm import tqdm

# Import project utilities
from models.model_multiprong import MultiProngAttDropoutModel
from utils.train_util import get_base_model_deets, get_config_for_folder, ckpt_restore_mprong
from utils.logging import configure_logging_format

logger = configure_logging_format()

# --- CONFIGURATION ---
EXP_FOLDER = '/vision/anishn/GTCC_CVPR2024/output/multi-task-setting/V1___GTCC_egoprocel'
INPUT_FEAT_DIR = '/vision/anishn/GTCC_Data_Processed_1fps/GTCC_1fps_protas/frames/'
OUTPUT_DIR = '/vision/anishn/GTCC_Data_Processed_1fps/GTCC_1fps_protas/aligned_features/'


def initialize_model(device):
    """Initialize and load the trained GTCC model"""
    logger.info(f"[1/4] Loading config from {EXP_FOLDER}")
    config = get_config_for_folder(EXP_FOLDER)

    logger.info(f"[DEBUG] Config DATASET_NAME: {config.DATASET_NAME}")
    logger.info(f"[DEBUG] Config ARCHITECTURE: {config.ARCHITECTURE}")
    logger.info(f"[DEBUG] Config LOSS_TYPE: {config.LOSS_TYPE}")

    # Get checkpoint path
    ckpt_path = os.path.join(EXP_FOLDER, 'ckpt', 'epoch-50.pt')

    if not os.path.exists(ckpt_path):
        # Fallback: find latest checkpoint
        logger.warning(f"[WARNING] {ckpt_path} not found, searching for any checkpoint...")
        ckpt_files = sorted(glob.glob(os.path.join(EXP_FOLDER, 'ckpt', 'epoch-*.pt')))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoints found in {EXP_FOLDER}/ckpt/")
        ckpt_path = ckpt_files[-1]
        logger.info(f"[INFO] Using checkpoint: {ckpt_path}")

    logger.info(f"[2/4] Loading checkpoint from {ckpt_path}")

    # Load checkpoint to inspect it
    ckpt_data = torch.load(ckpt_path, map_location='cpu')
    logger.info(f"[DEBUG] Checkpoint keys: {ckpt_data.keys()}")

    # Determine number of heads from checkpoint
    if 'model' in ckpt_data:
        model_state = ckpt_data['model']
    else:
        model_state = ckpt_data

    # Count head models in the state dict
    head_keys = [k for k in model_state.keys() if k.startswith('head_models.')]
    if head_keys:
        num_heads = max([int(k.split('.')[1]) for k in head_keys]) + 1
        logger.info(f"[DEBUG] Detected {num_heads} heads from checkpoint")
    else:
        num_heads = 16  # Default
        logger.warning(f"[WARNING] Could not detect heads from checkpoint, using default: {num_heads}")

    logger.info(f"[3/4] Restoring model with {num_heads} heads...")
    model, epoch, loss, _, _ = ckpt_restore_mprong(
        ckpt_path,
        num_heads=num_heads,
        dropout=config.LOSS_TYPE.get('GTCC', False),
        device=device
    )

    logger.info(f"[DEBUG] Restored model from epoch {epoch} with loss {loss}")
    logger.info(f"[DEBUG] Model type: {type(model)}")
    logger.info(f"[DEBUG] Model has {len(model.head_models)} head models")

    model.eval()
    logger.info(f"[4/4] Model loaded and set to eval mode")
    return model, config


def extract_features():
    """Extract aligned features from raw features using trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Initialize model
    model, config = initialize_model(device)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # Get list of feature files
    logger.info(f"Scanning for .npy files in {INPUT_FEAT_DIR}")
    feat_files = sorted([f for f in os.listdir(INPUT_FEAT_DIR) if f.endswith('.npy')])
    logger.info(f"Found {len(feat_files)} feature files")

    if len(feat_files) == 0:
        logger.error(f"No .npy files found in {INPUT_FEAT_DIR}")
        return

    # Check which files already have aligned features
    existing = set(os.listdir(OUTPUT_DIR))
    to_process = [f for f in feat_files if f not in existing]
    logger.info(f"Already processed: {len(existing)}")
    logger.info(f"To process: {len(to_process)}")

    # Process each file
    successful = 0
    failed = 0

    with torch.no_grad():
        for idx, f_name in enumerate(tqdm(to_process, desc="Generating Aligned Features")):
            try:
                # Load raw features
                feat_path = os.path.join(INPUT_FEAT_DIR, f_name)
                raw_feats = np.load(feat_path)

                if idx == 0:
                    logger.info(f"\n[DEBUG] First file analysis:")
                    logger.info(f"  File: {f_name}")
                    logger.info(f"  Raw features shape: {raw_feats.shape}")
                    logger.info(f"  Data type: {raw_feats.dtype}")

                # Convert to tensor and move to device
                feat_tensor = torch.from_numpy(raw_feats).float().to(device)

                if idx == 0:
                    logger.info(f"  Tensor shape: {feat_tensor.shape}")
                    logger.info(f"  Tensor device: {feat_tensor.device}")

                # Model expects a LIST of tensors
                inputs = [feat_tensor]

                # Forward pass through model
                out_dict = model(inputs)

                if idx == 0:
                    logger.info(f"\n[DEBUG] Model output analysis:")
                    logger.info(f"  Output keys: {out_dict.keys()}")
                    if isinstance(out_dict['outputs'], list):
                        logger.info(f"  Length of outputs list: {len(out_dict['outputs'])}")
                        logger.info(f"  Shape of outputs[0]: {out_dict['outputs'][0].shape}")

                # Extract aligned features
                aligned_feats = out_dict['outputs'][0].cpu().numpy()

                if idx == 0:
                    logger.info(f"  Aligned features shape: {aligned_feats.shape}")
                    logger.info(f"  Aligned features dtype: {aligned_feats.dtype}")

                # Save to output directory
                save_path = os.path.join(OUTPUT_DIR, f_name)
                np.save(save_path, aligned_feats)

                successful += 1

                if idx == 0:
                    logger.info(f"  Saved to: {save_path}\n")

            except Exception as e:
                failed += 1
                logger.error(f"[ERROR] Failed to process {f_name}: {str(e)}")
                if idx < 3:  # Print full traceback for first few files
                    import traceback
                    traceback.print_exc()
                continue

    logger.info(f"\n{'='*80}")
    logger.info(f"Aligned feature extraction complete!")
    logger.info(f"  Successful: {successful}/{len(to_process)}")
    logger.info(f"  Failed: {failed}/{len(to_process)}")
    logger.info(f"  Output directory: {OUTPUT_DIR}")
    logger.info(f"{'='*80}")


def verify_extraction():
    """Verify that extraction worked by comparing a few files"""
    logger.info("\n[VERIFICATION] Checking extracted features...")

    output_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npy')])
    if not output_files:
        logger.error("No output files found!")
        return

    logger.info(f"Total aligned feature files: {len(output_files)}")

    # Check first 3 files
    for f_name in output_files[:3]:
        input_path = os.path.join(INPUT_FEAT_DIR, f_name)
        output_path = os.path.join(OUTPUT_DIR, f_name)

        if not os.path.exists(input_path):
            logger.warning(f"Input file missing: {f_name}")
            continue

        input_feats = np.load(input_path)
        output_feats = np.load(output_path)

        logger.info(f"\nFile: {f_name}")
        logger.info(f"  Input shape:  {input_feats.shape}")
        logger.info(f"  Output shape: {output_feats.shape}")
        logger.info(f"  Time preserved: {input_feats.shape[0] == output_feats.shape[0]}")


if __name__ == "__main__":
    logger.info("="*80)
    logger.info("GTCC Aligned Feature Extraction for ProTAS")
    logger.info("="*80)

    # Main extraction
    extract_features()

    # Verify results
    verify_extraction()
