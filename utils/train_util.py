import glob
from easydict import EasyDict as edict
import json 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.model_singleprong import Resnet50Encoder, StackingEncoder, NaiveEncoder
from models.model_multiprong import MultiProngAttDropoutModel
from utils.logging import configure_logging_format

logger = configure_logging_format()



def save_dict_to_json_file(dictionary, filepath):
    with open(filepath, 'w') as fp:
        json.dump(dictionary, fp, indent=4)


def get_npy_shape_from_file(file_path):
    with open(file_path, 'rb') as f:
        version = np.lib.format.read_magic(f)
        shape, _, _ = np.lib.format.read_array_header_1_0(f) if version == (1, 0) else np.lib.format.read_array_header_2_0(f)
    return shape


def get_base_model_deets(config_obj):
    if type(config_obj) == dict:
        config_obj = edict(config_obj)
    architecture = config_obj.BASEARCH.ARCHITECTURE
    if architecture == 'temporal_stacking':
        return StackingEncoder, config_obj.BASEARCH.TEMPORAL_STACKING_ARCH
    elif architecture == 'naive':
        return NaiveEncoder, config_obj.BASEARCH.NAIVE_ARCH
    elif architecture == 'resnet50':
        return Resnet50Encoder, config_obj.BASEARCH.Resnet50_ARCH
    else:
        logger.error("Bad Architecture Value, check CONFIG object")
        exit(1)

def flatten_dataloader(dl):
    return [pair for batch in list(iter(dl)) for pair in list(zip(batch[0], batch[1]))]


def ckpt_restore_mprong(path, num_heads, dropout=False, device='cpu'):
    # Additional information
    checkpoint = torch.load(path, map_location="cpu")
    config_obj = checkpoint['config']
    base_model_class, base_model_params = get_base_model_deets(config_obj)

    # [LEARNABLE PROGRESS] Detect if checkpoint has progress_head
    # Check for both regular keys and DDP-wrapped keys (with 'module.' prefix)
    state_dict = checkpoint['model_state_dict']
    has_progress_head = any(
        k.startswith('progress_head.') or k.startswith('module.progress_head.')
        for k in state_dict.keys()
    )

    # Build progress_head_config if needed
    progress_head_config = None
    if has_progress_head:
        # Detect architecture type from checkpoint keys
        # - GRU-based: has 'progress_head.gru.*' keys
        # - Transformer: has 'progress_head.layers.*' and 'progress_head.alibi_slopes'
        # - DilatedConv: has 'progress_head.blocks.*' and 'progress_head.conv_dilated' patterns

        def has_key_pattern(pattern):
            return any(pattern in k for k in state_dict.keys())

        # Check for architecture indicators (handle both regular and DDP keys)
        has_transformer = has_key_pattern('progress_head.layers.') or has_key_pattern('module.progress_head.layers.')
        has_dilated_conv = has_key_pattern('progress_head.blocks.') or has_key_pattern('module.progress_head.blocks.')
        has_gru = has_key_pattern('progress_head.gru.') or has_key_pattern('module.progress_head.gru.')

        if has_transformer:
            # Transformer architecture detected
            print(f"[INFO] Detected TransformerProgressHead - loading with ALiBi attention")
            progress_head_config = {
                'architecture': 'transformer',
                'transformer_config': {
                    'd_model': 64,
                    'num_heads': 4,
                    'num_layers': 2,
                    'ffn_dim': 128,
                    'dropout': 0.1,
                }
            }
        elif has_dilated_conv:
            # DilatedConv architecture detected
            print(f"[INFO] Detected DilatedConvProgressHead - loading with causal dilated convolutions")
            progress_head_config = {
                'architecture': 'dilated_conv',
                'dilated_conv_config': {
                    'hidden_dim': 64,
                    'kernel_size': 3,
                    'dilations': [1, 2, 4, 8, 16, 32],
                    'dropout': 0.1,
                }
            }
        elif has_gru:
            # GRU-based architecture - detect position encoding
            gru_key = 'progress_head.gru.weight_ih_l0'
            module_gru_key = 'module.progress_head.gru.weight_ih_l0'

            use_position_encoding = False  # Default to no position encoding (V4+)
            if gru_key in state_dict:
                gru_input_dim = state_dict[gru_key].shape[1]
                # If input_dim is output_dim + 1, it has position encoding
                use_position_encoding = (gru_input_dim == config_obj['OUTPUT_DIMENSIONALITY'] + 1)
            elif module_gru_key in state_dict:
                gru_input_dim = state_dict[module_gru_key].shape[1]
                use_position_encoding = (gru_input_dim == config_obj['OUTPUT_DIMENSIONALITY'] + 1)

            progress_head_config = {
                'architecture': 'gru',
                'hidden_dim': 64,
                'use_gru': True,
                'use_position_encoding': use_position_encoding
            }
            if use_position_encoding:
                print(f"[INFO] Detected GRU ProgressHead (v3) - loading with position encoding")
            else:
                print(f"[INFO] Detected GRU ProgressHead (v2/v4) - loading without position encoding")
        else:
            # Fallback: default GRU config
            progress_head_config = {
                'architecture': 'gru',
                'hidden_dim': 64,
                'use_gru': True,
                'use_position_encoding': False
            }
            print(f"[INFO] Using default GRU ProgressHead configuration")

    if 'drop_layers' in config_obj['ARCHITECTURE'].keys():
        model = MultiProngAttDropoutModel(
            base_model_class=base_model_class,
            base_model_params=base_model_params,
            output_dimensionality=config_obj['OUTPUT_DIMENSIONALITY'],
            num_heads=num_heads,
            dropping=dropout,
            attn_layers=config_obj['ARCHITECTURE']['attn_layers'],
            drop_layers=config_obj['ARCHITECTURE']['drop_layers'],
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
            attn_layers=config_obj['ARCHITECTURE']['attn_layers'],
            use_progress_head=has_progress_head,
            progress_head_config=progress_head_config,
        ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config_obj['LEARNING_RATE'])

    # ADD: Handle DDP state dict (keys may have 'module.' prefix)
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        # Remove module. prefix
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # Remove 'module.'
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return (
        model,
        optimizer,
        checkpoint['epoch'],
        checkpoint['loss'],
        config_obj
    )


def ckpt_restore_sprong(path, device='cpu'):
    # Additional information
    checkpoint = torch.load(path, map_location="cpu")
    config_obj = checkpoint['config']
    base_model_class, base_model_params = get_base_model_deets(config_obj)
    model = base_model_class(**base_model_params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config_obj['LEARNING_RATE'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return (
        model,
        optimizer,
        checkpoint['epoch'],
        checkpoint['loss'],
        config_obj
    )



def get_data_subfolder_and_extension(architecture):
    """
        Based on the architecture, return the subfolder where the data files will be and the file-entension of the data files.
    """
    if architecture == 'resnet50':
        return 'frames', 'npy'
    else:
        return 'features', 'npy'


def get_config_for_folder(folder):
    try:
        with open(folder + '/config.json', 'r') as json_file:
            config = edict(json.load(json_file))
    except Exception as e:
        print(glob.glob(folder + '/*'))
        folder_to_check = glob.glob(folder + '/*')[0]
        with open(folder_to_check + '/config.json', 'r') as json_file:
            config = edict(json.load(json_file))
    return config

