from easydict import EasyDict as edict

# some variables
temporal_depth = 2
generic_linear_layer_sizes = [256, 1024, 512, 256]

# Get the configuration dictionary whose keys can be accessed with dot.
CONFIG = edict()
######################
## FOLDERS / Strings
CONFIG.DATASET_NAME = None
CONFIG.DATAFOLDER = None
CONFIG.EVAL_PLOTFOLDER = None
######################
## base model architecture
CONFIG.BASEARCH = edict()
CONFIG.BASEARCH.ARCHITECTURE = None
CONFIG.BASEARCH.Resnet50_ARCH = {
    'temporal_depth': temporal_depth,
}
CONFIG.BASEARCH.TEMPORAL_STACKING_ARCH = {
    'temporal_depth': temporal_depth,
    'conv_num_channels': 256,
    'drop_layers': [256, 1024, 512, 256],
}
CONFIG.BASEARCH.NAIVE_ARCH = {
    'layers': [1024, 512, 512, 256],
    'drop_layers': [256, 1024, 512, 256],
}
######################
## overall architecture
CONFIG.ARCHITECTURE = {
    'MCN': False,
    'drop_layers': generic_linear_layer_sizes,
    'attn_layers': [512, 1024, 512, 512],
    'num_heads': None,
}

######################
## loss
CONFIG.LOSS_TYPE = {
    'tcc': False,
    'GTCC': False,
    'LAV': False,
    'VAVA': False,
}
CONFIG.TCC_ORIGINAL_PARAMS = {
    'softmax_temp': .1,
    'alignment_variance': 0.001
}
CONFIG.GTCC_PARAMS = {
    'softmax_temp': .1,
    'max_gmm_iters': 8,
    'n_components': None,
    'delta': None,
    'gamma': None,
    'alignment_variance': 0,
}
CONFIG.LAV_PARAMS = {
    'min_temp': .1,
}
CONFIG.VAVA_PARAMS = {
}

######################
## Progress Loss Configuration
CONFIG.PROGRESS_LOSS = edict({
    'enabled': False,
    'method': 'cumulative_l2',      # 'cumulative_l2' or 'learnable'
    'lambda_fixed': 0.1,
    'learnable': {
        # Architecture selection: 'gru' (default), 'transformer', or 'dilated_conv'
        'architecture': 'gru',

        # GRU-specific config (used when architecture='gru')
        'hidden_dim': 64,
        'use_gru': True,
        'use_position_encoding': False,  # V4+: disabled by default

        # V9 architecture fixes (anti-saturation)
        'use_input_projection': False,   # Add Linear(input_dim→projection_dim) before GRU
        'projection_dim': 128,           # Target dimension for input projection
        'output_activation': 'sigmoid',  # 'sigmoid' or 'clamp' (clamped linear)
        'per_frame_count': False,        # Per-frame log(1+i)/log(1+max) instead of broadcast

        # V10 action conditioning + rate-of-change
        'use_action_conditioning': False, # Add action-class embedding for disambiguation
        'num_actions': 116,               # 0=unknown/background + 1-115 real actions
        'action_embed_dim': 16,           # Dimension of action class embedding
        'use_rate_of_change': False,      # Add frame-to-frame L2 distance as extra input

        # Transformer-specific config (used when architecture='transformer')
        'transformer_config': {
            'd_model': 64,
            'num_heads': 4,
            'num_layers': 2,
            'ffn_dim': 128,
            'dropout': 0.1,
        },

        # DilatedConv-specific config (used when architecture='dilated_conv')
        'dilated_conv_config': {
            'hidden_dim': 64,
            'kernel_size': 3,
            'dilations': [1, 2, 4, 8, 16, 32],
            'dropout': 0.1,
        },

        # Feature source: 'aligned' (128-d post-encoder) or 'raw' (512-d pre-alignment for resnet50)
        'features': 'aligned',

        # Frame count feature (shared across all architectures)
        'use_frame_count': True,     # Add log-scale frame count as input feature
        'frame_count_max': 300.0,    # Normalization factor: log(1+T)/log(1+max)

        # Training config (shared across all architectures)
        'min_segment_len': 3,
        'samples_per_video': 20,     # Number of times to sample per video (each picks a random action)
        'frames_per_segment': 5,     # Base number of target frames (used when adaptive_frames=False)
        'adaptive_frames': True,     # Scale target frames by action length
        'min_target_frames': 5,      # Minimum target frames per action
        'max_target_frames': 30,     # Maximum target frames per action
        'stratified_sampling': True, # Ensure early/mid/late parts of actions are covered

        # Loss formulation mode: 'uniform_mono' (default), 'sqrt_weighted', 'mse', 'legacy', 'dense'
        # 'dense': per-frame MSE on full action segments (disables frame_count automatically)
        'progress_loss_mode': 'legacy',

        # Legacy loss config (used when progress_loss_mode='legacy')
        'weighted_loss': True,       # Weight early frame errors more heavily
        'weight_cap': 20.0,          # Maximum weight for early frames
        'boundary_loss': True,       # Explicit supervision for first/last frames of actions
        'boundary_weight': 5.0,      # Weight multiplier for boundary loss

        # Monotonicity penalty config (used when progress_loss_mode='uniform_mono')
        'monotonicity_weight': 2.0,  # Weight for monotonicity penalty
        'monotonicity_margin': 0.01, # Minimum expected increase between consecutive predictions

        # Endpoint loss config (used when progress_loss_mode='uniform_mono' or 'mse')
        'endpoint_weight': 1.0,      # Weight for endpoint regularization (last frame → 1.0)

    },
})

######################
## global parameters
CONFIG.SKIP_RATE = None
CONFIG.MULTITASK = False
CONFIG.OUTPUT_DIMENSIONALITY = None
CONFIG.TRAIN_SPLIT = None
CONFIG.DATA_SIZE = None
CONFIG.LAZY_LOAD = True
CONFIG.DEBUG = False
CONFIG.BATCH_SIZE = None
CONFIG.LEARNING_RATE = None
CONFIG.NUM_EPOCHS = None
CONFIG.VERSION = None

# NAME OF FOLDER
CONFIG.EXPERIMENTNAME = None

# ProTAS specific settings
CONFIG.USE_PROTAS = False 
CONFIG.PROTAS_PARAMS = edict({
    'num_stages': 4,
    'num_layers': 10,
    'num_f_maps': 64,
    'dim': 2048,      # This should match your feature dimension (likely 2048 for ResNet/I3D)
    'num_classes': 31, # Match your mapping.txt count
    'causal': True,
    'use_graph': True # Set to True if your specific model used the graph head
})