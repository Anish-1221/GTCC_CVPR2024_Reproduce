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
        'hidden_dim': 64,
        'use_gru': True,
        'min_segment_len': 3,
        'samples_per_video': 5,      # Number of segments to sample per video
        'frames_per_segment': 3,     # Number of target frames per segment
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