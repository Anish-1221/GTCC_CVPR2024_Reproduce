# import glob
# import copy
# import argparse

# import pandas as pd
# from models.json_dataset import get_test_dataloaders
# import numpy as np
# import torch

# from utils.ckpt_save import get_ckpt_for_eval
# from utils.os_util import get_env_variable
# from models.json_dataset import data_json_labels_handles
# from utils.evaluation import (
#     PhaseProgression,
#     PhaseClassification,
#     KendallsTau,
#     WildKendallsTau,
#     EnclosedAreaError,
#     OnlineGeoProgressError
# )
# from utils.plotter import validate_folder
# from utils.logging import configure_logging_format
# from utils.train_util import get_config_for_folder

# # GLOBAL VARS
# logger = configure_logging_format()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def update_json_dict_and_save(ckpt_out_folder, test_object, historical_json, update_json):
#     if historical_json[test_object.name] is None:
#         historical_json[test_object.name] = {key: [value] for key, value in update_json.items()}
#     else:
#         for key, value in update_json.items():
#             historical_json[test_object.name][key].append(value)
#     json_to_save = copy.deepcopy(historical_json[test_object.name])
#     json_to_save['task'].append('MEAN')
#     for k in json_to_save.keys():
#         if k != 'task':
#             json_to_save[k].append(np.mean(json_to_save[k]) if None not in json_to_save[k] else None)
#     pd.DataFrame(json_to_save).to_csv(f'{ckpt_out_folder}/{test_object}.csv')

# def begin_eval_loop_over_tasks(config, folder_to_test, tests_to_run, tasks, test_tasks, test_dl_dict, model=None):
#     ##########################################
#     # for each ckpt in this folder....
#     ##########################################
#     out_folder = f'{folder_to_test}/EVAL'
#     validate_folder(out_folder)
#     json_test_result = {test_object.name: None for test_object in tests_to_run}

#     # If multitask setting, then grab single model
#     multi_prong_bool = config["MULTITASK"]

#     if model is not None:
#         logger.info("Using manually provided ProTAS model. Skipping GTCC checkpoint loading.")
#         model.eval()
#         epoch = "ProTAS_Final"
#         ckpt_handle = "ProTAS_Eval" # This names the output subfolder in /EVAL/
#     else:
#         if multi_prong_bool:
#             if config['ARCHITECTURE']['num_heads'] is None:
#                 num_heads = len(tasks)
#             else:
#                 num_heads = config['ARCHITECTURE']['num_heads']
#             model, epoch, ckpt_handle = get_ckpt_for_eval(
#                 ckpt_parent_folder=folder_to_test,
#                 config=config,
#                 num_heads=num_heads,
#                 device=device
#             )

#     for task in sorted(test_tasks):
#         logger.info(f'\n{"*" * 40}\n{"*" * 40}\n{task}\n{"*" * 40}\n{"*" * 40}')
        
#         if not multi_prong_bool and model is None:
#             if f'{folder_to_test}/{task}' not in glob.glob(folder_to_test + '/*'):
#                 continue
#             taskfolder_to_test = f'{folder_to_test}/{task}'
#             model, epoch, ckpt_handle = get_ckpt_for_eval(
#                 ckpt_parent_folder=taskfolder_to_test,
#                 config=config,
#                 device=device
#             )
#         ckpt_out_folder = f'{out_folder}/{ckpt_handle}'
#         validate_folder(ckpt_out_folder)
#         ####################
#         # loop tests for this ckpt
#         ####################
#         for test_object in tests_to_run:
#             ####################
#             # run the test
#             ####################
#             logger.info(f'** Beginning Test: {test_object} for {folder_to_test.split("/")[-1]}')
#             eval_results_dict = test_object(
#                 model,
#                 config,
#                 epoch,
#                 {task: test_dl_dict[task]},
#                 folder_to_test,
#                 [task]
#             )
#             if eval_results_dict is not None:
#                 update_json_dict_and_save(
#                     ckpt_out_folder=ckpt_out_folder,
#                     test_object=test_object,
#                     historical_json=json_test_result,
#                     update_json=eval_results_dict
#                 )
#             logger.info(f'** Finished Test: {test_object} for {task}')


# if __name__ == '__main__':
#     ##########################################
#     # quick parser code to specify folder to test.
#     parser = argparse.ArgumentParser(description='Please specify the parameters of the experiment.')
#     parser.add_argument('-f', '--folder', required=True) 
#     args = parser.parse_args()

#     tests_to_run = {
#         # EnclosedAreaError(),
#         OnlineGeoProgressError(),
#         # KendallsTau(),
#         # WildKendallsTau(),
#         # PhaseClassification(),
#         # PhaseProgression(),
#     }

#     dset_json_folder = get_env_variable('JSON_DPATH')
#     folder_to_test = args.folder
#     logger.info(f'Beginning test suite for {folder_to_test.split("/")[-1]}')

#     ##########################################
#     # get config!
#     config = get_config_for_folder(folder_to_test)

#     # ADD THIS: Force the ProTAS flag and params into the config
#     config.USE_PROTAS = True
#     config.PROTAS_PARAMS = {
#     'num_stages': 4,
#     'num_layers': 10,
#     'num_f_maps': 64,
#     'dim': 2048,           # ResNet features are 2048-dim
#     'num_classes': 30,      # Match your mapping.txt
#     'causal': True,         # Based on your --causal flag
#     'use_graph': True,      # Based on your --graph flag
#     'learnable': True,      # Based on your --learnable_graph flag
#     # Path to the .pkl file used during training
#     'init_graph_path': '/vision/anishn/ProTAS/data_1fps/egoprocel_subset1_S/graph/graph.pkl' 
# }

#     # Initialize the model with the ProTAS flag
#     from models.model_multiprong import MultiProngAttDropoutModel
#     model = MultiProngAttDropoutModel(
#         base_model_class=None, # Not used for ProTAS
#         base_model_params=None, 
#         output_dimensionality=2048,
#         num_heads=30,
#         use_protas=True,
#         protas_params=config.PROTAS_PARAMS
#     )

#     # LOAD PROTAS WEIGHTS MANUALLY
#     protas_path = "/u/anishn/models/egoprocel_subset1_S_1fps/egoprocel_subset1_S/split_1/epoch-50.model"
#     state_dict = torch.load(protas_path, map_location=device)
#     # Handle Trainer wrapper if necessary
#     if 'model_state_dict' in state_dict:
#         model.protas_model.load_state_dict(state_dict['model_state_dict'])
#     else:
#         model.protas_model.load_state_dict(state_dict)
    
#     model.to(device)
#     model.eval()

#     ##########################################
#     # dataset util form saved json
#     data_structure = data_json_labels_handles(dset_json_folder, dset_name=config.DATASET_NAME)
#     TASKS = data_structure.keys()
#     testTASKS = TASKS # edit if you see fit

#     ##########################################
#     # get all test dataloaders
#     test_dataloaders = get_test_dataloaders(
#         tasks=testTASKS,
#         data_structure=data_structure,
#         config=config,
#         device=device
#     )

#     ##########################################
#     # print summary information
#     ##########################################
#     logger.info(f'Model architecture is {config.BASEARCH.ARCHITECTURE}')
#     logger.info(f'Dataset is {config.DATASET_NAME}')
#     logger.info(f'Test deck is {tests_to_run}')
#     logger.info(f'Folder to run is {folder_to_test}')
#     logger.info(f'tasks are {TASKS}')
 
#     begin_eval_loop_over_tasks(
#         config,
#         folder_to_test,
#         tests_to_run,
#         TASKS,
#         testTASKS,
#         test_dataloaders,
#         model=model
#     )

import glob
import copy
import argparse
import sys

import json
import pandas as pd
from models.json_dataset import get_test_dataloaders, jsondataset_from_splits, data_json_labels_handles
from torch.utils.data import DataLoader
from utils.collate_functions import jsondataset_collate_fn
from utils.train_util import get_data_subfolder_and_extension
import numpy as np
import torch

from utils.ckpt_save import get_ckpt_for_eval
from utils.os_util import get_env_variable

# [4FPS MODIFICATION] Added --level argument to switch between video-level and action-level evaluation
# Parse args early to determine which evaluation module to import
# This is needed because the import must happen before we use the evaluation classes
_temp_parser = argparse.ArgumentParser(add_help=False)
_temp_parser.add_argument('--level', choices=['video', 'action'], default='action')
_temp_args, _ = _temp_parser.parse_known_args()

# [4FPS MODIFICATION] Conditional import based on --level argument
# video: uses utils.evaluation (progress 0->1 across entire video)
# action: uses utils.evaluation_action_level (progress 0->1 per action segment)
if _temp_args.level == 'video':
    from utils.evaluation import (
        PhaseProgression,
        PhaseClassification,
        KendallsTau,
        WildKendallsTau,
        EnclosedAreaError,
        OnlineGeoProgressError
    )
    EVAL_LEVEL = 'video'
else:
    from utils.evaluation_action_level import (
        PhaseProgression,
        PhaseClassification,
        KendallsTau,
        WildKendallsTau,
        EnclosedAreaError,
        OnlineGeoProgressError
    )
    EVAL_LEVEL = 'action'

from utils.plotter import validate_folder
from utils.logging import configure_logging_format
from utils.train_util import get_config_for_folder

# GLOBAL VARS
logger = configure_logging_format()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_splits_from_json(splits_path='/vision/anishn/GTCC_CVPR2024/data_splits.json'):
    """Load pre-computed train/val/test splits."""
    with open(splits_path, 'r') as f:
        splits_data = json.load(f)
    return splits_data['splits']


def update_json_dict_and_save(ckpt_out_folder, test_object, historical_json, update_json):
    if historical_json[test_object.name] is None:
        historical_json[test_object.name] = {key: [value] for key, value in update_json.items()}
    else:
        for key, value in update_json.items():
            historical_json[test_object.name][key].append(value)
    json_to_save = copy.deepcopy(historical_json[test_object.name])
    json_to_save['task'].append('MEAN')
    for k in json_to_save.keys():
        if k != 'task':
            json_to_save[k].append(np.mean(json_to_save[k]) if None not in json_to_save[k] else None)
    pd.DataFrame(json_to_save).to_csv(f'{ckpt_out_folder}/{test_object}.csv')

def begin_eval_loop_over_tasks(config, folder_to_test, tests_to_run, tasks, test_tasks, test_dl_dict, eval_level='action'):
    ##########################################
    # for each ckpt in this folder....
    ##########################################
    # [4FPS MODIFICATION] Output folder now includes evaluation level suffix
    # This prevents video_level and action_level results from overwriting each other
    out_folder = f'{folder_to_test}/EVAL_{eval_level}_level'
    validate_folder(out_folder)
    json_test_result = {test_object.name: None for test_object in tests_to_run}

    # If multitask setting, then grab single model
    multi_prong_bool = config["MULTITASK"]
    if multi_prong_bool:
        if config['ARCHITECTURE']['num_heads'] is None:
            num_heads = len(tasks)
        else:
            num_heads = config['ARCHITECTURE']['num_heads']
        model, epoch, ckpt_handle = get_ckpt_for_eval(
            ckpt_parent_folder=folder_to_test,
            config=config,
            num_heads=num_heads,
            device=device
        )

    for task in sorted(test_tasks):
        logger.info(f'\n{"*" * 40}\n{"*" * 40}\n{task}\n{"*" * 40}\n{"*" * 40}')
        
        if not multi_prong_bool:
            if f'{folder_to_test}/{task}' not in glob.glob(folder_to_test + '/*'):
                continue
            taskfolder_to_test = f'{folder_to_test}/{task}'
            model, epoch, ckpt_handle = get_ckpt_for_eval(
                ckpt_parent_folder=taskfolder_to_test,
                config=config,
                device=device
            )
        ckpt_out_folder = f'{out_folder}/{ckpt_handle}'
        validate_folder(ckpt_out_folder)
        ####################
        # loop tests for this ckpt
        ####################
        for test_object in tests_to_run:
            ####################
            # run the test
            ####################
            logger.info(f'** Beginning Test: {test_object} for {folder_to_test.split("/")[-1]}')
            eval_results_dict = test_object(
                model,
                config,
                epoch,
                {task: test_dl_dict[task]},
                folder_to_test,
                [task]
            )
            if eval_results_dict is not None:
                update_json_dict_and_save(
                    ckpt_out_folder=ckpt_out_folder,
                    test_object=test_object,
                    historical_json=json_test_result,
                    update_json=eval_results_dict
                )
            logger.info(f'** Finished Test: {test_object} for {task}')


if __name__ == '__main__':
    ##########################################
    # quick parser code to specify folder to test.
    parser = argparse.ArgumentParser(description='Please specify the parameters of the experiment.')
    parser.add_argument('-f', '--folder', required=True)
    # [4FPS MODIFICATION] Added --level argument to switch between evaluation modes
    # video: progress 0->1 across entire video (uses utils.evaluation)
    # action: progress 0->1 per action segment (uses utils.evaluation_action_level)
    parser.add_argument('--level', choices=['video', 'action'], default='action',
                        help='Evaluation level: video (0->1 per video) or action (0->1 per action segment)')
    args = parser.parse_args()

    tests_to_run = {
        # EnclosedAreaError(),
        OnlineGeoProgressError(),
        # KendallsTau(),
        # WildKendallsTau(),
        # PhaseClassification(),
        # PhaseProgression(),
    }

    dset_json_folder = get_env_variable('JSON_DPATH')
    folder_to_test = args.folder
    logger.info(f'Beginning test suite for {folder_to_test.split("/")[-1]}')

    ##########################################
    # get config!
    config = get_config_for_folder(folder_to_test)

    ##########################################
    # dataset util form saved json
    data_structure = data_json_labels_handles(dset_json_folder, dset_name=config.DATASET_NAME)
    TASKS = data_structure.keys()
    testTASKS = TASKS # edit if you see fit

    ##########################################
    # get all test dataloaders using fixed splits from data_splits.json
    # [FIX] Previously used get_test_dataloaders with random splits,
    # now using jsondataset_from_splits with fixed train/val/test splits
    splits_dict = load_splits_from_json()

    data_subfolder_name, datafile_extension = get_data_subfolder_and_extension(
        architecture=config.BASEARCH.ARCHITECTURE
    )
    data_folder = f'{config.DATAFOLDER}/{data_subfolder_name}'

    test_dataloaders = {}
    for task in testTASKS:
        test_set = jsondataset_from_splits(
            task=task,
            task_json=data_structure[task],
            data_folder=data_folder,
            splits_dict=splits_dict,
            split_type='test',
            extension=datafile_extension,
            lazy_loading=config.LAZY_LOAD
        )
        logger.info(f'{len(test_set)} videos in test set for {task}')
        batch_size = config.BATCH_SIZE if config.BATCH_SIZE else len(test_set)
        test_dataloaders[task] = DataLoader(
            test_set,
            batch_size=batch_size,
            collate_fn=jsondataset_collate_fn,
            drop_last=False,  # Don't drop last batch for evaluation
            shuffle=False
        )

    ##########################################
    # print summary information
    ##########################################
    logger.info(f'Model architecture is {config.BASEARCH.ARCHITECTURE}')
    logger.info(f'Dataset is {config.DATASET_NAME}')
    # [4FPS MODIFICATION] Log the evaluation level being used
    logger.info(f'Evaluation level is {EVAL_LEVEL}_level')
    logger.info(f'Test deck is {tests_to_run}')
    logger.info(f'Folder to run is {folder_to_test}')
    logger.info(f'tasks are {TASKS}')
 
    # [4FPS MODIFICATION] Pass eval_level to output to separate folders
    begin_eval_loop_over_tasks(
        config,
        folder_to_test,
        tests_to_run,
        TASKS,
        testTASKS,
        test_dataloaders,
        eval_level=EVAL_LEVEL  # 'video' or 'action' based on --level argument
    )