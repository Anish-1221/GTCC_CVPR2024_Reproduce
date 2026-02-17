import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import imageio
import glob
import random
import os
import pandas as pd
import time
import json
from utils.collate_functions import jsondataset_collate_fn
from utils.logging import configure_logging_format
from utils.train_util import get_data_subfolder_and_extension, get_npy_shape_from_file


logger = configure_logging_format()

def get_fps_scale_factor(handle):
    """Determine FPS scaling based on video source dataset"""
    if 'OP' in handle or 'P0' in handle or 'P1' in handle or 'P2' in handle:  # EGTEA
        return 24
    elif handle.startswith('S') and '_' in handle:  # CMU_Kitchens
        return 30
    elif 'tent' in handle.lower():  # EpicTent
        return 60
    elif handle.isdigit() or (len(handle) == 4 and handle.isdigit()):  # MECCANO
        return 12
    elif 'Head_' in handle:  # PC_assembly/disassembly
        return 12
    else:
        print(f"WARNING: Unknown dataset for {handle}, defaulting to 30 FPS")
        return 30

def jsondataset_get_train_test(
    task,
    task_json,
    data_folder,
    device,
    split,
    extension='npy',
    data_size=None,
    lazy_loading=False
):
    return (
        JSONDataset(
            task=task,
            task_json=task_json,
            data_folder=data_folder,
            split=[split],
            extension=extension,
            loader_type='train',
            lazy_loading=lazy_loading
        ),
        JSONDataset(
            task=task,        
            task_json=task_json,
            data_folder=data_folder,
            split=[split],
            extension=extension,
            loader_type='test',
            lazy_loading=lazy_loading
        )
    )

class JSONDataset(Dataset):
    def __init__(
        self,
        task,
        task_json,
        data_folder,
        split,
        extension='npy',
        loader_type=True,
        lazy_loading=True
    ):
        """
            This filepath must maintain the following structure
                data_folder/
                    - embeddings/
                        baseball_pitch/
                            0001.npy
                            ...
                        ...
                    - times/
                        baseball_pitch/
                            0001.csv
                            ...
                        ...
        """
        # make sure split is either [train-test-split] or [train-testval-split, testval-split]
        assert type(split) == list and len(split) in [1,2] and all([0 < p < 1 for p in split])
        # make sure if only train/test split is give that loader_type isnt val
        with_val = len(split) == 2
        assert with_val or loader_type != 'val'

        self.data_folder = data_folder + f'/{task}'
        self.task = task
        self.split = split

        N = len(task_json['handles'])
        all_embeddings_files = glob.glob(f'{data_folder}/*')
        num_vids_total = len(task_json['handles'])
        train_end_idx = round(num_vids_total * split[0])
        test_start_idx = round(train_end_idx + (num_vids_total-train_end_idx) * split[1]) if with_val else train_end_idx
        all_handle_indices = list(range(N))
        if loader_type == 'train':
            active_hdl_indices = all_handle_indices[:train_end_idx]
        elif loader_type == 'val':
            assert with_val
            active_hdl_indices = all_handle_indices[train_end_idx:test_start_idx]
        elif loader_type == 'test':
            active_hdl_indices = all_handle_indices[test_start_idx:num_vids_total]
        else:
            logger.error("Bad loader type (should be in ['train', 'val', 'test'])")
            exit(1)


        datas = []
        times = []
        names = []
        self.action_set = set()
        for i, (handle, action_sequence, start_times, end_times) in enumerate(zip(
            task_json['handles'], task_json['hdl_actions'], task_json['hdl_start_times'], task_json['hdl_end_times']
        )):
            for a in action_sequence:
                self.action_set.add(a)

            if i not in active_hdl_indices:
                continue
            assert f'{data_folder}/{handle}.{extension}' in all_embeddings_files, f'File {handle}.{extension} not in {data_folder} folder'
            # log the data filename
            data = f'{data_folder}/{handle}.{extension}'
            # times_dict = {'step': action_sequence, 'start_frame': [int(t) for t in start_times], 'end_frame': [int(t) for t in end_times], 'name': handle}
            fps_scale = get_fps_scale_factor(handle)
            times_dict = {'step': action_sequence, 'start_frame': [int(t / fps_scale) for t in start_times], 'end_frame': [int(t / fps_scale) for t in end_times], 'name': handle}
            N = get_npy_shape_from_file(data)[0]
            
            if N > times_dict['end_frame'][-1] - 1:
                times_dict['end_frame'][-1] = N-1
            # elif N < times_dict['end_frame'][-1] - 1:
            #     while N < times_dict['end_frame'][-1] - 1:
            #         if N > times_dict['start_frame'][-1]+1:
            #             times_dict['end_frame'][-1] = N-1
            #             break
            #         else:
            #             times_dict['start_frame'] = times_dict['start_frame'][:-1]
            #             times_dict['end_frame'] = times_dict['end_frame'][:-1]


            elif N < times_dict['end_frame'][-1] - 1:
                while len(times_dict['end_frame']) > 0 and N < times_dict['end_frame'][-1] - 1:
                    if N > times_dict['start_frame'][-1]+1:
                        times_dict['end_frame'][-1] = N-1
                        break
                    else:
                        times_dict['start_frame'] = times_dict['start_frame'][:-1]
                        times_dict['end_frame'] = times_dict['end_frame'][:-1]
                
                # Skip videos with no valid actions
                if len(times_dict['end_frame']) == 0:
                    print(f"WARNING: Skipping {handle} - all actions removed (video too short)")
                    continue

            # add the times data dict
            datas.append(data if lazy_loading else np.load(data))
            times.append(times_dict)
            names.append(task + '_' + handle)

        # logger.info("Embeddings folder and time label folder contain same file, times are in order, moving on")
        self.times = times
        self.data_label_name = list(zip(datas, times, names))
        if loader_type == 'test':
            random.seed(1)
        random.shuffle(self.data_label_name)

    def __len__(self):
        """
            gives the length of the dataset
        """
        return len(self.data_label_name)
    
    def __getitem__(self, index):
        """
            responsible for returning the 'index'^th data and label from wherever.
        """
        if type(self.data_label_name[index][0]) == str:
            return self.data_label_name[index][0], self.data_label_name[index][1], self.data_label_name[index][2]
        else:
            return torch.from_numpy(self.data_label_name[index][0]), self.data_label_name[index][1], self.data_label_name[index][2]


def jsondataset_from_splits(
    task: str,
    task_json: dict,
    data_folder: str,
    splits_dict: dict,
    split_type: str,
    extension: str = 'npy',
    lazy_loading: bool = True
):
    """
    Create a dataset from pre-defined splits (loaded from data_splits.json).

    Args:
        task: Task name (e.g., 'BaconAndEggs.egtea')
        task_json: Task metadata from data_structure (contains handles, hdl_actions, etc.)
        data_folder: Path to feature embeddings folder
        splits_dict: Pre-loaded splits from data_splits.json (from load_splits_from_json)
        split_type: One of 'train', 'val', 'test'
        extension: File extension for features (default 'npy')
        lazy_loading: Whether to lazy-load features (default True)

    Returns:
        JSONDatasetFromSplits instance
    """
    # Get allowed handles for this task and split type
    if task not in splits_dict:
        logger.warning(f"Task '{task}' not in splits_dict, returning empty set")
        allowed_handles = set()
    else:
        allowed_handles = set(splits_dict[task][split_type])

    return JSONDatasetFromSplits(
        task=task,
        task_json=task_json,
        data_folder=data_folder,
        allowed_handles=allowed_handles,
        extension=extension,
        lazy_loading=lazy_loading
    )


class JSONDatasetFromSplits(Dataset):
    """
    Dataset that only includes handles from a pre-defined split.

    This class is used when loading data from data_splits.json which contains
    deterministic, stratified train/val/test splits.
    """

    def __init__(
        self,
        task: str,
        task_json: dict,
        data_folder: str,
        allowed_handles: set,
        extension: str = 'npy',
        lazy_loading: bool = True
    ):
        """
        Initialize dataset with only videos in allowed_handles.

        Args:
            task: Task name
            task_json: Task metadata from data_structure
            data_folder: Path to feature embeddings
            allowed_handles: Set of video handles to include in this dataset
            extension: File extension for features
            lazy_loading: Whether to lazy-load features
        """
        self.data_folder = data_folder + f'/{task}'
        self.task = task
        self.allowed_handles = allowed_handles

        all_embeddings_files = glob.glob(f'{data_folder}/*')

        datas = []
        times = []
        names = []
        self.action_set = set()

        for i, (handle, action_sequence, start_times, end_times) in enumerate(zip(
            task_json['handles'], task_json['hdl_actions'],
            task_json['hdl_start_times'], task_json['hdl_end_times']
        )):
            # Only include handles in our split
            if handle not in allowed_handles:
                continue

            for a in action_sequence:
                self.action_set.add(a)

            # Verify file exists
            file_path = f'{data_folder}/{handle}.{extension}'
            if file_path not in all_embeddings_files:
                logger.warning(f'File {handle}.{extension} not in {data_folder} folder, skipping')
                continue

            data = file_path
            fps_scale = get_fps_scale_factor(handle)
            times_dict = {
                'step': action_sequence,
                'start_frame': [int(t / fps_scale) for t in start_times],
                'end_frame': [int(t / fps_scale) for t in end_times],
                'name': handle
            }

            N = get_npy_shape_from_file(data)[0]

            # Handle frame count mismatches (same logic as JSONDataset)
            if N > times_dict['end_frame'][-1] - 1:
                times_dict['end_frame'][-1] = N - 1
            elif N < times_dict['end_frame'][-1] - 1:
                while len(times_dict['end_frame']) > 0 and N < times_dict['end_frame'][-1] - 1:
                    if N > times_dict['start_frame'][-1] + 1:
                        times_dict['end_frame'][-1] = N - 1
                        break
                    else:
                        times_dict['start_frame'] = times_dict['start_frame'][:-1]
                        times_dict['end_frame'] = times_dict['end_frame'][:-1]
                        times_dict['step'] = times_dict['step'][:-1] if len(times_dict['step']) > 1 else times_dict['step']

                # Skip videos with no valid actions
                if len(times_dict['end_frame']) == 0:
                    logger.warning(f"Skipping {handle} - all actions removed (video too short)")
                    continue

            # Add the data
            datas.append(data if lazy_loading else np.load(data))
            times.append(times_dict)
            names.append(task + '_' + handle)

        self.times = times
        self.data_label_name = list(zip(datas, times, names))

        # Shuffle with fixed seed for reproducibility
        random.seed(42)
        random.shuffle(self.data_label_name)

    def __len__(self):
        """Return the number of videos in this dataset."""
        return len(self.data_label_name)

    def __getitem__(self, index):
        """Return the index-th data item."""
        if type(self.data_label_name[index][0]) == str:
            return self.data_label_name[index][0], self.data_label_name[index][1], self.data_label_name[index][2]
        else:
            return torch.from_numpy(self.data_label_name[index][0]), self.data_label_name[index][1], self.data_label_name[index][2]


def data_json_labels_handles(dset_json_folder, dset_name='egoprocel'):
    subset_specifier = None
    if dset_name in ['cmu', 'egtea']:
        subset_specifier = dset_name
        dset_name = 'egoprocel'

    data_json = f"{dset_json_folder}/{dset_name}.json"
    with open(data_json, 'r') as file:
        d = json.load(file)
        if subset_specifier is None:
            return d
        else:
            return {key: value for key, value in d.items() if subset_specifier in key}


def get_test_dataloaders(tasks, data_structure, config, device):
    batch_size = config.BATCH_SIZE
    data_subfolder_name, datafile_extension = get_data_subfolder_and_extension(architecture=config.BASEARCH.ARCHITECTURE)
    data_folder = f'{config.DATAFOLDER}/{data_subfolder_name}'
    test_dataloaders = {}
    for task in tasks:
        _, test_set = jsondataset_get_train_test(
            task=task,
            task_json=data_structure[task],
            data_folder=data_folder,
            device=device,
            split=config.TRAIN_SPLIT[0],
            extension=datafile_extension,
            data_size=config.DATA_SIZE,
            lazy_loading=config.LAZY_LOAD
        )
        logger.debug(f'{len(test_set)} vids in test set for {task}')
        if batch_size is None:
            batch_size = len(test_set)
        test_dataloaders[task] = DataLoader(test_set, batch_size=batch_size, collate_fn=jsondataset_collate_fn, drop_last=True, shuffle=False)
    return test_dataloaders


def get_train_dataloaders(tasks, data_structure, config, device):
    batch_size = config.BATCH_SIZE
    data_subfolder_name, datafile_extension = get_data_subfolder_and_extension(architecture=config.BASEARCH.ARCHITECTURE)
    data_folder = f'{config.DATAFOLDER}/{data_subfolder_name}'
    test_dataloaders = {}
    for task in tasks:
        _, test_set = jsondataset_get_train_test(
            task=task,
            task_json=data_structure[task],
            data_folder=data_folder,
            device=device,
            split=config.TRAIN_SPLIT[0],
            extension=datafile_extension,
            data_size=config.DATA_SIZE,
            lazy_loading=config.LAZY_LOAD
        )
        # logger.debug(f'{len(test_set)} vids in train set for {task}')
        if batch_size is None:
            batch_size = len(test_set)
        test_dataloaders[task] = DataLoader(test_set, batch_size=batch_size, collate_fn=jsondataset_collate_fn, drop_last=True, shuffle=False)
    return test_dataloaders