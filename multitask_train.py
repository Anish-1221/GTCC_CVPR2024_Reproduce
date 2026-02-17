import json

import torch
from torch.utils.data import DataLoader
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from utils.os_util import get_env_variable
from utils.repeating_distributed_sampler import RepeatingDistributedSampler
from utils.loss_entry import get_loss_function
from utils.logging import configure_logging_format
from utils.plotter import validate_folder
from utils.train_util import (
    get_base_model_deets,
    get_data_subfolder_and_extension,
    save_dict_to_json_file
)
from utils.collate_functions import jsondataset_collate_fn
from models.json_dataset import jsondataset_get_train_test, data_json_labels_handles
from models.alignment_training_loop import alignment_training_loop
from models.model_multiprong import MultiProngAttDropoutModel
from configs.entry_config import get_generic_config


if __name__ == '__main__':
    import os
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")

    use_ddp = 'LOCAL_RANK' in os.environ

    if use_ddp:
        # DDP Setup
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        world_size = dist.get_world_size()

    else:
        #  Single GPU mode (python)
        # Check if CUDA_VISIBLE_DEVICES is set
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        gpu_id = int(visible_devices.split(',')[0]) if visible_devices else 0
        
        local_rank = 0
        
        # Force the specific GPU
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # This is device 0 in the visible list
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
        
        world_size = 1
        print(f"Single GPU mode - using device: {device}")

    # setup logger and pytorch device
    logger = configure_logging_format()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # environment variable
    dset_json_folder = get_env_variable('JSON_DPATH')

    # get configuration for this experiment
    CFG = get_generic_config(multi_task_setting=True)

    if local_rank == 0:
        # initialize folder for logging output
        master_experiment_foldername = CFG.EVAL_PLOTFOLDER + f'/{CFG.EXPERIMENTNAME}'
        validate_folder(master_experiment_foldername)
        save_dict_to_json_file(CFG, master_experiment_foldername + '/config.json')

    # Broadcast folder name in DDP mode
    if use_ddp:
        folder_list = [CFG.EVAL_PLOTFOLDER + f'/{CFG.EXPERIMENTNAME}'] if local_rank == 0 else [None]
        dist.broadcast_object_list(folder_list, src=0)
        master_experiment_foldername = folder_list[0]
    else:
        master_experiment_foldername = CFG.EVAL_PLOTFOLDER + f'/{CFG.EXPERIMENTNAME}'

    # dataset structure json, get DSET variables
    data_structure = data_json_labels_handles(dset_json_folder, dset_name=CFG.DATASET_NAME)
    TASKS = data_structure.keys()
    data_subfolder_name, datafile_extension = get_data_subfolder_and_extension(architecture=CFG.BASEARCH.ARCHITECTURE)
    data_folder = f'{CFG.DATAFOLDER}/{data_subfolder_name}'


    train_dataloaders = {}
    train_samplers = {} if use_ddp else None

    for task in TASKS:
        batch_size = CFG.BATCH_SIZE
        train_set, test_set = jsondataset_get_train_test(
            task=task,
            task_json=data_structure[task],
            data_folder=data_folder,
            device=device,
            split=CFG.TRAIN_SPLIT[0],
            extension=datafile_extension,
            data_size=CFG.DATA_SIZE,
            lazy_loading=CFG.LAZY_LOAD
        )

        if use_ddp:
            # Use RepeatingDistributedSampler to handle small datasets
            # This ensures tasks with fewer videos than batch_size*world_size still train
            train_sampler = RepeatingDistributedSampler(
                train_set,
                batch_size=CFG.BATCH_SIZE,
                num_replicas=world_size,
                rank=local_rank,
                shuffle=True,
                seed=42,
                drop_last=True
            )

            # Log warning if repetition is needed (only on rank 0)
            if local_rank == 0:
                info = train_sampler.get_repeat_info()
                if info['needs_repetition']:
                    logger.warning(
                        f"Task '{task}': Small dataset ({info['original_dataset_size']} videos) "
                        f"repeated {info['repeat_factor']}x to ensure training batches. "
                        f"Each GPU gets {info['samples_per_gpu']} samples "
                        f"({info['batches_per_gpu']} batches of {info['batch_size']})."
                    )

            train_samplers[task] = train_sampler
            train_dataloaders[task] = DataLoader(train_set, batch_size=CFG.BATCH_SIZE, sampler=train_sampler, collate_fn=jsondataset_collate_fn, drop_last=True, shuffle=False, num_workers=4, pin_memory=True)

        else:
            # Single GPU mode - also handle small datasets
            if len(train_set) < CFG.BATCH_SIZE:
                # Use RepeatingDistributedSampler with num_replicas=1 for single GPU
                train_sampler = RepeatingDistributedSampler(
                    train_set,
                    batch_size=CFG.BATCH_SIZE,
                    num_replicas=1,
                    rank=0,
                    shuffle=True,
                    seed=42,
                    drop_last=True
                )
                info = train_sampler.get_repeat_info()
                logger.warning(
                    f"Task '{task}': Small dataset ({info['original_dataset_size']} videos) "
                    f"repeated {info['repeat_factor']}x to ensure training batches."
                )
                train_dataloaders[task] = DataLoader(train_set, batch_size=CFG.BATCH_SIZE, sampler=train_sampler, collate_fn=jsondataset_collate_fn, drop_last=True, shuffle=False)
            else:
                train_dataloaders[task] = DataLoader(train_set, batch_size=CFG.BATCH_SIZE, collate_fn=jsondataset_collate_fn, drop_last=True, shuffle=True)

        if local_rank == 0:
            logger.debug(f'{len(train_dataloaders[task].dataset)} in train set for {task}')
            logger.debug("Dataloaders successfully obtained.")

    if CFG.ARCHITECTURE['MCN']:
        if CFG.ARCHITECTURE['num_heads'] is None:
            num_tks = len(TASKS)
        else:
            num_tks = CFG.ARCHITECTURE['num_heads']
        base_model_class, base_model_params = get_base_model_deets(CFG)
        model = MultiProngAttDropoutModel(
            base_model_class=base_model_class,
            base_model_params=base_model_params,
            output_dimensionality=CFG.OUTPUT_DIMENSIONALITY,
            num_heads=num_tks,
            dropping=CFG.LOSS_TYPE['GTCC'],
            attn_layers=CFG.ARCHITECTURE['attn_layers'],
            drop_layers=CFG.ARCHITECTURE['drop_layers'],
        )
    else:
        base_model_class, base_model_params = get_base_model_deets(CFG)
        model = base_model_class(**base_model_params)

    model = model.to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)


    # Determine loss type from config for method-specific memory thresholds
    loss_type = None
    for lt, enabled in CFG.LOSS_TYPE.items():
        if enabled:
            loss_type = lt
            break

    alignment_training_loop(
        model,
        train_dataloaders,
        get_loss_function(CFG),
        master_experiment_foldername,
        CONFIG=CFG,
        local_rank=local_rank,
        train_samplers=train_samplers,
        loss_type=loss_type
    )

    # Cleanup
    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()
