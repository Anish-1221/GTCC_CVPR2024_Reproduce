import json
import random
import time

import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from models.model_multiprong import logger
from utils.train_util import save_dict_to_json_file
from utils.ckpt_save import ckpt_save
from utils.plotter import validate_folder
from utils.tensorops import preprocess_batch, contains_non_float_values
from utils.loss_functions import TCC_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def print_memory_stats(stage, local_rank=0):
#     """Print detailed memory stats"""
#     if local_rank == 0 and torch.cuda.is_available():
#         allocated = torch.cuda.memory_allocated() / 1024**3  # GB
#         reserved = torch.cuda.memory_reserved() / 1024**3
#         max_allocated = torch.cuda.max_memory_allocated() / 1024**3
#         print(f"[{stage}] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Peak: {max_allocated:.2f}GB")

# def alignment_training_loop(
#         model,
#         train_dl_dict,
#         loss_fn: callable,
#         foldername: str,
#         CONFIG,
#         GPU_intensive=False,
#         test_dl_dict=None,
#         local_rank=0,
#         train_samplers=None
#     ):
#     """this takes the config for model training and executes the training job

#     Args:
#         model (nn.Module): pytorch model to train. must take (input_batch, task) and output a return dict
#         train_dl_dict (dict): train split {t: DL for t in tasks}
#         loss_fn (callable): give (input, times, epoch) to get loss
#         foldername (str): folderpath for plotting and documentation
#         CONFIG (easydict): config EASY dict that follows format of ./configs.
#         GPU_intensive (bool): whether to be sparing with GPU memory or not

#     Returns:
#         None
#     """
#     logger.info(model)

#     #################################
#     ### write the config dict for documentation
#     num_epochs = CONFIG.NUM_EPOCHS
#     learning_rate = CONFIG.LEARNING_RATE
#     save_dict_to_json_file(CONFIG, foldername + '/config.json')

#     #################################
#     ### optimizer variable
#     optimizer = optim.Adam(
#         model.parameters(), lr=learning_rate
#     )

#     #################################
#     ### get checkpointing folder
#     ckpt_folder = foldername + '/' + 'ckpt'
#     validate_folder(ckpt_folder)
#     model_to_save = model.module if hasattr(model, 'module') else model

#     ckpt_save(
#         model_t=model_to_save,
#         optimizer_t=optimizer,
#         epoch_t=0,
#         loss_t=10000000,
#         filename=ckpt_folder + f'/epoch-0.pt',
#         config=CONFIG
#     )
#     train_loss_to_plot = []
#     epoch_losses_to_plot = []
#     more_epochs_bool = True
    
#     for epoch in range(num_epochs):
#         if not more_epochs_bool:
#             break

#         if train_samplers is not None:
#             for task, sampler in train_samplers.items():
#                 sampler.set_epoch(epoch)
                
#         running_loss = 0
#         start = time.time()
#         model.train()
#         all_sub_batches = _get_all_batches_with_taskid(train_dl_dict)
#         time_lengs = []
        
#         # DEBUG: Track what's happening
#         total_batches = 0
#         skipped_length = 0
#         skipped_exception = 0
#         successful_batches = 0
        
#         for i, (task, (inputs, times)) in enumerate(all_sub_batches):
#             total_batches += 1
#             s = time.time()
#             torch.cuda.empty_cache()

#             inputs, times = preprocess_batch(inputs, times, device=device if GPU_intensive else 'cpu', skip_rate=CONFIG.SKIP_RATE)

#             if any(seq.shape[0] < 2 for seq in inputs):
#                 skipped_length += 1
#                 if local_rank == 0 and i < 5:  # Log first few
#                     logger.warning(f"Skipping task {task}: Sequence length < 2 after preprocessing.")
#                 continue

#             output_dict = model(inputs)
#             del inputs

#             try:
#                 loss_dict = loss_fn(output_dict, epoch)

#                 if loss_dict is None or loss_dict['total_loss'] is None:
#                     raise ValueError("Loss calculation returned None")

#             except Exception as e:
#                 skipped_exception += 1
#                 if local_rank == 0 and skipped_exception < 5:  # Log first few
#                     logger.warning(f"Skipping batch in task {task} due to: {e}")
#                 continue  # REMOVED optimizer.zero_grad()

#             loss = loss_dict['total_loss']
#             if contains_non_float_values(loss):
#                 logger.error(f'Loss was NAN! exiting now')
#                 more_epochs_bool = False
#                 break
#             del output_dict
#             running_loss += loss.item()
#             train_loss_to_plot.append(loss.item())

#             # Original code - NO optimizer.zero_grad()
#             loss.backward()
#             del loss
#             nn.utils.clip_grad_norm_(model.parameters(), max_norm=.00001, norm_type=2)
#             optimizer.step()
#             time_lengs.append(time.time() - s)
#             successful_batches += 1
        
#         # Log statistics - only on rank 0
#         if local_rank == 0:
#             logger.info(f"Epoch {epoch+1} stats - Total: {total_batches}, Successful: {successful_batches}, Skipped (length): {skipped_length}, Skipped (exception): {skipped_exception}")
            
#             avg_loss = running_loss / successful_batches if successful_batches > 0 else 0.0
#             epoch_losses_to_plot.append(avg_loss)
#             _simple_loss_plot(
#                 epoch_losses_to_plot, 
#                 plot_title='Training Loss over epochs', 
#                 filename=f'{foldername}/train_loss_epochlevel.png', 
#                 condition=len(epoch_losses_to_plot) > 0,
#                 scatter=False
#             )
#             logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f} ({time.time() - start:.2f}s)")
        
#         # Checkpoint saving - only on rank 0
#         if (epoch % 5 == 0 or epoch >= num_epochs-3) and local_rank == 0:
#             model_to_save = model.module if hasattr(model, 'module') else model
#             ckpt_save(
#                 model_t=model_to_save,
#                 optimizer_t=optimizer,
#                 epoch_t=epoch,
#                 loss_t=running_loss / successful_batches if successful_batches > 0 else 0.0,
#                 filename=ckpt_folder + f'/epoch-{epoch+1}.pt',
#                 config=CONFIG
#             )

import gc

# Method-specific maximum pairwise products to prevent OOM in loss function
# Loss functions compute pairwise distance matrices between all video pairs
# Memory scales with max(Li × Lj), not sum of frames
# LAV uses SoftDTW which requires ~2-3x more memory than TCC/VAVA due to:
#   - Full (N+2)×(M+2) DP table allocation
#   - Intermediate R matrices stored for backward pass
#   - Additional N×N and M×M matrices for IntraContrast
MAX_PAIRWISE_THRESHOLDS = {
    'LAV': 4_000_000,       # LAV uses more memory due to SoftDTW
    'GTCC': 8_000_000,      # ~8M pairwise products max for 22.5GB GPU at 4fps
    'tcc': 8_000_000,       # Same as GTCC
    'VAVA': 6_000_000,      # VAVA uses moderate extra memory for OT
    'default': 8_000_000,   # Default threshold
}

def print_memory_stats(stage, local_rank=0):
    """Print memory stats at key checkpoints"""
    if local_rank == 0 and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"[MEM {stage}] Allocated: {allocated:.2f}GB")

def alignment_training_loop(
        model,
        train_dl_dict,
        loss_fn: callable,
        foldername: str,
        CONFIG,
        GPU_intensive=False,
        test_dl_dict=None,
        local_rank=0,
        train_samplers=None,
        loss_type=None
    ):
    """this takes the config for model training and executes the training job

    Args:
        model (nn.Module): pytorch model to train. must take (input_batch, task) and output a return dict
        train_dl_dict (dict): train split {t: DL for t in tasks}
        loss_fn (callable): give (input, times, epoch) to get loss
        foldername (str): folderpath for plotting and documentation
        CONFIG (easydict): config EASY dict that follows format of ./configs.
        GPU_intensive (bool): whether to be sparing with GPU memory or not

    Returns:
        None
    """
    logger.info(model)

    #################################
    ### write the config dict for documentation
    num_epochs = CONFIG.NUM_EPOCHS
    learning_rate = CONFIG.LEARNING_RATE
    save_dict_to_json_file(CONFIG, foldername + '/config.json')

    #################################
    ### optimizer variable
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate
    )

    #################################
    ### get checkpointing folder
    ckpt_folder = foldername + '/' + 'ckpt'
    validate_folder(ckpt_folder)
    model_to_save = model.module if hasattr(model, 'module') else model

    ckpt_save(
        model_t=model_to_save,
        optimizer_t=optimizer,
        epoch_t=0,
        loss_t=10000000,
        filename=ckpt_folder + f'/epoch-0.pt',
        config=CONFIG
    )
    train_loss_to_plot = []
    epoch_losses_to_plot = []
    more_epochs_bool = True
    
    for epoch in range(num_epochs):
        if not more_epochs_bool:
            break

        if train_samplers is not None:
            for task, sampler in train_samplers.items():
                sampler.set_epoch(epoch)
                
        running_loss = 0
        start = time.time()
        model.train()
        all_sub_batches = _get_all_batches_with_taskid(train_dl_dict)
        time_lengs = []
        
        # DEBUG: Track what's happening
        total_batches = 0
        skipped_length = 0
        skipped_exception = 0
        successful_batches = 0
        
        for i, (task, (inputs, times)) in enumerate(all_sub_batches):
            total_batches += 1
            s = time.time()
            optimizer.zero_grad(set_to_none=True)

            # Memory check at batch start
            print_memory_stats(f"Batch {i} START", local_rank)
            torch.cuda.empty_cache()

            inputs, times = preprocess_batch(inputs, times, device=device if GPU_intensive else 'cpu', skip_rate=CONFIG.SKIP_RATE)

            # Skip if sequences too short
            # [4FPS FIX] Increased minimum from 2 to 10 frames to avoid numerical issues
            # Videos with only 2-9 frames cause alignment loss to be ~0, leading to NaN/Inf
            MIN_SEQ_LENGTH = 10
            if any(seq.shape[0] < MIN_SEQ_LENGTH for seq in inputs):
                skipped_length += 1
                if local_rank == 0 and skipped_length <= 3:
                    logger.warning(f"[SKIP SHORT] Task: {task}, lengths={[seq.shape[0] for seq in inputs]}")
                del inputs
                continue

            # Skip if max pairwise product too large (PREVENTS OOM in loss function)
            # Loss functions compute N×M matrices for each video pair
            # Memory scales with the largest pairwise product, not sum of frames
            seq_lengths = sorted([seq.shape[0] for seq in inputs], reverse=True)
            # Max pairwise is the product of the two longest videos
            if len(seq_lengths) >= 2:
                max_pairwise = seq_lengths[0] * seq_lengths[1]
            else:
                max_pairwise = seq_lengths[0] ** 2

            # Use method-specific threshold (LAV needs stricter limit due to SoftDTW memory)
            max_pairwise_threshold = MAX_PAIRWISE_THRESHOLDS.get(loss_type, MAX_PAIRWISE_THRESHOLDS['default'])

            if max_pairwise > max_pairwise_threshold:
                skipped_length += 1
                if local_rank == 0:
                    logger.warning(
                        f"[SKIP LARGE] Task: {task}, max_pairwise={max_pairwise:,} > {max_pairwise_threshold:,} ({loss_type}), "
                        f"lengths={seq_lengths}"
                    )
                del inputs
                continue

            # Log batch info
            total_frames = sum(seq_lengths)
            if local_rank == 0:
                logger.info(f"[BATCH {i}] Task: {task}, lengths={seq_lengths}, total={total_frames}, max_pairwise={max_pairwise:,}")

            try:
                output_dict = model(inputs)
                print_memory_stats(f"After forward pass", local_rank)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if local_rank == 0:
                        logger.error(f"[OOM FORWARD] Task: {task} - skipping")
                    del inputs
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    skipped_exception += 1
                    continue
                raise
            del inputs

            try:
                loss_dict = loss_fn(output_dict, epoch)
                print_memory_stats(f"After loss computation", local_rank)

                if loss_dict is None or loss_dict['total_loss'] is None:
                    raise ValueError("Loss calculation returned None")

            except Exception as e:
                skipped_exception += 1
                if local_rank == 0 and skipped_exception < 5:
                    if "out of memory" in str(e).lower():
                        logger.error(f"[OOM LOSS] Task: {task}")
                    else:
                        logger.warning(f"Skipping batch in task {task}: {type(e).__name__}")
                try:
                    del output_dict
                except:
                    pass
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                continue

            loss = loss_dict['total_loss']
            if contains_non_float_values(loss):
                logger.error(f'Loss was NAN! exiting now')
                more_epochs_bool = False
                break
            del output_dict
            running_loss += loss.item()
            train_loss_to_plot.append(loss.item())

            if local_rank == 0:
                logger.info(f"[BATCH {i}] Loss: {loss.item():.4f}")

            # Backward pass
            try:
                loss.backward()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if local_rank == 0:
                        logger.error(f"[OOM BACKWARD] Task: {task} - skipping")
                    del loss
                    gc.collect()
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    skipped_exception += 1
                    continue
                raise

            del loss
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=.00001, norm_type=2)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            print_memory_stats(f"After step", local_rank)

            time_lengs.append(time.time() - s)
            successful_batches += 1
            torch.cuda.empty_cache()
        
        # Log statistics - only on rank 0
        if local_rank == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"Epoch {epoch+1} stats - Total: {total_batches}, Success: {successful_batches}, Skipped: {skipped_length}, OOM: {skipped_exception}")
            logger.info(f"Epoch {epoch+1} complete - Final memory: {allocated:.2f}GB")

            avg_loss = running_loss / successful_batches if successful_batches > 0 else 0.0
            epoch_losses_to_plot.append(avg_loss)
            _simple_loss_plot(
                epoch_losses_to_plot,
                plot_title='Training Loss over epochs',
                filename=f'{foldername}/train_loss_epochlevel.png',
                condition=len(epoch_losses_to_plot) > 0,
                scatter=False
            )
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f} ({time.time() - start:.2f}s)")
        
        # Checkpoint saving - only on rank 0
        if (epoch % 5 == 0 or epoch >= num_epochs-3) and local_rank == 0:
            model_to_save = model.module if hasattr(model, 'module') else model
            ckpt_save(
                model_t=model_to_save,
                optimizer_t=optimizer,
                epoch_t=epoch,
                loss_t=running_loss / successful_batches if successful_batches > 0 else 0.0,
                filename=ckpt_folder + f'/epoch-{epoch+1}.pt',
                config=CONFIG
            )
    # train_loss_to_plot = []
    # epoch_losses_to_plot = []
    # more_epochs_bool = True
    # for epoch in range(num_epochs):
    #     if not more_epochs_bool:
    #         break
    #     # ADD THIS: Set random seed for consistent shuffling across GPUs (optional but recommended)
    #     random.seed(42 + epoch)
    #     running_loss = 0
    #     start = time.time()
    #     model.train()
    #     all_sub_batches = _get_all_batches_with_taskid(train_dl_dict)
    #     time_lengs = []
    #     num_batches = 0
    #     # DEBUG: Track what's happening
    #     total_batches = 0
    #     skipped_length = 0
    #     skipped_exception = 0
    #     successful_batches = 0
    #     for i, (task, (inputs, times)) in enumerate(all_sub_batches):
    #         total_batches += 1
    #         s = time.time()
    #         torch.cuda.empty_cache()

    #         # process inputs, send through model
    #         inputs, times = preprocess_batch(inputs, times, device=device if GPU_intensive else 'cpu', skip_rate=CONFIG.SKIP_RATE)

    #         # 1. OPTIONAL PRE-CHECK: Fast skip if length is obviously too small
    #         if any(seq.shape[0] < 2 for seq in inputs):
    #             skipped_length += 1
    #             if local_rank == 0 and i < 5:  # Log first few
    #                 logger.warning(f"Skipping task {task}: Sequence length < 2 after preprocessing.")
    #             continue

    #         output_dict = model(inputs)
    #         del inputs

    #         try:
    #             # calculate loss
    #             loss_dict = loss_fn(output_dict, epoch)

    #             # If loss_fn returns None instead of throwing an error (depends on loss_fn logic)
    #             if loss_dict is None or loss_dict['total_loss'] is None:
    #                 raise ValueError("Loss calculation returned None")

    #         except Exception as e:
    #             skipped_exception += 1
    #             if local_rank == 0 and skipped_exception < 5:  # Log first few
    #                 logger.warning(f"Skipping batch in task {task} due to: {e}")
    #             optimizer.zero_grad() 
    #             continue

    #         # check + record loss
    #         loss = loss_dict['total_loss']
    #         if contains_non_float_values(loss):
    #             logger.error(f'Loss was NAN! exiting now')
    #             more_epochs_bool = False
    #             break
    #         del output_dict
    #         running_loss += loss.item()
    #         train_loss_to_plot.append(loss.item())

    #         # step update
    #         loss.backward()
    #         del loss
    #         nn.utils.clip_grad_norm_(model.parameters(), max_norm=.00001, norm_type=2)
    #         optimizer.step()
    #         time_lengs.append(time.time() - s)
    #         successful_batches += 1
    #         # print(np.mean(time_lengs) * (len(all_sub_batches) - i), end='\r')

    #     if local_rank == 0:
    #         logger.info(f"Epoch {epoch+1} stats - Total: {total_batches}, Successful: {successful_batches}, Skipped (length): {skipped_length}, Skipped (exception): {skipped_exception}")
    #         avg_loss = running_loss / successful_batches if successful_batches > 0 else 0.0
    #         epoch_losses_to_plot.append(avg_loss)
    #         _simple_loss_plot(
    #             epoch_losses_to_plot, 
    #             plot_title='Training Loss over epochs', 
    #             filename=f'{foldername}/train_loss_epochlevel.png', 
    #             condition=len(epoch_losses_to_plot) > 0,
    #             scatter=False
    #         )
    #         logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f} ({time.time() - start:.2f}s)")
    #     if (epoch % 5 == 0 or epoch >= num_epochs-3) and local_rank == 0:
    #         model_to_save = model.module if hasattr(model, 'module') else model
    #         ckpt_save(
    #             model_t=model_to_save,
    #             optimizer_t=optimizer,
    #             epoch_t=epoch,
    #             loss_t=running_loss / successful_batches if successful_batches > 0 else 0.0,
    #             filename=ckpt_folder + f'/epoch-{epoch+1}.pt',
    #             config=CONFIG
    #         )




def _get_all_batches_with_taskid(dl_dict):
    all_batches = [(task, (inputs, times)) for task, dl in dl_dict.items() for i, (inputs, times) in enumerate(dl)]
    random.shuffle(all_batches)
    return [b for b in all_batches]

def _simple_loss_plot(loss_list, plot_title, filename, condition, scatter=False):
    if condition:
        plt.clf()
        (plt.scatter if scatter else plt.plot)([i for i in range(len(loss_list))], loss_list)
        plt.title(plot_title)
        plt.savefig(filename)