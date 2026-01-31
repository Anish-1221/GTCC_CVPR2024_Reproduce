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

def print_memory_stats(stage, local_rank=0):
    """Print detailed memory stats"""
    if local_rank == 0 and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[MEM {stage}] Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Peak: {max_allocated:.2f}GB")

def alignment_training_loop(
        model,
        train_dl_dict,
        loss_fn: callable,
        foldername: str,
        CONFIG,
        GPU_intensive=False,
        test_dl_dict=None,
        local_rank=0,
        train_samplers=None
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
            
            # DEBUG: Print task info
            if local_rank == 0:
                logger.info(f"[BATCH {i}] Task: {task}")
            
            print_memory_stats(f"Batch {i} START - Task {task}", local_rank)
            torch.cuda.empty_cache()

            inputs, times = preprocess_batch(inputs, times, device=device if GPU_intensive else 'cpu', skip_rate=CONFIG.SKIP_RATE)
            print_memory_stats(f"After preprocess", local_rank)

            if any(seq.shape[0] < 2 for seq in inputs):
                skipped_length += 1
                if local_rank == 0 and i < 5:  # Log first few
                    logger.warning(f"Skipping task {task}: Sequence length < 2 after preprocessing.")
                continue

            # DEBUG: Check sequence info AFTER preprocessing
            if local_rank == 0:
                seq_lengths = [seq.shape[0] for seq in inputs]
                logger.info(f"[BATCH {i}] After preprocessing - Num sequences: {len(inputs)}, Lengths: {seq_lengths}")

            output_dict = model(inputs)
            print_memory_stats(f"After forward pass", local_rank)
            del inputs

            try:
                loss_dict = loss_fn(output_dict, epoch)
                print_memory_stats(f"After loss computation", local_rank)

                if loss_dict is None or loss_dict['total_loss'] is None:
                    raise ValueError("Loss calculation returned None")

            except Exception as e:
                skipped_exception += 1
                if local_rank == 0 and skipped_exception < 5:  # Log first few
                    logger.warning(f"Skipping batch in task {task} due to: {e}")
                    if "out of memory" in str(e):
                        try:
                            logger.error(f"[OOM] Task: {task}, Sequences: {len(output_dict['outputs'])}, Lengths: {[o.shape[0] for o in output_dict['outputs']]}")
                        except:
                            logger.error(f"[OOM] Task: {task} (could not get sequence info)")
                torch.cuda.empty_cache()
                gc.collect()
                continue

            loss = loss_dict['total_loss']
            if contains_non_float_values(loss):
                logger.error(f'Loss was NAN! exiting now')
                more_epochs_bool = False
                break
            del output_dict
            running_loss += loss.item()
            train_loss_to_plot.append(loss.item())

            # DEBUG: Log loss value
            if local_rank == 0:
                logger.info(f"[BATCH {i}] Loss value: {loss.item():.4f}")

            print_memory_stats(f"Before backward", local_rank)
            
            # DEBUG: Wrap backward to catch OOM specifically here
            try:
                loss.backward()
                print_memory_stats(f"After backward", local_rank)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if local_rank == 0:
                        logger.error(f"[OOM in BACKWARD] Task: {task}, Loss value: {loss.item():.4f}")
                    torch.cuda.empty_cache()
                    gc.collect()
                    skipped_exception += 1
                    del loss
                    continue
                else:
                    raise
            
            del loss
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=.00001, norm_type=2)
            optimizer.step()
            # ALSO ADD: Zero gradients AFTER step
            optimizer.zero_grad(set_to_none=True)
            print_memory_stats(f"After optimizer step", local_rank)
            
            time_lengs.append(time.time() - s)
            successful_batches += 1
            
            # DEBUG: Clear cache after successful batch
            torch.cuda.empty_cache()
        
        # Log statistics - only on rank 0
        if local_rank == 0:
            logger.info(f"Epoch {epoch+1} stats - Total: {total_batches}, Successful: {successful_batches}, Skipped (length): {skipped_length}, Skipped (exception): {skipped_exception}")
            
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