import json
import math
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
    'LAV': 3_000_000,       # Lowered: LAV backward pass needs more memory than forward (OOM at 3.4M+)
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
        val_dl_dict=None,
        local_rank=0,
        train_samplers=None,
        loss_type=None
    ):
    """Training loop with validation support.

    Args:
        model (nn.Module): pytorch model to train. must take (input_batch, task) and output a return dict
        train_dl_dict (dict): train split {t: DL for t in tasks}
        loss_fn (callable): give (input, times, epoch) to get loss
        foldername (str): folderpath for plotting and documentation
        CONFIG (easydict): config EASY dict that follows format of ./configs.
        GPU_intensive (bool): whether to be sparing with GPU memory or not
        val_dl_dict (dict): validation split {t: DL for t in tasks} (optional)
        local_rank (int): GPU rank for DDP
        train_samplers (dict): DDP samplers for training
        loss_type (str): Type of loss function being used

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
    # Note: We no longer save an initial checkpoint here.
    # Checkpoints are saved only when validation loss improves (as best_model.pt)

    train_loss_to_plot = []
    epoch_losses_to_plot = []
    val_losses_to_plot = []
    best_val_loss = float('inf')
    best_val_loss_combined = float('inf')
    best_val_loss_alignment = float('inf')
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
                loss_dict = loss_fn(output_dict, epoch, times=times)
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

            # [FIX] Skip batches with extreme loss to prevent weight corruption
            # Normal GTCC loss is ~40k-250k. Values > 1e7 indicate numerical instability.
            MAX_SAFE_LOSS = 1e7
            if loss.item() > MAX_SAFE_LOSS:
                logger.warning(f"[EXTREME LOSS] {loss.item():.2e} > {MAX_SAFE_LOSS:.2e} - skipping batch to protect weights")
                del output_dict, loss_dict, loss
                gc.collect()
                torch.cuda.empty_cache()
                skipped_exception += 1
                continue

            if contains_non_float_values(loss):
                logger.error(f'Loss was NAN! exiting now')
                more_epochs_bool = False
                break
            del output_dict
            running_loss += loss.item()
            train_loss_to_plot.append(loss.item())

            if local_rank == 0:
                # Log total loss and breakdown if available
                log_msg = f"[BATCH {i}] Loss: {loss.item():.4f}"
                if 'alignment_loss' in loss_dict:
                    log_msg += f" | Align: {loss_dict['alignment_loss'].item():.4f}"
                if 'progress_loss' in loss_dict:
                    log_msg += f" | Progress: {loss_dict['progress_loss'].item():.6f}"
                logger.info(log_msg)

            # Backward pass
            try:
                loss.backward()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if local_rank == 0:
                        logger.error(f"[OOM BACKWARD] Task: {task} - skipping")
                    # CRITICAL: Delete all tensors that hold computation graph references
                    del loss
                    del loss_dict
                    # Clear all model gradients explicitly
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad = None
                    gc.collect()
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    skipped_exception += 1
                    continue
                raise

            # Fix: Rescale progress_head gradients to be unscaled by lambda
            # After backward with (alignment + lambda * progress), progress_head has lambda * grad
            # We divide by lambda so progress_head gets unscaled grad(progress)
            # Encoder keeps: grad(alignment) + lambda * grad(progress)
            progress_config = getattr(CONFIG, 'PROGRESS_LOSS', {})
            if progress_config.get('enabled', False) and progress_config.get('method') == 'learnable':
                progress_lambda = progress_config.get('lambda_fixed', 1.0)
                model_to_check = model.module if hasattr(model, 'module') else model
                if hasattr(model_to_check, 'progress_head') and progress_lambda != 1.0 and progress_lambda > 0:
                    for param in model_to_check.progress_head.parameters():
                        if param.grad is not None:
                            param.grad = param.grad / progress_lambda

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

        # === AGGRESSIVE MEMORY CLEANUP BEFORE VALIDATION ===
        # OOM errors during training can leave memory stuck in PyTorch's CUDA cache
        # This ensures we have maximum available memory for validation
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if local_rank == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"[PRE-VALIDATION] Memory after cleanup: {allocated:.2f}GB")

        # === VALIDATION AFTER EACH EPOCH ===
        if val_dl_dict is not None and local_rank == 0:
            val_loss = run_validation_epoch(
                model=model,
                val_dl_dict=val_dl_dict,
                loss_fn=loss_fn,
                epoch=epoch,
                config=CONFIG,
                local_rank=local_rank,
                loss_type=loss_type
            )
            val_losses_to_plot.append(val_loss)

            # Plot validation loss
            _simple_loss_plot(
                val_losses_to_plot,
                plot_title='Validation Loss over epochs',
                filename=f'{foldername}/val_loss_epochlevel.png',
                condition=len(val_losses_to_plot) > 0,
                scatter=False
            )

            # Plot combined train/val loss
            _plot_train_val_loss(
                epoch_losses_to_plot,
                val_losses_to_plot,
                filename=f'{foldername}/train_val_loss.png'
            )

            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}")

            # Get model to save
            model_to_save = model.module if hasattr(model, 'module') else model

            # Dual checkpoint system: track combined loss and alignment-only loss
            val_loss_combined = val_loss
            val_loss_alignment = val_loss  # Default: same as combined if no progress loss

            # [NaN PROTECTION] Skip checkpoint saving if loss is NaN/Inf
            def is_valid_loss(loss_val):
                return loss_val is not None and not math.isnan(loss_val) and not math.isinf(loss_val)

            # Save checkpoint only if validation loss improved (original behavior)
            if is_valid_loss(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f"New best validation loss: {val_loss:.4f} - saving checkpoint")
                ckpt_save(
                    model_t=model_to_save,
                    optimizer_t=optimizer,
                    epoch_t=epoch,
                    loss_t=val_loss,
                    filename=ckpt_folder + f'/best_model.pt',
                    config=CONFIG
                )
            elif not is_valid_loss(val_loss):
                logger.warning(f"[NaN DETECTED] val_loss={val_loss} - skipping checkpoint save")

            # Save best combined model (alignment + progress)
            if is_valid_loss(val_loss_combined) and val_loss_combined < best_val_loss_combined:
                best_val_loss_combined = val_loss_combined
                logger.info(f"New best COMBINED loss: {val_loss_combined:.4f}")
                ckpt_save(
                    model_t=model_to_save,
                    optimizer_t=optimizer,
                    epoch_t=epoch,
                    loss_t=val_loss_combined,
                    filename=ckpt_folder + '/best_model_combined.pt',
                    config=CONFIG
                )

            # Save best alignment-only model
            if is_valid_loss(val_loss_alignment) and val_loss_alignment < best_val_loss_alignment:
                best_val_loss_alignment = val_loss_alignment
                logger.info(f"New best ALIGNMENT loss: {val_loss_alignment:.4f}")
                ckpt_save(
                    model_t=model_to_save,
                    optimizer_t=optimizer,
                    epoch_t=epoch,
                    loss_t=val_loss_alignment,
                    filename=ckpt_folder + '/best_model_alignment.pt',
                    config=CONFIG
                )

        # Restore model to training mode for next epoch
        model.train()
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


def _plot_train_val_loss(train_losses, val_losses, filename):
    """Plot training and validation losses on the same figure."""
    plt.clf()
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    if val_losses:
        val_epochs = range(1, len(val_losses) + 1)
        plt.plot(val_epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(filename)


def run_validation_epoch(
    model,
    val_dl_dict: dict,
    loss_fn: callable,
    epoch: int,
    config,
    local_rank: int = 0,
    loss_type: str = None
) -> float:
    """
    Run validation for one epoch.

    CRITICAL: This function sets model.eval() and uses torch.no_grad().
    The caller must restore model.train() after this function returns.

    Args:
        model: The model (will be set to eval mode)
        val_dl_dict: Dict of validation dataloaders per task
        loss_fn: Loss function (same as training)
        epoch: Current epoch number
        config: Configuration object
        local_rank: GPU rank for DDP
        loss_type: Type of loss for threshold selection

    Returns:
        Average validation loss across all batches
    """
    # CRITICAL: Set model to evaluation mode
    model.eval()

    running_loss = 0.0
    successful_batches = 0
    skipped_batches = 0

    # CRITICAL: Disable gradient computation for validation
    with torch.no_grad():
        all_val_batches = _get_all_batches_with_taskid(val_dl_dict)

        for i, (task, (inputs, times)) in enumerate(all_val_batches):
            torch.cuda.empty_cache()

            inputs, times = preprocess_batch(
                inputs, times,
                device=device,
                skip_rate=config.SKIP_RATE
            )

            # Skip short sequences (same logic as training)
            MIN_SEQ_LENGTH = 10
            if any(seq.shape[0] < MIN_SEQ_LENGTH for seq in inputs):
                skipped_batches += 1
                del inputs
                continue

            # Skip if max pairwise product too large
            seq_lengths = sorted([seq.shape[0] for seq in inputs], reverse=True)
            if len(seq_lengths) >= 2:
                max_pairwise = seq_lengths[0] * seq_lengths[1]
            else:
                max_pairwise = seq_lengths[0] ** 2

            max_pairwise_threshold = MAX_PAIRWISE_THRESHOLDS.get(
                loss_type, MAX_PAIRWISE_THRESHOLDS['default']
            )

            if max_pairwise > max_pairwise_threshold:
                skipped_batches += 1
                del inputs
                continue

            try:
                # Forward pass (no gradients)
                output_dict = model(inputs)

                # Compute loss
                loss_dict = loss_fn(output_dict, epoch, times=times)

                if loss_dict is not None and loss_dict['total_loss'] is not None:
                    loss = loss_dict['total_loss']
                    if not contains_non_float_values(loss):
                        running_loss += loss.item()
                        successful_batches += 1

                del output_dict

            except Exception as e:
                skipped_batches += 1
                if local_rank == 0:
                    logger.warning(f"[VAL] Skipping batch in {task}: {type(e).__name__}")

            del inputs
            gc.collect()
            torch.cuda.empty_cache()

    avg_val_loss = running_loss / successful_batches if successful_batches > 0 else float('inf')

    if local_rank == 0:
        logger.info(f"[VAL] Batches: {successful_batches}, Skipped: {skipped_batches}, Avg Loss: {avg_val_loss:.4f}")

    return avg_val_loss