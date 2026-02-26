from utils.loss_functions import GTCC_loss, TCC_loss, LAV_loss, VAVA_loss
from utils.tensorops import (
    get_trueprogress_per_action,
    get_normalized_predicted_progress_action,
    sample_action_segment_with_random_index,
    sample_action_segment_with_multiple_frames
)

import torch
import random
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_loss_function(config_obj, num_epochs=None):
    loss_booldict = config_obj.LOSS_TYPE
    TCC_ORIGINAL_PARAMS = config_obj.TCC_ORIGINAL_PARAMS
    GTCC_PARAMS = config_obj.GTCC_PARAMS
    LAV_PARAMS = config_obj.LAV_PARAMS
    VAVA_PARAMS = config_obj.VAVA_PARAMS
    PROGRESS_CONFIG = getattr(config_obj, 'PROGRESS_LOSS', {'enabled': False})

    def _alignment_loss_fn(output_dict_list, epoch, times=None):
        if type(output_dict_list) != list:
            output_dict_list = [output_dict_list]
        ################################
        # Dict for returning loss results.
        ################################
        loss_return_dict = {}
        ################################
        # set some starter values
        ################################
        loss_return_dict['total_loss'] = torch.tensor(0).float().to(device)
        loss_return_dict['alignment_loss'] = torch.tensor(0).float().to(device)

        if PROGRESS_CONFIG.get('enabled', False):
            loss_return_dict['progress_loss'] = torch.tensor(0).float().to(device)

        for loss_term, verdict in loss_booldict.items():
            if verdict:
                loss_return_dict[loss_term + '_loss'] = torch.tensor(0).float().to(device)

        ################################
        # for each batch output.....
        ################################
        for idx, output_dict in enumerate(output_dict_list):
            if len(output_dict['outputs']) < 2:
                continue
            # check each loss term, should we add?? verdict will tell
            for loss_term, verdict in loss_booldict.items():
                if verdict:
                    coefficient = 1
                    if loss_term == 'GTCC':
                        specific_loss = GTCC_loss(
                            output_dict['outputs'],
                            dropouts=output_dict['dropouts'],
                            epoch=epoch,
                            **GTCC_PARAMS
                        )
                    elif loss_term == 'tcc':
                        specific_loss = TCC_loss(
                            output_dict['outputs'], **TCC_ORIGINAL_PARAMS
                        )
                    elif loss_term == 'LAV':
                        specific_loss = LAV_loss(
                            output_dict['outputs'], **LAV_PARAMS
                        )
                    elif loss_term == 'VAVA':
                        specific_loss = VAVA_loss(
                            output_dict['outputs'], global_step=epoch, **VAVA_PARAMS
                        )
                    else:
                        print(f"BAD LOSS TERM: {loss_term}, {verdict}")
                        exit(1)

                    loss_return_dict[loss_term + '_loss'] += specific_loss
                    loss_return_dict['alignment_loss'] += coefficient * specific_loss
                    loss_return_dict['total_loss'] += coefficient * specific_loss

            # Progress loss computation
            if PROGRESS_CONFIG.get('enabled', False) and times is not None:
                progress_lambda = PROGRESS_CONFIG.get('lambda_fixed', 0.1)
                method = PROGRESS_CONFIG.get('method', 'cumulative_l2')

                if method == 'cumulative_l2':
                    # Cumulative L2 method - sample multiple frames from ALL videos
                    # Only sample from non-background action frames
                    BACKGROUND_LABELS = ['0', 'SIL', 'background']
                    learnable_config = PROGRESS_CONFIG.get('learnable', {})
                    samples_per_video = learnable_config.get('samples_per_video', 5)
                    frames_per_segment = learnable_config.get('frames_per_segment', 3)
                    total_frames_per_video = samples_per_video * frames_per_segment  # ~15 frames per video

                    if len(output_dict['outputs']) > 0:
                        total_progress_loss = 0.0
                        num_samples = 0

                        # Process ALL videos in the batch
                        for vid_idx in range(len(output_dict['outputs'])):
                            if vid_idx >= len(times):
                                continue
                            video_features = output_dict['outputs'][vid_idx]
                            time_dict = copy.deepcopy(times[vid_idx])

                            T = video_features.shape[0]
                            if T >= 2:
                                time_dict['end_frame'][-1] = T - 1

                                # Compute progress for all frames once
                                pred_progress = get_normalized_predicted_progress_action(video_features, time_dict)
                                gt_progress = get_trueprogress_per_action(time_dict).to(video_features.device)

                                # Get all non-background frame indices
                                valid_frame_indices = []
                                for step, start, end in zip(
                                    time_dict['step'], time_dict['start_frame'], time_dict['end_frame']
                                ):
                                    if step not in BACKGROUND_LABELS:
                                        # Clamp to valid range
                                        start = max(0, min(start, T - 1))
                                        end = max(start, min(end, T - 1))
                                        valid_frame_indices.extend(range(start, end + 1))

                                if len(valid_frame_indices) > 0:
                                    # Sample multiple random frames from non-background actions
                                    num_frames_to_sample = min(total_frames_per_video, len(valid_frame_indices))
                                    frame_indices = random.sample(valid_frame_indices, num_frames_to_sample)

                                    for frame_idx in frame_indices:
                                        p_loss = torch.abs(pred_progress[frame_idx] - gt_progress[frame_idx])
                                        total_progress_loss = total_progress_loss + p_loss
                                        num_samples += 1

                        # Average the progress loss over all samples
                        if num_samples > 0:
                            avg_progress_loss = total_progress_loss / num_samples
                            loss_return_dict['progress_loss'] += avg_progress_loss
                            loss_return_dict['total_loss'] += progress_lambda * avg_progress_loss

                elif method == 'learnable' and 'progress_head' in output_dict:
                    # Learnable head method - sample multiple segments from ALL videos
                    # with multiple frames per segment for extra data augmentation
                    progress_head = output_dict['progress_head']
                    learnable_config = PROGRESS_CONFIG.get('learnable', {})
                    min_seg_len = learnable_config.get('min_segment_len', 3)
                    samples_per_video = learnable_config.get('samples_per_video', 5)
                    frames_per_segment = learnable_config.get('frames_per_segment', 3)

                    if len(output_dict['outputs']) > 0:
                        total_progress_loss = 0.0
                        num_samples = 0

                        # Process ALL videos in the batch
                        for vid_idx in range(len(output_dict['outputs'])):
                            if vid_idx >= len(times):
                                continue
                            vid_emb = output_dict['outputs'][vid_idx]

                            # Sample multiple segments per video (data augmentation)
                            for _ in range(samples_per_video):
                                # Get multiple (segment, gt_progress) pairs for different target frames
                                frame_samples = sample_action_segment_with_multiple_frames(
                                    vid_emb, times[vid_idx],
                                    min_segment_len=min_seg_len,
                                    frames_per_segment=frames_per_segment
                                )

                                # Process each target frame within the segment
                                for seg_emb, gt_prog in frame_samples:
                                    if seg_emb is not None and seg_emb.shape[0] >= 2:
                                        pred_prog = progress_head(seg_emb)
                                        gt_tensor = torch.tensor(gt_prog, device=pred_prog.device, dtype=pred_prog.dtype)
                                        p_loss = torch.abs(pred_prog - gt_tensor)
                                        total_progress_loss = total_progress_loss + p_loss
                                        num_samples += 1

                        # Average the progress loss over all samples
                        if num_samples > 0:
                            avg_progress_loss = total_progress_loss / num_samples
                            loss_return_dict['progress_loss'] += avg_progress_loss
                            loss_return_dict['total_loss'] += progress_lambda * avg_progress_loss

        return loss_return_dict
    return _alignment_loss_fn
