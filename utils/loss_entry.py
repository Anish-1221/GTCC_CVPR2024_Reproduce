from utils.loss_functions import GTCC_loss, TCC_loss, LAV_loss, VAVA_loss
from utils.tensorops import (
    get_trueprogress_per_action,
    get_normalized_predicted_progress_action,
    sample_action_segment_with_random_index
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
                    # Cumulative L2 method - action level
                    if len(output_dict['outputs']) > 0:
                        video_idx = random.randint(0, len(output_dict['outputs']) - 1)
                        video_features = output_dict['outputs'][video_idx]
                        time_dict = copy.deepcopy(times[video_idx])

                        T = video_features.shape[0]
                        if T >= 2:
                            frame_idx = random.randint(0, T - 1)
                            time_dict['end_frame'][-1] = T - 1

                            pred_progress = get_normalized_predicted_progress_action(video_features, time_dict)
                            gt_progress = get_trueprogress_per_action(time_dict).to(video_features.device)

                            p_loss = torch.abs(pred_progress[frame_idx] - gt_progress[frame_idx])
                            loss_return_dict['progress_loss'] += p_loss
                            loss_return_dict['total_loss'] += progress_lambda * p_loss

                elif method == 'learnable' and 'progress_head' in output_dict:
                    # Learnable head method - action level (one sample per batch)
                    progress_head = output_dict['progress_head']
                    learnable_config = PROGRESS_CONFIG.get('learnable', {})
                    min_seg_len = learnable_config.get('min_segment_len', 3)

                    if len(output_dict['outputs']) > 0:
                        # Sample ONE random video from the batch
                        vid_idx = random.randint(0, len(output_dict['outputs']) - 1)
                        if vid_idx < len(times):
                            vid_emb = output_dict['outputs'][vid_idx]

                            seg_emb, gt_prog, _ = sample_action_segment_with_random_index(
                                vid_emb, times[vid_idx], min_segment_len=min_seg_len
                            )

                            if seg_emb is not None and seg_emb.shape[0] >= 2:
                                pred_prog = progress_head(seg_emb)
                                gt_tensor = torch.tensor(gt_prog, device=pred_prog.device, dtype=pred_prog.dtype)
                                p_loss = torch.abs(pred_prog - gt_tensor)
                                loss_return_dict['progress_loss'] += p_loss
                                loss_return_dict['total_loss'] += progress_lambda * p_loss

        return loss_return_dict
    return _alignment_loss_fn
