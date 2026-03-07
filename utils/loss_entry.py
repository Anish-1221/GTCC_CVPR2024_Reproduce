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
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BACKGROUND_LABELS = ['0', 'SIL', 'background']


def _action_label_to_idx(step):
    """Convert action label string to integer index. Background/unknown → 0."""
    if step in BACKGROUND_LABELS:
        return 0
    try:
        return int(step)
    except (ValueError, TypeError):
        return 0


def _compute_learnable_progress_loss(output_dict, times, progress_head, learnable_config):
    """
    Shared progress loss computation for learnable head.
    Dispatches to the appropriate loss mode based on learnable_config['progress_loss_mode'].

    Returns (progress_loss_tensor, num_samples) where progress_loss_tensor has grad_fn
    if num_samples > 0, otherwise returns (0.0, 0).
    """
    progress_loss_mode = learnable_config.get('progress_loss_mode', 'uniform_mono')
    min_seg_len = learnable_config.get('min_segment_len', 3)
    samples_per_video = learnable_config.get('samples_per_video', 10)
    frames_per_segment = learnable_config.get('frames_per_segment', 5)
    use_stratified = learnable_config.get('stratified_sampling', True)
    use_adaptive_frames = learnable_config.get('adaptive_frames', True)
    min_target_frames = learnable_config.get('min_target_frames', 5)
    max_target_frames = learnable_config.get('max_target_frames', 30)

    if progress_loss_mode == 'dense':
        return _progress_loss_dense(output_dict, times, progress_head, learnable_config)
    elif progress_loss_mode == 'legacy':
        return _progress_loss_legacy(output_dict, times, progress_head, learnable_config)
    elif progress_loss_mode == 'sqrt_weighted':
        return _progress_loss_sqrt_weighted(output_dict, times, progress_head, learnable_config)
    elif progress_loss_mode == 'mse':
        return _progress_loss_mse(output_dict, times, progress_head, learnable_config)
    else:  # 'uniform_mono' (default)
        return _progress_loss_uniform_mono(output_dict, times, progress_head, learnable_config)


def _get_progress_features(output_dict, cfg):
    """Get the correct feature tensor for progress head based on config."""
    use_raw = cfg.get('features', 'aligned') == 'raw'
    if use_raw and 'raw_features' in output_dict:
        return output_dict['raw_features']
    return output_dict['outputs']


def _progress_loss_dense(output_dict, times, progress_head, cfg):
    """Dense per-frame MSE: feed full action segment, supervise every frame.

    For each non-background action, feeds the complete action embeddings to the
    progress head with dense_output=True, getting per-frame predictions.
    Loss is MSE against linear ground truth [1/L, 2/L, ..., 1.0].
    """
    total_loss = 0.0
    num_actions = 0

    features = _get_progress_features(output_dict, cfg)
    for vid_idx in range(len(features)):
        if vid_idx >= len(times):
            continue
        vid_emb = features[vid_idx]
        vid_times = times[vid_idx]
        T = vid_emb.shape[0]

        for step, start, end in zip(
            vid_times['step'], vid_times['start_frame'], vid_times['end_frame']
        ):
            if step in BACKGROUND_LABELS:
                continue
            action_len = end - start + 1
            if action_len < 2:
                continue

            start_c = max(0, min(start, T - 1))
            end_c = max(start_c, min(end, T - 1))
            action_emb = vid_emb[start_c:end_c + 1]
            L = action_emb.shape[0]

            if L < 2:
                continue

            # Dense prediction: per-frame progress for the full action
            action_idx = _action_label_to_idx(step)
            pred = progress_head(action_emb, dense_output=True, action_idx=action_idx)  # (L,)

            # Ground truth: linear ramp [1/L, 2/L, ..., 1.0]
            gt = torch.arange(1, L + 1, device=pred.device, dtype=pred.dtype) / L

            total_loss = total_loss + torch.nn.functional.mse_loss(pred, gt)
            num_actions += 1

    if num_actions == 0:
        return 0.0, 0
    return total_loss / num_actions, num_actions


def _progress_loss_uniform_mono(output_dict, times, progress_head, cfg):
    """Option A: Uniform L1 + Monotonicity Penalty + Endpoint Regularization."""
    mono_weight = cfg.get('monotonicity_weight', 2.0)
    mono_margin = cfg.get('monotonicity_margin', 0.01)
    endpoint_weight = cfg.get('endpoint_weight', 1.0)
    min_seg_len = cfg.get('min_segment_len', 3)
    samples_per_video = cfg.get('samples_per_video', 10)
    use_stratified = cfg.get('stratified_sampling', True)
    use_adaptive_frames = cfg.get('adaptive_frames', True)
    min_target_frames = cfg.get('min_target_frames', 5)
    max_target_frames = cfg.get('max_target_frames', 30)
    frames_per_segment = cfg.get('frames_per_segment', 5)

    total_l1_loss = 0.0
    num_l1_samples = 0
    total_mono_loss = 0.0
    num_mono_pairs = 0
    total_endpoint_loss = 0.0
    num_endpoints = 0

    features = _get_progress_features(output_dict, cfg)
    for vid_idx in range(len(features)):
        if vid_idx >= len(times):
            continue
        vid_emb = features[vid_idx]
        vid_times = times[vid_idx]
        T = vid_emb.shape[0]

        # Endpoint regularization: last frame of each action → 1.0
        for step, start, end in zip(
            vid_times['step'], vid_times['start_frame'], vid_times['end_frame']
        ):
            if step in BACKGROUND_LABELS:
                continue
            action_len = end - start + 1
            if action_len < 2:
                continue
            start_c = max(0, min(start, T - 1))
            end_c = max(start_c, min(end, T - 1))
            full_emb = vid_emb[start_c:end_c + 1]
            if full_emb.shape[0] >= 2:
                ep_action_idx = _action_label_to_idx(step)
                pred_end = progress_head(full_emb, action_idx=ep_action_idx)
                total_endpoint_loss = total_endpoint_loss + torch.abs(pred_end - 1.0)
                num_endpoints += 1

        # Sample segments and compute L1 + monotonicity
        for _ in range(samples_per_video):
            frame_samples, sampled_action = sample_action_segment_with_multiple_frames(
                vid_emb, vid_times,
                min_segment_len=min_seg_len,
                frames_per_segment=frames_per_segment,
                stratified=use_stratified,
                adaptive_frames=use_adaptive_frames,
                min_target_frames=min_target_frames,
                max_target_frames=max_target_frames,
                return_action_name=True
            )
            sampled_action_idx = _action_label_to_idx(sampled_action) if sampled_action else 0

            # Collect predictions for monotonicity (all from same action, sorted by time)
            action_preds = []
            for seg_emb, gt_prog in frame_samples:
                if seg_emb is not None and seg_emb.shape[0] >= 2:
                    pred_prog = progress_head(seg_emb, action_idx=sampled_action_idx)
                    gt_tensor = torch.tensor(gt_prog, device=pred_prog.device, dtype=pred_prog.dtype)

                    # Uniform L1
                    total_l1_loss = total_l1_loss + torch.abs(pred_prog - gt_tensor)
                    num_l1_samples += 1
                    action_preds.append((gt_prog, pred_prog))

            # Monotonicity penalty: consecutive predictions should increase
            if len(action_preds) >= 2:
                action_preds.sort(key=lambda x: x[0])
                for j in range(1, len(action_preds)):
                    pred_earlier = action_preds[j - 1][1]
                    pred_later = action_preds[j][1]
                    violation = pred_earlier - pred_later + mono_margin
                    total_mono_loss = total_mono_loss + torch.clamp(violation, min=0)
                    num_mono_pairs += 1

    # Combine losses
    if num_l1_samples == 0:
        return 0.0, 0

    avg_l1 = total_l1_loss / num_l1_samples
    total_loss = avg_l1

    if num_mono_pairs > 0:
        avg_mono = total_mono_loss / num_mono_pairs
        total_loss = total_loss + mono_weight * avg_mono

    if num_endpoints > 0:
        avg_endpoint = total_endpoint_loss / num_endpoints
        total_loss = total_loss + endpoint_weight * avg_endpoint

    return total_loss, num_l1_samples


def _progress_loss_sqrt_weighted(output_dict, times, progress_head, cfg):
    """Option B: Sqrt weighting + reduced boundary loss."""
    weight_cap = cfg.get('weight_cap', 5.0)
    boundary_weight = cfg.get('boundary_weight', 1.0)
    min_seg_len = cfg.get('min_segment_len', 3)
    samples_per_video = cfg.get('samples_per_video', 10)
    use_stratified = cfg.get('stratified_sampling', True)
    use_adaptive_frames = cfg.get('adaptive_frames', True)
    min_target_frames = cfg.get('min_target_frames', 5)
    max_target_frames = cfg.get('max_target_frames', 30)
    frames_per_segment = cfg.get('frames_per_segment', 5)

    total_progress_loss = 0.0
    num_samples = 0
    total_boundary_loss = 0.0
    num_boundary_samples = 0

    features = _get_progress_features(output_dict, cfg)
    for vid_idx in range(len(features)):
        if vid_idx >= len(times):
            continue
        vid_emb = features[vid_idx]
        vid_times = times[vid_idx]
        T = vid_emb.shape[0]

        # Boundary loss (reduced weight)
        for step, start, end in zip(
            vid_times['step'], vid_times['start_frame'], vid_times['end_frame']
        ):
            if step in BACKGROUND_LABELS:
                continue
            action_len = end - start + 1
            if action_len < 2:
                continue
            start_c = max(0, min(start, T - 1))
            end_c = max(start_c, min(end, T - 1))

            bnd_action_idx = _action_label_to_idx(step)
            first_emb = vid_emb[start_c:start_c + 1]
            if first_emb.shape[0] >= 1:
                pred_first = progress_head(first_emb, action_idx=bnd_action_idx)
                gt_first = 1.0 / action_len
                total_boundary_loss = total_boundary_loss + boundary_weight * torch.abs(pred_first - gt_first)
                num_boundary_samples += 1

            full_emb = vid_emb[start_c:end_c + 1]
            if full_emb.shape[0] >= 2:
                pred_last = progress_head(full_emb, action_idx=bnd_action_idx)
                total_boundary_loss = total_boundary_loss + boundary_weight * torch.abs(pred_last - 1.0)
                num_boundary_samples += 1

        # Sampled segments with sqrt weighting
        for _ in range(samples_per_video):
            frame_samples, sampled_action = sample_action_segment_with_multiple_frames(
                vid_emb, vid_times,
                min_segment_len=min_seg_len,
                frames_per_segment=frames_per_segment,
                stratified=use_stratified,
                adaptive_frames=use_adaptive_frames,
                min_target_frames=min_target_frames,
                max_target_frames=max_target_frames,
                return_action_name=True
            )
            sampled_action_idx = _action_label_to_idx(sampled_action) if sampled_action else 0

            for seg_emb, gt_prog in frame_samples:
                if seg_emb is not None and seg_emb.shape[0] >= 2:
                    pred_prog = progress_head(seg_emb, action_idx=sampled_action_idx)
                    gt_tensor = torch.tensor(gt_prog, device=pred_prog.device, dtype=pred_prog.dtype)

                    # Sqrt weighting: 1/sqrt(gt_prog) instead of 1/gt_prog
                    weight = 1.0 / math.sqrt(max(gt_prog, 1.0 / weight_cap))
                    weight = min(weight, weight_cap)

                    p_loss = weight * torch.abs(pred_prog - gt_tensor)
                    total_progress_loss = total_progress_loss + p_loss
                    num_samples += 1

    if num_samples == 0:
        return 0.0, 0

    avg_loss = total_progress_loss / num_samples
    if num_boundary_samples > 0:
        avg_loss = avg_loss + total_boundary_loss / num_boundary_samples

    return avg_loss, num_samples


def _progress_loss_mse(output_dict, times, progress_head, cfg):
    """Option C: MSE loss + endpoint regularization."""
    endpoint_weight = cfg.get('endpoint_weight', 1.0)
    min_seg_len = cfg.get('min_segment_len', 3)
    samples_per_video = cfg.get('samples_per_video', 10)
    use_stratified = cfg.get('stratified_sampling', True)
    use_adaptive_frames = cfg.get('adaptive_frames', True)
    min_target_frames = cfg.get('min_target_frames', 5)
    max_target_frames = cfg.get('max_target_frames', 30)
    frames_per_segment = cfg.get('frames_per_segment', 5)

    total_mse_loss = 0.0
    num_samples = 0
    total_endpoint_loss = 0.0
    num_endpoints = 0

    features = _get_progress_features(output_dict, cfg)
    for vid_idx in range(len(features)):
        if vid_idx >= len(times):
            continue
        vid_emb = features[vid_idx]
        vid_times = times[vid_idx]
        T = vid_emb.shape[0]

        # Endpoint regularization
        for step, start, end in zip(
            vid_times['step'], vid_times['start_frame'], vid_times['end_frame']
        ):
            if step in BACKGROUND_LABELS:
                continue
            action_len = end - start + 1
            if action_len < 2:
                continue
            start_c = max(0, min(start, T - 1))
            end_c = max(start_c, min(end, T - 1))
            mse_ep_action_idx = _action_label_to_idx(step)
            full_emb = vid_emb[start_c:end_c + 1]
            if full_emb.shape[0] >= 2:
                pred_end = progress_head(full_emb, action_idx=mse_ep_action_idx)
                total_endpoint_loss = total_endpoint_loss + (pred_end - 1.0) ** 2
                num_endpoints += 1

        # Sampled segments with MSE
        for _ in range(samples_per_video):
            frame_samples, sampled_action = sample_action_segment_with_multiple_frames(
                vid_emb, vid_times,
                min_segment_len=min_seg_len,
                frames_per_segment=frames_per_segment,
                stratified=use_stratified,
                adaptive_frames=use_adaptive_frames,
                min_target_frames=min_target_frames,
                max_target_frames=max_target_frames,
                return_action_name=True
            )
            sampled_action_idx = _action_label_to_idx(sampled_action) if sampled_action else 0

            for seg_emb, gt_prog in frame_samples:
                if seg_emb is not None and seg_emb.shape[0] >= 2:
                    pred_prog = progress_head(seg_emb, action_idx=sampled_action_idx)
                    gt_tensor = torch.tensor(gt_prog, device=pred_prog.device, dtype=pred_prog.dtype)

                    p_loss = (pred_prog - gt_tensor) ** 2
                    total_mse_loss = total_mse_loss + p_loss
                    num_samples += 1

    if num_samples == 0:
        return 0.0, 0

    avg_mse = total_mse_loss / num_samples
    total_loss = avg_mse

    if num_endpoints > 0:
        avg_endpoint = total_endpoint_loss / num_endpoints
        total_loss = total_loss + endpoint_weight * avg_endpoint

    return total_loss, num_samples


def _progress_loss_legacy(output_dict, times, progress_head, cfg):
    """Legacy: weighted L1 + boundary loss (original behavior)."""
    use_weighted_loss = cfg.get('weighted_loss', True)
    weight_cap = cfg.get('weight_cap', 20.0)
    use_boundary_loss = cfg.get('boundary_loss', True)
    boundary_weight = cfg.get('boundary_weight', 5.0)
    min_seg_len = cfg.get('min_segment_len', 3)
    samples_per_video = cfg.get('samples_per_video', 10)
    use_stratified = cfg.get('stratified_sampling', True)
    use_adaptive_frames = cfg.get('adaptive_frames', True)
    min_target_frames = cfg.get('min_target_frames', 5)
    max_target_frames = cfg.get('max_target_frames', 30)
    frames_per_segment = cfg.get('frames_per_segment', 5)

    total_progress_loss = 0.0
    total_weight = 0.0
    num_samples = 0
    total_boundary_loss = 0.0
    num_boundary_samples = 0

    features = _get_progress_features(output_dict, cfg)
    for vid_idx in range(len(features)):
        if vid_idx >= len(times):
            continue
        vid_emb = features[vid_idx]
        vid_times = times[vid_idx]
        T = vid_emb.shape[0]

        if use_boundary_loss:
            for step, start, end in zip(
                vid_times['step'], vid_times['start_frame'], vid_times['end_frame']
            ):
                if step in BACKGROUND_LABELS:
                    continue
                action_len = end - start + 1
                if action_len < 2:
                    continue
                start_c = max(0, min(start, T - 1))
                end_c = max(start_c, min(end, T - 1))
                leg_action_idx = _action_label_to_idx(step)

                first_emb = vid_emb[start_c:start_c + 1]
                if first_emb.shape[0] >= 1:
                    pred_first = progress_head(first_emb, action_idx=leg_action_idx)
                    gt_first = 1.0 / action_len
                    total_boundary_loss += boundary_weight * torch.abs(pred_first - gt_first)
                    num_boundary_samples += 1

                full_emb = vid_emb[start_c:end_c + 1]
                if full_emb.shape[0] >= 2:
                    pred_last = progress_head(full_emb, action_idx=leg_action_idx)
                    total_boundary_loss += boundary_weight * torch.abs(pred_last - 1.0)
                    num_boundary_samples += 1

        for _ in range(samples_per_video):
            frame_samples, sampled_action = sample_action_segment_with_multiple_frames(
                vid_emb, vid_times,
                min_segment_len=min_seg_len,
                frames_per_segment=frames_per_segment,
                stratified=use_stratified,
                adaptive_frames=use_adaptive_frames,
                min_target_frames=min_target_frames,
                max_target_frames=max_target_frames,
                return_action_name=True
            )
            sampled_action_idx = _action_label_to_idx(sampled_action) if sampled_action else 0

            for seg_emb, gt_prog in frame_samples:
                if seg_emb is not None and seg_emb.shape[0] >= 2:
                    pred_prog = progress_head(seg_emb, action_idx=sampled_action_idx)
                    gt_tensor = torch.tensor(gt_prog, device=pred_prog.device, dtype=pred_prog.dtype)

                    if use_weighted_loss:
                        weight = 1.0 / max(gt_prog, 1.0 / weight_cap)
                        weight = min(weight, weight_cap)
                    else:
                        weight = 1.0

                    p_loss = weight * torch.abs(pred_prog - gt_tensor)
                    total_progress_loss = total_progress_loss + p_loss
                    total_weight += weight
                    num_samples += 1

    if num_samples == 0:
        return 0.0, 0

    if use_weighted_loss and total_weight > 0:
        avg_loss = total_progress_loss / total_weight
    else:
        avg_loss = total_progress_loss / num_samples

    if use_boundary_loss and num_boundary_samples > 0:
        avg_loss = avg_loss + total_boundary_loss / num_boundary_samples

    return avg_loss, num_samples

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
                    learnable_config = PROGRESS_CONFIG.get('learnable', {})
                    progress_loss, n_samples = _compute_learnable_progress_loss(
                        output_dict, times, output_dict['progress_head'], learnable_config
                    )
                    if n_samples > 0:
                        loss_return_dict['progress_loss'] += progress_loss
                        loss_return_dict['total_loss'] += progress_lambda * progress_loss

        return loss_return_dict
    return _alignment_loss_fn


def get_progress_only_loss_function(config_obj):
    """
    Loss function that computes ONLY the learnable progress loss.
    No alignment loss, no lambda multiplier.
    Used with progress-head-only training mode (encoder frozen).
    """
    PROGRESS_CONFIG = getattr(config_obj, 'PROGRESS_LOSS', {'enabled': False})

    def _progress_only_loss_fn(output_dict_list, epoch, times=None):
        if type(output_dict_list) != list:
            output_dict_list = [output_dict_list]

        loss_return_dict = {
            'total_loss': torch.tensor(0).float().to(device),
            'progress_loss': torch.tensor(0).float().to(device),
        }

        if times is None:
            return loss_return_dict

        for idx, output_dict in enumerate(output_dict_list):
            if len(output_dict['outputs']) < 2:
                continue
            if 'progress_head' not in output_dict:
                continue

            learnable_config = PROGRESS_CONFIG.get('learnable', {})
            progress_loss, n_samples = _compute_learnable_progress_loss(
                output_dict, times, output_dict['progress_head'], learnable_config
            )

            if n_samples > 0:
                loss_return_dict['progress_loss'] += progress_loss
                loss_return_dict['total_loss'] += progress_loss

        return loss_return_dict
    return _progress_only_loss_fn
