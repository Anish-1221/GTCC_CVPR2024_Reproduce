import os
import random
import shutil
import matplotlib.pyplot as plt
import json

from scipy.stats import kendalltau
from external_util.ot_pytorch import sink
import torch.nn as nn
from models.json_dataset import data_json_labels_handles
from utils.os_util import get_env_variable
from utils.plotter import validate_folder
from utils.trainers import train_and_evaluate_svm, train_linear_regressor, svm_normalize_embedded_dl
from utils.tensorops import compute_eae_between_dict_vids, contains_non_float_values, flatten_dataloader_and_get_dict, get_average_train_cum_distance, get_cum_matrix, get_target_alignment_with_dict, get_trueprogress, preprocess_batch, get_trueprogress_per_action
from utils.logging import configure_logging_format
import torch
import numpy as np

logger = configure_logging_format()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PhaseProgression:
    def __init__(self) -> None:
        self.name = 'phaseprogression'
    
    def __str__(self):
        return self.name

    def __call__(self, model, config_obj, epoch, test_dataloaders, testfolder, tasks, normalize=False):
        for task in tasks:
            embedded_dl = flatten_dataloader_and_get_dict(model, test_dataloaders[task], config_obj.SKIP_RATE, device='cpu')
            if normalize:
                embedded_dl = svm_normalize_embedded_dl(embedded_dl=embedded_dl)
            X = []
            y = []
            for i, (outputs_dict, tdict) in enumerate(embedded_dl):
                outputs = outputs_dict['outputs']
                if contains_non_float_values(outputs):
                    continue
                true_prog = get_trueprogress(tdict).detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
                for k, frame in enumerate(outputs):
                    X.append(frame)
                    y.append(true_prog[k])
            _, r2 = train_linear_regressor(X, y, normalize=normalize)
            
            if len(tasks) == 1:
                return {'task': task, 'phase_prog': r2}

class PhaseClassification:
    def __init__(self) -> None:
        self.name = 'phaseclassification'
    
    def __str__(self):
        return self.name

    def __call__(self, model, config_obj, epoch, test_dataloaders, testfolder, tasks, normalize=True):
        percents = [.1, .5, 1]
        df = {'task': []}
        for percent in percents:
            df[f'phase_classification_ovo_{percent}'] = []

        for task in tasks:
            # first get set of all actions in the test set
            action_set = set()
            num_videos = 0
            num_data_points = 0
            embedded_dl = flatten_dataloader_and_get_dict(model, test_dataloaders[task], config_obj.SKIP_RATE, device='cpu')
            for i, (_, tdict) in enumerate(embedded_dl):
                num_videos += 1
                list(map(action_set.add, tdict['step']))
                num_data_points += tdict['end_frame'][-1]+1
            id_to_action = dict(list(zip(range(len(action_set)), action_set)))
            action_to_id = {v: k for k, v in id_to_action.items()}

            X_set = []
            Y_set = []
            for i, (outputs_dict, tdict) in enumerate(embedded_dl):
                outputs = outputs_dict['outputs']
                for t in range(outputs.shape[0]):
                    X = outputs[t]
                    Y = None
                    for step, start, end in zip(tdict['step'], tdict['start_frame'], tdict['end_frame']):
                        if start <= t <= end:
                            Y = action_to_id[step]
                    if not contains_non_float_values(X) and Y is not None:
                        X_set.append(X.clone().detach().cpu().numpy())
                        Y_set.append(Y)
            # shuffle
            if len(X_set) == 0:
                return None

            x_y_pairs = list(zip(X_set, Y_set))
            random.shuffle(x_y_pairs)
            X_set_r, Y_set_r = map(list, list(zip(*x_y_pairs)))

            if contains_non_float_values(X_set_r):
                print("X_set_r")
                exit(1)
            if contains_non_float_values(Y_set_r):
                print(Y_set_r)
                exit(1)
            df['task'] = task
            for percent in percents:
                barrier = round(percent * len(X_set_r))
                trained_classifier, accuracy = train_and_evaluate_svm(X_set_r[:barrier], Y_set_r[:barrier], X_set_r, Y_set_r, normalize=normalize)
                df[f'phase_classification_ovo_{percent}'] = accuracy

            if len(tasks) == 1:
                return df

class KendallsTau:
    def __init__(self) -> None:
        self.name = 'KT'
    
    def __str__(self):
        return self.name

    def __call__(self, model, config_obj, epoch, test_dataloaders, testfolder, tasks, normalize=False):
        for task in tasks:
            sci_ktau_list = []
            embedded_dl = flatten_dataloader_and_get_dict(model, test_dataloaders[task], config_obj.SKIP_RATE, device='cpu')
            if normalize:
                embedded_dl = svm_normalize_embedded_dl(embedded_dl=embedded_dl)
            for i, (odict1, _) in enumerate(embedded_dl):
                v1 = odict1['outputs']
                N = v1.shape[0]
                for j, (odict2, _) in enumerate(embedded_dl):
                    if i == j:
                        continue
                    v2 = odict2['outputs']
                    neighbors = torch.cdist(v1, v2).argmin(dim=1).detach().cpu().numpy()
                    ktau_sci = kendalltau(neighbors, np.arange(N)).statistic
                    sci_ktau_list.append(ktau_sci)

            if len(tasks) == 1:
                return {'task': task, 'sci-KT': np.mean(sci_ktau_list)}

class WildKendallsTau:
    def __init__(self) -> None:
        self.name = 'AKT'
    
    def __str__(self):
        return self.name

    def __call__(self, model, config_obj, epoch, test_dataloaders, testfolder, tasks, normalize=False):
        for task in tasks:
            ktau_list = []
            sci_ktau_list = []
            for (inputs, times) in test_dataloaders[task]:
                inputs, times = preprocess_batch(inputs, times, skip_rate=config_obj.SKIP_RATE)
                with torch.no_grad():
                    outputs_dict = model(inputs)
                del inputs
                outputs = outputs_dict['outputs']
                for i, (v1, t1) in enumerate(zip(outputs, times)):
                    N = v1.shape[0]
                    for j, (v2, t2) in enumerate(zip(outputs, times)):
                        if i == j:
                            continue
                        neighbors = torch.cdist(v1, v2).argmin(dim=1).detach().cpu().numpy()
                        true_alignment = get_target_alignment_with_dict(v1.shape[0], v2.shape[0], t1, t2).to(device)
                        indices = np.nonzero(true_alignment).clone().cpu().numpy().T
                        align_options = {}
                        for a, b in zip(indices[0], indices[1]):
                            if a not in align_options.keys():
                                align_options[a] = [b]
                            else:
                                align_options[a].append(b)
                        concord_pairs = 0
                        disconcord_pairs = 0
                        for i in range(N-1):
                            if i not in align_options.keys():
                                continue #SIL
                            p_options = align_options[i]
                            for j in range(i+1, N):
                                if j not in align_options.keys():
                                    continue #SIL
                                q = neighbors[j]
                                concord = False
                                for p in p_options:
                                    if p < q:
                                        concord = True
                                        continue
                                if concord:
                                    concord_pairs += 1
                                else:
                                    disconcord_pairs += 1

                        ktau = (concord_pairs - disconcord_pairs) / ((N * (N-1)) / 2)
                        ktau_list.append(ktau)
            if len(tasks) == 1:
                return {'task': task, 'my-KT': np.mean(ktau_list)}

class EnclosedAreaError:
    def __init__(self) -> None:
        self.name = 'eae'
    
    def __str__(self):
        return self.name

    def __call__(self, model, config_obj, epoch, test_dataloaders, testfolder, tasks, normalize=False):
        results = {'task': [], 'eae': []}
        for task in tasks:
            eae_list = []
            embedded_dl = flatten_dataloader_and_get_dict(model, test_dataloaders[task], config_obj.SKIP_RATE, device='cpu')
            if normalize:
                embedded_dl = svm_normalize_embedded_dl(embedded_dl=embedded_dl)
            for i, (odict1, t1) in enumerate(embedded_dl):
                v1 = odict1['outputs']
                for j, (odict2, t2) in enumerate(embedded_dl):
                    if i == j:
                        continue
                    v2 = odict2['outputs']
                    eae = compute_eae_between_dict_vids(v1, v2, t1, t2, wild=config_obj.DATASET_NAME in ['egoprocel']) # here add in-the-wild datasets
                    if eae is not None:
                        eae_list.append(eae)

            results['task'].append(task)
            results['eae'].append(np.mean(eae_list))
            if len(tasks) == 1:
                return {'task': task, 'eae': np.mean(eae_list)}

class OnlineGeoProgressError:
    def __init__(self) -> None:
        self.name = 'ogpe'
    
    def __str__(self):
        return self.name

    def __call__(self, model, config_obj, epoch, test_dataloaders, testfolder, tasks, normalize=False, plotting=False):
        # [4FPS FIX] Detect if model is 4fps based on folder path
        is_4fps = '4fps' in testfolder

        if is_4fps:
            PROTAS_BASE = '/vision/anishn/ProTAS/data_4fps/'
            print(f"[INFO] Detected 4fps model, using 4fps paths")
        else:
            PROTAS_BASE = '/vision/anishn/ProTAS/data_1fps/'

        plot_folder = testfolder + '/plotting_progress'
        validate_folder(plot_folder)

        # [LEARNABLE PROGRESS] Check if model uses learnable progress head
        # Learnable models don't need action_means.json - they predict progress directly
        use_learnable_progress = hasattr(model, 'use_progress_head') and model.use_progress_head

        if use_learnable_progress:
            print(f"[INFO] Detected LEARNABLE progress model - using ProgressHead directly")
            action_means = None  # Not needed for learnable models
        else:
            # [CUMULATIVE L2] Load action means from EXPERIMENT FOLDER
            # Each model has its own action_means.json stored alongside its checkpoint
            action_means_path = os.path.join(testfolder, 'action_means.json')

            action_means = None
            if os.path.exists(action_means_path):
                with open(action_means_path, 'r') as f:
                    action_means = json.load(f)
                print(f"[INFO] Loaded action means from experiment folder: {len(action_means)} actions")
            else:
                print(f"[ERROR] Action means not found at {action_means_path}")
                print(f"[ERROR] Run: python generate_aligned_features.py --exp_folder {testfolder}")
                print(f"[ERROR] Then: python calculate_action_means.py --exp_folder {testfolder}")
                return None
        
        def parse_groundtruth_to_segments(gt_path):
            """Read ground truth .txt and parse into segments"""
            with open(gt_path, 'r') as f:
                action_names = [line.strip() for line in f if line.strip()]
            
            segments = []
            if not action_names:
                return segments, 0
            
            current_action = action_names[0]
            start_frame = 0
            
            for i in range(1, len(action_names)):
                if action_names[i] != current_action:
                    segments.append({
                        'name': current_action,
                        'start': start_frame,
                        'end': i - 1
                    })
                    current_action = action_names[i]
                    start_frame = i
            
            segments.append({
                'name': current_action,
                'start': start_frame,
                'end': len(action_names) - 1
            })
            
            return segments, len(action_names)
        
        for task in tasks:
            taskplot = plot_folder + f'/{task}'
            validate_folder(taskplot)
            dset_json_folder = get_env_variable('JSON_DPATH')
            data_structure = data_json_labels_handles(dset_json_folder, dset_name=config_obj['DATASET_NAME'])

            # [LEARNABLE] Skip train_cum calculation for learnable progress models
            if use_learnable_progress:
                train_cum_means = {task: 1.0}  # Placeholder, not used
                train_cum_vars = {task: 0.0}
            elif action_means is not None:
                # [4FPS FIX] Skip expensive get_average_train_cum_distance when action_means.json exists
                # action_means provides per-action normalization; train_cum_means is only a fallback
                # For 4fps models, this avoids loading 2-3GB files that cause OOM
                all_means = [v['mean'] for v in action_means.values() if v['mean'] > 0]
                fallback_mean = sum(all_means) / len(all_means) if all_means else 1.0
                train_cum_means = {task: fallback_mean}
                train_cum_vars = {task: 0.0}
                print(f"[INFO] Using action_means.json (4fps={is_4fps}) - skipped get_average_train_cum_distance")
            else:
                # Original behavior when no action_means.json
                if os.path.isdir(testfolder + '/ckpt'):
                    train_cum_means, train_cum_vars = get_average_train_cum_distance(
                        model, testfolder, data_structure, targ_task=task, skip_rate=config_obj.SKIP_RATE
                    )
                elif os.path.isdir(testfolder + f'/{task}/ckpt'):
                    train_cum_means, train_cum_vars = get_average_train_cum_distance(
                        model, testfolder + f'/{task}', data_structure, targ_task=task, skip_rate=config_obj.SKIP_RATE
                    )

                if None in [train_cum_means, train_cum_vars]:
                    return None
                if task not in train_cum_means.keys():
                    print(f'BTW {task} not in tasks')
                    return None
            
            gpe_list = []
            embedded_dl = flatten_dataloader_and_get_dict(model, test_dataloaders[task], config_obj.SKIP_RATE, device='cpu')

            if normalize:
                embedded_dl = svm_normalize_embedded_dl(embedded_dl=embedded_dl)
            
            for i, (outputs_dict, tdict) in enumerate(embedded_dl):
                outputs = outputs_dict['outputs']
                current_len = outputs.shape[0]
                
                # Extract video name
                video_name = outputs_dict.get('name', '')
                video_name = os.path.splitext(os.path.basename(video_name))[0]
                
                # Read ground truth from .txt file (action names, not IDs)
                # Search ALL 5 subsets to find the video's ground truth
                SUBSETS = ['egoprocel_subset1_S', 'egoprocel_subset2_OP_P', 'egoprocel_subset3_tent',
                           'egoprocel_subset4_numbers', 'egoprocel_subset5_head']

                gt_path = None
                for subset in SUBSETS:
                    potential_path = f'{PROTAS_BASE}/{subset}/groundTruth/{video_name}.txt'
                    if os.path.exists(potential_path):
                        gt_path = potential_path
                        break

                if gt_path is None:
                    print(f"[WARNING] No ground truth found for {video_name}, skipping")
                    continue
                
                # Parse ground truth into action segments
                segments, total_frames = parse_groundtruth_to_segments(gt_path)
                
                # Create tdict for get_trueprogress_per_action
                tdict = {
                    'step': [seg['name'] for seg in segments],
                    'start_frame': [seg['start'] for seg in segments],
                    'end_frame': [min(seg['end'], current_len-1) for seg in segments],
                    'name': video_name
                }
                
                # Get action-local ground truth progress (0->1 per action)
                true_progress = get_trueprogress_per_action(tdict)

                # Calculate predicted progress
                pred2_progress = torch.zeros(current_len)

                if use_learnable_progress:
                    # [LEARNABLE PROGRESS] Use ProgressHead for ONLINE per-frame prediction
                    # Similar to how cumulative_l2 computes cumulative distance at each frame
                    progress_head = outputs_dict.get('progress_head')
                    if progress_head is None:
                        print(f"[ERROR] Learnable model missing progress_head in output")
                        continue

                    for seg in segments:
                        action_name = seg['name']
                        start = max(0, min(seg['start'], current_len-1))
                        end = max(0, min(seg['end'], current_len-1))

                        if start >= current_len or action_name in ['SIL', 'background']:
                            continue

                        # ONLINE PER-FRAME PREDICTION:
                        # For each frame t, predict progress using only frames [start:t+1]
                        # This mirrors how cumulative_l2 computes cumulative distance at each frame
                        with torch.no_grad():
                            for t in range(start, end + 1):
                                # Extract partial segment from action start up to current frame
                                partial_segment = outputs[start:t+1].to(device)

                                # Predict progress at frame t using only past frames (within this action)
                                pred_progress_t = progress_head(partial_segment)
                                pred2_progress[t] = pred_progress_t.item()
                else:
                    # [CUMULATIVE L2] Use action means for normalization
                    for seg in segments:
                        action_name = seg['name']
                        start = max(0, min(seg['start'], current_len-1))
                        end = max(0, min(seg['end'], current_len-1))

                        if start >= current_len:
                            continue

                        # Extract segment outputs (cumulative distance resets automatically)
                        segment_outputs = outputs[start:end+1]
                        segment_cum = get_cum_matrix(segment_outputs)

                        if action_name not in ['SIL', 'background']:
                            # Use action-specific mean normalization
                            if action_name in action_means:
                                action_mean = action_means[action_name]['mean']
                                if action_mean > 0:
                                    pred2_progress[start:end+1] = segment_cum / action_mean
                                else:
                                    print(f"[WARNING] Zero mean for action '{action_name}', skipping")
                            else:
                                print(f"[WARNING] Action '{action_name}' not in action_means, using video-level mean")
                                # Fallback to video-level mean
                                if train_cum_means[task] > 0:
                                    pred2_progress[start:end+1] = segment_cum / train_cum_means[task]
                        # else: background stays at 0

                # Create mask to exclude background/SIL frames (only evaluate on action frames)
                action_mask = torch.zeros(len(true_progress), dtype=torch.bool)
                for seg in segments:
                    if seg['name'] not in ['SIL', 'background']:
                        seg_start = max(0, min(seg['start'], len(true_progress)-1))
                        seg_end = max(0, min(seg['end'], len(true_progress)-1))
                        action_mask[seg_start:seg_end+1] = True

                # Calculate action-level progress error - ONLY on action frames
                if action_mask.any():
                    errors = torch.abs(true_progress - pred2_progress)
                    gpe = errors[action_mask].mean()
                else:
                    gpe = torch.tensor(0.0)
                gpe_list.append(gpe.item())
                
                if plotting:
                    plt.clf()
                    a = true_progress.detach().cpu().numpy()
                    b = pred2_progress.detach().cpu().numpy()
                    plt.plot(a, color='green', label='GT Action-Local Progress')
                    plt.plot(b, color='blue', label='GTCC Predicted Progress')
                    plt.fill_between(range(len(a)), a, b, color='red', alpha=0.5, label='Error')
                    plt.xticks(np.arange(0, len(a), step=max(1, len(a) // 5)), fontsize=10)
                    plt.yticks(fontsize=10)
                    plt.xlabel('Frame', fontsize=15)
                    plt.ylabel('Progress', fontsize=15)
                    plt.title(f'Action-Level OGPE: {video_name}')
                    plt.legend()
                    plt.savefig(f'{taskplot}/{video_name}.pdf')
            
            if not gpe_list:
                print(f"[WARNING] No data processed for task {task}")
                continue

            result = {
                'task': task,
                'ogpe': np.mean(gpe_list),
                'num_videos': len(gpe_list)
            }

            # CoV only meaningful for cumulative L2 models
            if not use_learnable_progress and train_cum_means[task] > 0:
                result['CoV'] = train_cum_vars[task] / train_cum_means[task]
            else:
                result['CoV'] = 0.0  # N/A for learnable models

            return result

    # def __call__(self, model, config_obj, epoch, test_dataloaders, testfolder, tasks, normalize=False, plotting=False):
    #     WHITELIST_PATH = '/vision/anishn/GTCC_CVPR2024/evaluation_video_whitelist.json'
    #     plot_folder = testfolder + '/plotting_progress'
    #     validate_folder(plot_folder)
        
    #     # Load whitelist
    #     video_whitelist = None
    #     if os.path.exists(WHITELIST_PATH):
    #         with open(WHITELIST_PATH, 'r') as f:
    #             data = json.load(f)
    #         video_whitelist = set(data['video_names'])
    #         print(f"[INFO] Test set whitelist: {len(video_whitelist)} videos")

    #     PROTAS_BASE = '/vision/anishn/ProTAS/data_1fps/'
    #     SUBSETS = ['egoprocel_subset1_S', 'egoprocel_subset2_OP_P', 'egoprocel_subset3_tent',
    #             'egoprocel_subset4_numbers', 'egoprocel_subset5_head']

    #     def load_action_mapping(mapping_path):
    #         mapping = {}
    #         with open(mapping_path, 'r') as f:
    #             for line in f:
    #                 if line.strip():
    #                     parts = line.strip().replace(':', ' ').split()
    #                     if len(parts) >= 2:
    #                         mapping[parts[1]] = int(parts[0])
    #         return mapping

    #     # Load ALL subset mappings
    #     subset_mappings = {}
    #     for subset in SUBSETS:
    #         mapping_path = f'{PROTAS_BASE}/{subset}/mapping.txt'
    #         if os.path.exists(mapping_path):
    #             subset_mappings[subset] = load_action_mapping(mapping_path)

    #     def parse_groundtruth_to_segments(gt_path):
    #         """Read ground truth .txt and parse into segments"""
    #         with open(gt_path, 'r') as f:
    #             action_names = [line.strip() for line in f if line.strip()]
            
    #         # Group consecutive identical actions into segments
    #         segments = []
    #         if not action_names:
    #             return segments
            
    #         current_action = action_names[0]
    #         start_frame = 0
            
    #         for i in range(1, len(action_names)):
    #             if action_names[i] != current_action:
    #                 segments.append({
    #                     'name': current_action,
    #                     'start': start_frame,
    #                     'end': i - 1
    #                 })
    #                 current_action = action_names[i]
    #                 start_frame = i
            
    #         # Add final segment
    #         segments.append({
    #             'name': current_action,
    #             'start': start_frame,
    #             'end': len(action_names) - 1
    #         })
            
    #         return segments, len(action_names)

    #     for task in tasks:
    #         taskplot = plot_folder + f'/{task}'
    #         validate_folder(taskplot)
    #         dset_json_folder = get_env_variable('JSON_DPATH')
    #         data_structure = data_json_labels_handles(dset_json_folder, dset_name=config_obj['DATASET_NAME'])

    #         if hasattr(config_obj, 'USE_PROTAS') and config_obj.USE_PROTAS:
    #             train_cum_means = {task: 1.0}
    #             train_cum_vars = {task: 0.0}
    #         else:
    #             if os.path.isdir(testfolder + '/ckpt'):
    #                 train_cum_means, train_cum_vars = get_average_train_cum_distance(
    #                     model, testfolder, data_structure, targ_task=task, skip_rate=config_obj.SKIP_RATE
    #                 )
    #             elif os.path.isdir(testfolder + f'/{task}/ckpt'):
    #                 train_cum_means, train_cum_vars = get_average_train_cum_distance(
    #                     model, testfolder + f'/{task}', data_structure, targ_task=task, skip_rate=config_obj.SKIP_RATE
    #                 )

    #         if None in [train_cum_means, train_cum_vars]:
    #             return None
    #         if task not in train_cum_means.keys():
    #             print(f'BTW {task} not in tasks')
    #             return None

    #         gpe_list = []

    #         if hasattr(config_obj, 'USE_PROTAS') and config_obj.USE_PROTAS:
    #             embedded_dl = []
    #             model.eval()
    #             import numpy as np
    #             features_base_path = '/vision/anishn/ProTAS/data_1fps/egoprocel_subset1_S/features/'

    #             with torch.no_grad():
    #                 for batch_idx, data in enumerate(test_dataloaders[task]):
    #                     video_paths = data[0]
    #                     metadata_dicts = data[1]
                        
    #                     for v_idx, full_path in enumerate(video_paths):
    #                         filename = os.path.basename(full_path)
    #                         filename_no_ext = os.path.splitext(filename)[0]

    #                         # Filter to test set only
    #                         if video_whitelist is not None and filename_no_ext not in video_whitelist:
    #                             continue
                            
    #                         feat_path = os.path.join(features_base_path, filename_no_ext + '.npy')
    #                         if not os.path.exists(feat_path):
    #                             feat_path = os.path.join(features_base_path, filename_no_ext + '.pt')

    #                         try:
    #                             if feat_path.endswith('.npy'):
    #                                 features = np.load(feat_path)
    #                                 inputs = torch.from_numpy(features).float()
    #                             else:
    #                                 inputs = torch.load(feat_path).float()
                                
    #                             inputs = inputs.to(device)
    #                             if inputs.dim() == 2:
    #                                 inputs = inputs.unsqueeze(0)

    #                             out_dict = model(inputs)
    #                             out_dict['name'] = filename_no_ext
                                
    #                             # We'll create proper tdict later from ground truth
    #                             embedded_dl.append((out_dict, filename_no_ext))
                                
    #                         except Exception as e:
    #                             print(f"[ERROR] Failed to load {filename_no_ext}: {e}")
    #                             continue
    #         else:
    #             embedded_dl = flatten_dataloader_and_get_dict(
    #                 model, test_dataloaders[task], config_obj.SKIP_RATE, device='cpu'
    #             )

    #         if normalize:
    #             embedded_dl = svm_normalize_embedded_dl(embedded_dl=embedded_dl)

    #         # Process each video
    #         for i, item in enumerate(embedded_dl):
    #             if hasattr(config_obj, 'USE_PROTAS') and config_obj.USE_PROTAS:
    #                 outputs_dict, video_name = item
    #             else:
    #                 outputs_dict, tdict = item
    #                 video_name = outputs_dict.get('name', '')

    #             if hasattr(config_obj, 'USE_PROTAS') and config_obj.USE_PROTAS:
    #                 prog_matrix = outputs_dict['progress'].squeeze()  # [Classes, Time]
    #                 current_len = prog_matrix.shape[-1]
                    
    #                 # Find which subset this video belongs to
    #                 video_subset = None
    #                 gt_path = None
    #                 for subset in SUBSETS:
    #                     potential_path = f'{PROTAS_BASE}/{subset}/groundTruth/{video_name}.txt'
    #                     if os.path.exists(potential_path):
    #                         video_subset = subset
    #                         gt_path = potential_path
    #                         break
                    
    #                 if video_subset is None or gt_path is None:
    #                     print(f"[WARNING] No ground truth found for {video_name}, skipping")
    #                     continue
                    
    #                 # Read ground truth and parse into segments
    #                 segments, total_frames = parse_groundtruth_to_segments(gt_path)
                    
    #                 # Get the correct mapping for this subset
    #                 action_to_id = subset_mappings[video_subset]
                    
    #                 # Build ground truth active channel
    #                 T = current_len
    #                 gt_active_channel = torch.zeros(T, dtype=torch.long).to(device)
                    
    #                 # Create tdict for get_trueprogress_per_action
    #                 tdict = {
    #                     'step': [seg['name'] for seg in segments],
    #                     'start_frame': [seg['start'] for seg in segments],
    #                     'end_frame': [min(seg['end'], T-1) for seg in segments],
    #                     'name': video_name
    #                 }
                    
    #                 # Fill the channel mask with the GT step IDs
    #                 for seg in segments:
    #                     cls_name = seg['name']
    #                     cls_id = action_to_id.get(cls_name, 0)
                        
    #                     start = max(0, min(seg['start'], T-1))
    #                     end = max(0, min(seg['end'], T-1))
                        
    #                     if start < T and end < T:
    #                         gt_active_channel[start:end+1] = cls_id
                    
    #                 # Get true progress (action-local)
    #                 true_progress = get_trueprogress_per_action(tdict).to(device)
                    
    #                 # Gather predicted progress from the ground truth channels
    #                 pred2_progress = torch.gather(
    #                     prog_matrix, 0, gt_active_channel.unsqueeze(0)
    #                 ).squeeze(0)
                    
    #             else:
    #                 # GTCC path
    #                 outputs = outputs_dict['outputs']
    #                 current_len = outputs.shape[0]
                    
    #                 tdict['end_frame'][-1] = current_len - 1
    #                 true_progress = get_trueprogress_per_action(tdict).to(device)
                    
    #                 pred2_progress = get_cum_matrix(outputs)
    #                 if pred2_progress.sum() == 0:
    #                     continue
    #                 pred2_progress = pred2_progress / train_cum_means[task]
                
    #             # Calculate action-level progress error
    #             gpe = torch.mean(torch.abs(true_progress - pred2_progress))
    #             gpe_list.append(gpe.item())

    #             if plotting:
    #                 plt.clf()
    #                 a = true_progress.detach().cpu().numpy()
    #                 b = pred2_progress.detach().cpu().numpy()
    #                 plt.plot(a, color='green', label='GT Action-Local Progress')
    #                 plt.plot(b, color='blue', label='Predicted Progress')
    #                 plt.fill_between(range(len(a)), a, b, color='red', alpha=0.3, label='Error')
    #                 plt.xticks(np.arange(0, len(a), step=max(1, len(a) // 5)), fontsize=10)
    #                 plt.yticks(fontsize=10)
    #                 plt.xlabel('Frame', fontsize=15)
    #                 plt.ylabel('Progress', fontsize=15)
    #                 plt.title(f'Action-Level OGPE: {video_name}')
    #                 plt.legend()
    #                 plt.savefig(f'{taskplot}/{video_name}.pdf')

    #         if not gpe_list:
    #             print(f"[WARNING] No data processed for task {task}")
    #             continue

    #         return {
    #             'task': task,
    #             'ogpe': np.mean(gpe_list),
    #             'CoV': train_cum_vars[task] / train_cum_means[task],
    #             'num_videos': len(gpe_list)
    #         }
