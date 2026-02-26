import traceback
import json
from easydict import EasyDict as edict
import time
import random

from models.json_dataset import get_train_dataloaders
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from sklearn import svm
from sklearn.metrics import accuracy_score
import scipy.stats as stats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_target_alignment_with_dict(l1, l2, times_dict1, times_dict2):
    Y = torch.zeros(l1, l2)
    for step1, start1, end1 in zip(times_dict1['step'], times_dict1['start_frame'], times_dict1['end_frame']):
        for step2, start2, end2 in zip(times_dict2['step'], times_dict2['start_frame'], times_dict2['end_frame']):
            # condition = (step1.split('-')[0] == step2.split('-')[0] and 'SIL' not in [step1, step2]) if type(step1) == str else (step1 == step2) #MNIST !!!!!!!!!!!!!!!!!!
            condition = 'SIL' not in [step1, step2] and (step1 == step2)
            if condition:
                d1 = end1 - start1
                d2 = end2 - start2
                # for i1, i2 in zip(torch.linspace(start1, end1, steps=max(d1, d2)), torch.linspace(start2, end2, steps=max(d1, d2))):
                for i1, i2 in zip(range(start1, end1), torch.linspace(start2, end2, steps=d1)):
                    i, j = (min(l1-1, i1), min(l2-1, round(i2.item())))
                    Y[i, j] = 1
                        
    return Y / Y.sum()


def compute_eae_between_dict_vids(v1, v2, t1, t2, d1=None, return_plot_info=False, device='cpu', wild=False):
    if d1 is None:
        d1 = np.array([False for i in range(v1.shape[0])])
    true_alignment = get_target_alignment_with_dict(v1.shape[0], v2.shape[0], t1, t2).to(device)
    indices = np.nonzero(true_alignment).clone().cpu().numpy().T
    indices = list(zip(indices[0], indices[1]))
    dict_indices = {}
    y_set = set()
    for x, y in indices:
        y_set.add(y)
        if x.item() in dict_indices.keys():
            dict_indices[x.item()].append(y.item())
        else:
            dict_indices[x.item()] = [y.item()]
    
    # get total alignable distance for EAE calculation
    y_set = sorted(list(y_set))
    if wild:
        dist_matrix = np.zeros((v2.shape[0]))
        dist_matrix[y_set] = 1
        get_dist = lambda idx_pair: dist_matrix[min(idx_pair[0], idx_pair[1]):max(idx_pair[0], idx_pair[1])].sum()
        vert_add_number = sum([1 for i in range(len(y_set)-1) if y_set[i+1] - y_set[i] == 1])
    else:
        get_dist = lambda idx_pair: np.abs(idx_pair[0] - idx_pair[1])
        vert_add_number = v2.shape[0]

    cdist = 1/torch.cdist(v1, v2)
    nearest_neighbor = cdist.clone().cpu().detach().numpy().argmax(axis=1)
    eae_area = 0
    total_area = 0
    chosen_true_to_plot = []
    for x in range(v1.shape[0]):
        if x in dict_indices.keys(): # this means there is a true alignment for this frame
            if d1[x] and d1[x] is not None:
                eae_area += len(y_set)
                total_area += len(y_set)
            else:
                neighbors = dict_indices[x]
                chosen_neighbor = neighbors[np.argmin([np.abs(nearest_neighbor[x] - neighbors[i]) for i in range(len(neighbors))])]
                eae_item = get_dist((nearest_neighbor[x], chosen_neighbor))
                eae_area += eae_item
                total_area += vert_add_number
                chosen_true_to_plot.append((x, chosen_neighbor))
    np_keys = np.array(list(dict_indices.keys()))
    true_indices_to_plot = indices
    nearest_neighbor_to_plot = list(zip(np_keys.tolist(), nearest_neighbor[np_keys].tolist()))
    
    if return_plot_info:
        return list(zip(*true_indices_to_plot)), list(zip(*nearest_neighbor_to_plot)), list(zip(*chosen_true_to_plot)), y_set
    elif total_area > 0:
        return eae_area / total_area
    else:
        return None


def contains_non_float_values(tensor):
    def check_tensor(data):
        # Check for NaN values
        nan_check = torch.isnan(data)
        
        # Check for positive infinity (inf) values
        pos_inf_check = torch.isinf(data)
        
        # Check for negative infinity (-inf) values
        neg_inf_check = torch.isinf(data) & (data < 0)
        
        # Combine the checks for NaN, inf, and -inf values
        has_non_float_values = torch.any(nan_check | pos_inf_check | neg_inf_check)
        
        return has_non_float_values.item()
    if torch.is_tensor(tensor):
        return check_tensor(tensor)
    elif isinstance(tensor, np.ndarray):
        return check_tensor(torch.from_numpy(tensor))
    elif isinstance(tensor, list) and len(tensor) > 0 and isinstance(tensor[0], np.ndarray): # list of numpies
        return any([check_tensor(torch.from_numpy(one_array)) for one_array in tensor])
    elif isinstance(tensor, list) and len(tensor) > 0 and torch.is_tensor(tensor): # list of numpies
        return any([check_tensor(one_array) for one_array in tensor])
    elif isinstance(tensor, list) and len(tensor) > 0 and type(tensor[0]) == int: # list of numpies
        tensor = np.array(tensor)
        return check_tensor(torch.from_numpy(tensor))
    else:
        print("Bad input, must be (tensor, np.ndarray) or a list of either")
        exit(1)


def get_gmm_lfbgf(
        probability,
        n_components=5,
        device='cpu',
        debug=False,
        max_iters=50,
        history_size=10,
        max_iter=4,
        loss_fn='KL'
    ):
    if probability.shape[0] < 2:
        print(f"[WARNING] Skipping GMM: sequence length {probability.shape[0]} is too short.")
        return None, None, None
    #################################
    ### Loss functions
    #################################
    def kl_divergence_loss(mus, sigmas, psis):
        all_gaussians = get_gaussians(mus=mus, vars=sigmas)
        gmm = get_spread(spread_vals=psis) @ all_gaussians
        gmm_log = torch.log(gmm + 1e-30)
        p_log = torch.log(probability + 1e-30)

        kl_div = torch.sum(torch.sum(probability * (p_log - gmm_log))) + \
            torch.sum(torch.sum(gmm * (gmm_log - p_log)))
        p_log_diff = torch.diff(p_log)
        intensity_div = torch.abs(
            (p_log_diff / p_log_diff.max() + p_log[:-1] / p_log.max()) * ((p_log - gmm_log)[:-1] + (p_log - gmm_log)[1:])
        ).sum()

        if kl_div < 0 or contains_non_float_values(kl_div):
            print(probability.sum())
            raise Exception
        return kl_div

    def get_spread(spread_vals):
        return softmax(spread_vals)
    
    def get_gaussians(mus, vars):
        stds = torch.sqrt(torch.clamp(vars.view(-1, 1), .5))
        mus = torch.clamp(mus.view(-1, 1), 0, N-1)
        all_gaussians = torch.exp(
            -0.5 * (torch.subtract(arange, mus)  / stds) ** 2
        ) / (
            stds * (2 * torch.pi) ** 0.5
        )
        all_gaussians = torch.divide(all_gaussians.T, all_gaussians.sum(dim=1)).T
        return all_gaussians
    
    def closure():
        lbfgs.zero_grad()
        if loss_fn == 'KL':
            objective = kl_divergence_loss(means, stds, spreads)
        else:
            print('ERROR')
            exit(1)
        if type(objective) == tuple:
            return objective
        objective.backward()
        return objective
    try:
        with torch.enable_grad():
            probability = probability.to(device).detach()
            dtype = torch.float32
            N = probability.shape[0]
            arange = torch.arange(N, dtype=dtype).to(device).detach()
            start = time.time()
            softmax = nn.Softmax(dim=0).to(device)

            means = torch.nn.Parameter(torch.tensor(
                    [1 + i * (N-2)/n_components for i in range(n_components)],
                    requires_grad=True,
                    dtype=dtype,
                    device=device
                ))
            stds = torch.nn.Parameter(torch.tensor(
                    [N for i in range(n_components)],
                    requires_grad=True,
                    dtype=dtype,
                    device=device
                ))
            spreads = torch.nn.Parameter(torch.tensor(
                    [1 for i in range(n_components)],
                    requires_grad=True,
                    dtype=dtype,
                    device=device
                ))
            lbfgs = optim.LBFGS(
                [
                    {'params': [means, stds, spreads]}
                ],
                history_size=history_size,
                max_iter=max_iter, 
                line_search_fn="strong_wolfe",
                lr=.5
            )
            
            s = time.time()
            for _ in range(max_iters):
                lbfgs.step(closure)
            if debug:
                speed = time.time() - s
                return get_gaussians(means, stds), get_spread(spreads), {
                    'KL': (kl_divergence_loss(means, stds, spreads).detach().item(), speed),
                    'mus': means,
                    'stds': stds,
                }
            else:
                return get_gaussians(means, stds), get_spread(spreads), means
    except Exception as e:
        traceback.print_exc()
        exit(1)
        return None, None, None
        


def preprocess_batch(inputs, times, device='cpu', skip_rate=None):
    
    if type(inputs[0]) == str:
        new_inputs = []
        for input, tdict in zip(inputs, times):
            np_array = np.load(input)
            T = np_array.shape[0]
            if skip_rate is not None:
                assert type(skip_rate) == int and skip_rate < T
                seq_to_use = np_array[::skip_rate]
                skipT = seq_to_use.shape[0]
                del np_array
                tdict['start_frame'] = list(map(lambda t: round((t/T) * skipT), tdict['start_frame']))
                tdict['end_frame'] = list(map(lambda t: round((t/T) * skipT), tdict['end_frame']))
            else:
                seq_to_use = np_array
                skipT = seq_to_use.shape[0]
            if tdict['end_frame'][-1] != skipT-1:
                tdict['end_frame'][-1] =  skipT-1
            new_inputs.append(torch.from_numpy(seq_to_use).to(torch.float32).to(device))
        return new_inputs, times
    return [input.to(device) for input in inputs], times


    
def get_average_train_cum_distance(model, testfolder, data_structure, targ_task=None, skip_rate=None):
    try:
        with open(testfolder + '/config.json', 'r') as json_file:
            config = edict(json.load(json_file))
    except Exception as e:
        exit(1)
    data_folder = config['DATAFOLDER']
    tasks = data_structure.keys()
    train_dls = get_train_dataloaders(tasks, data_structure, config, device=device)
    

    means = {}
    vars = {}
    with torch.no_grad():
        for task in tasks:
            if targ_task is not None and targ_task != task:
                continue
            for inputs, times in train_dls[task]:
                inputs, times = preprocess_batch(inputs, times, device=device, skip_rate=skip_rate)
                outputs = model(inputs)['outputs']
                cum_dists = []
                for output in outputs:
                    cum_total = get_cum_matrix(output).max()
                    if not contains_non_float_values(cum_total):
                        cum_dists.append(cum_total.item())
                if len(cum_dists) == 0:
                    print("MODEL DIVERGED")
                    return None, None
                means[task] = np.mean(cum_dists)
                vars[task] = np.var(cum_dists)
                break
    return means, vars


def get_trueprogress(time_dict):
    assert type(time_dict) == dict
    N = time_dict['end_frame'][-1] + 1
    progress = torch.zeros(N)
    prev_prg = 0
    prg = 1 / sum([1 if step != 'SIL' else 0 for step in time_dict['step']])
    for step, start, end in zip(time_dict['step'], time_dict['start_frame'], time_dict['end_frame']):
        if step != 'SIL':
            progress[start:end+1] = torch.linspace(prev_prg, prev_prg + prg, round(end - start + 1))
            prev_prg = prev_prg + prg
        else:
            progress[start:end+1] = progress[start-1]
    return progress

def get_trueprogress_per_action(time_dict):
    """Each action goes 0->1 independently"""
    # '0' is the action ID for 'background' in egoprocel.json
    BACKGROUND_LABELS = ['0', 'SIL', 'background']
    N = time_dict['end_frame'][-1] + 1
    progress = torch.zeros(N)

    for step, start, end in zip(time_dict['step'], time_dict['start_frame'], time_dict['end_frame']):
        if step not in BACKGROUND_LABELS:
            segment_length = end - start + 1
            progress[start:end+1] = torch.arange(1, segment_length + 1, dtype=torch.float) / segment_length
        else:
            progress[start:end+1] = 0

    return progress


def get_cum_matrix(video):
    P = torch.zeros(video.shape[0])
    for t in range(1, video.shape[0]):
        P[t] = P[t-1] + torch.linalg.norm(video[t] - video[t-1])
    return P


def get_normalized_predicted_progress_action(features, time_dict):
    """
    Compute predicted progress normalized to [0, 1] for ACTION-LEVEL.
    Cumulative L2 distance resets at each action boundary.
    SIL/background segments get progress = 0.
    """
    # '0' is the action ID for 'background' in egoprocel.json
    BACKGROUND_LABELS = ['0', 'SIL', 'background']
    T = features.shape[0]
    pred_progress = torch.zeros(T, device=features.device)

    for step, start, end in zip(time_dict['step'],
                                 time_dict['start_frame'],
                                 time_dict['end_frame']):
        start = max(0, min(start, T - 1))
        end = max(0, min(end, T - 1))

        if step in BACKGROUND_LABELS:
            pred_progress[start:end+1] = 0
        else:
            segment_features = features[start:end+1]
            if segment_features.shape[0] < 2:
                pred_progress[start:end+1] = 0.5
            else:
                segment_cum = get_cum_matrix(segment_features)
                max_val = segment_cum[-1]
                if max_val > 0:
                    pred_progress[start:end+1] = segment_cum / max_val
                else:
                    seg_len = end - start + 1
                    pred_progress[start:end+1] = torch.linspace(0, 1, seg_len, device=features.device)

    return pred_progress


def sample_action_segment_with_random_index(embeddings, times_dict, min_segment_len=3):
    """
    Sample a random segment within a random non-SIL action.
    Returns segment embeddings up to random target index and GT progress.
    """
    # '0' is the action ID for 'background' in egoprocel.json
    BACKGROUND_LABELS = ['0', 'SIL', 'background']
    valid_actions = []
    for idx, (step, start, end) in enumerate(zip(
        times_dict['step'], times_dict['start_frame'], times_dict['end_frame']
    )):
        action_length = end - start + 1
        if step not in BACKGROUND_LABELS and action_length >= min_segment_len:
            valid_actions.append((idx, step, start, end))

    if len(valid_actions) == 0:
        return None, None, None

    _, action_name, action_start, action_end = random.choice(valid_actions)
    action_length = action_end - action_start + 1

    max_start = action_end - min_segment_len + 1
    if max_start < action_start:
        max_start = action_start

    seg_start = random.randint(action_start, max_start)
    seg_end = random.randint(seg_start + min_segment_len - 1, action_end)
    target_idx = random.randint(seg_start, seg_end)

    T = embeddings.shape[0]
    seg_start = max(0, min(seg_start, T - 1))
    target_idx = max(seg_start, min(target_idx, T - 1))

    segment_embeddings = embeddings[seg_start:target_idx + 1]
    progress_at_target = (target_idx - action_start + 1) / action_length
    gt_progress = min(1.0, max(0.0, progress_at_target))

    return segment_embeddings, gt_progress, action_name


def sample_action_segment_with_multiple_frames(embeddings, times_dict, min_segment_len=3, frames_per_segment=3):
    """
    Sample a random non-SIL action, then sample multiple target frames within that action.
    For each target frame, return embeddings from action_start to target_frame.

    Returns list of (segment_embeddings, gt_progress) tuples for each target frame.
    - segment_embeddings: embeddings[action_start : target_frame+1]
    - gt_progress: (target_frame - action_start + 1) / action_length
    """
    BACKGROUND_LABELS = ['0', 'SIL', 'background']
    valid_actions = []
    for idx, (step, start, end) in enumerate(zip(
        times_dict['step'], times_dict['start_frame'], times_dict['end_frame']
    )):
        action_length = end - start + 1
        if step not in BACKGROUND_LABELS and action_length >= min_segment_len:
            valid_actions.append((idx, step, start, end))

    if len(valid_actions) == 0:
        return []

    _, action_name, action_start, action_end = random.choice(valid_actions)
    action_length = action_end - action_start + 1

    T = embeddings.shape[0]
    action_start = max(0, min(action_start, T - 1))
    action_end = max(action_start, min(action_end, T - 1))
    action_length = action_end - action_start + 1

    # Sample multiple target frames within the action [action_start, action_end]
    # Ensure target is at least min_segment_len frames from action_start
    min_target = action_start + min_segment_len - 1
    if min_target > action_end:
        min_target = action_start + 1  # Fallback: at least 2 frames

    possible_targets = list(range(min_target, action_end + 1))
    if len(possible_targets) == 0:
        return []

    num_frames = min(frames_per_segment, len(possible_targets))
    target_indices = random.sample(possible_targets, num_frames)

    results = []
    for target_idx in target_indices:
        # Embeddings from action_start to target_frame (inclusive)
        segment_embeddings = embeddings[action_start:target_idx + 1]
        if segment_embeddings.shape[0] < 2:
            continue
        # GT progress: how far into the action is this target frame
        progress_at_target = (target_idx - action_start + 1) / action_length
        gt_progress = min(1.0, max(0.0, progress_at_target))
        results.append((segment_embeddings, gt_progress))

    return results


def flatten_dataloader_and_get_dict(model, dl, skip_rate=None, device='cpu'):
    l = []
    for i, (inputs, times) in enumerate(list(iter(dl))):
        data, times = preprocess_batch(inputs, times, skip_rate=skip_rate)
        with torch.no_grad():
            output_dict = model(data)
        del data

        # [LEARNABLE PROGRESS] Separate progress_head module from tensor outputs
        progress_head = output_dict.pop('progress_head', None)

        keys = output_dict.keys()
        for j in range(len(output_dict['outputs'])):
            out_dict = {key: output_dict[key][j].to(device) for key in keys}
            out_dict['name'] = inputs[j]
            # Re-attach progress_head module (not per-video, shared)
            if progress_head is not None:
                out_dict['progress_head'] = progress_head
            l.append((out_dict, times[j]))
    return l
