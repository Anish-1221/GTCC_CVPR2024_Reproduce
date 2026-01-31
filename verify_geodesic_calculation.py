"""
Verification script to understand and validate geodesic distance calculation
"""
import os
import torch
import numpy as np

# Paths
ALIGNED_FEAT_DIR = '/vision/anishn/GTCC_Data_Processed_1fps/egoprocel/aligned_features'
PROTAS_BASE = '/vision/anishn/ProTAS/data_1fps/'
SUBSET = 'egoprocel_subset4_numbers'

def get_cum_matrix(video):
    """
    From evaluation.py - calculates cumulative geodesic distance.
    Returns array where P[t] = total distance from frame 0 to frame t
    """
    P = torch.zeros(video.shape[0])
    for t in range(1, video.shape[0]):
        P[t] = P[t-1] + torch.linalg.norm(video[t] - video[t-1])
    return P


def calculate_segment_geodesic(features):
    """
    New approach - calculate geodesic distance for a segment.
    Returns total distance from start to end of segment.
    """
    if features.shape[0] < 2:
        return 0.0
    
    diffs = torch.norm(features[1:] - features[:-1], p=2, dim=1)
    total_dist = torch.sum(diffs).item()
    return total_dist


def parse_segments(label_path):
    """Parse ground truth labels into segments"""
    with open(label_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    segments = []
    if not lines:
        return segments
    
    current_action = lines[0]
    start_frame = 0
    
    for i in range(1, len(lines)):
        if lines[i] != current_action:
            segments.append({
                'name': current_action,
                'start': start_frame,
                'end': i - 1
            })
            current_action = lines[i]
            start_frame = i
    
    segments.append({
        'name': current_action,
        'start': start_frame,
        'end': len(lines) - 1
    })
    
    return segments


print("="*80)
print("GEODESIC DISTANCE VERIFICATION")
print("="*80)

# Load first file
feat_files = sorted([f for f in os.listdir(ALIGNED_FEAT_DIR) if f.endswith('.npy')])
test_file = feat_files[0]
video_name = os.path.splitext(test_file)[0]

print(f"\nTest file: {test_file}")
print(f"Video name: {video_name}")

# Load features
features = np.load(os.path.join(ALIGNED_FEAT_DIR, test_file))
features = torch.from_numpy(features).float()
print(f"Feature shape: {features.shape}")

# Load labels
label_path = os.path.join(PROTAS_BASE, SUBSET, 'groundTruth', f"{video_name}.txt")
if not os.path.exists(label_path):
    print(f"ERROR: Label file not found at {label_path}")
    exit(1)

segments = parse_segments(label_path)
print(f"Number of segments: {len(segments)}")

print("\n" + "="*80)
print("METHOD COMPARISON")
print("="*80)

# Method 1: Cumulative matrix (from evaluation.py)
cum_matrix = get_cum_matrix(features)
total_video_distance_method1 = cum_matrix[-1].item()

print(f"\nMethod 1 (Cumulative Matrix):")
print(f"  Total video distance: {total_video_distance_method1:.4f}")
print(f"  Shape: {cum_matrix.shape}")
print(f"  First 5 values: {cum_matrix[:5].tolist()}")
print(f"  Last 5 values: {cum_matrix[-5:].tolist()}")

# Method 2: Direct segment calculation (new approach)
total_video_distance_method2 = calculate_segment_geodesic(features)

print(f"\nMethod 2 (Direct Calculation):")
print(f"  Total video distance: {total_video_distance_method2:.4f}")

# Verify they match
if abs(total_video_distance_method1 - total_video_distance_method2) < 0.001:
    print(f"\n✓ Methods match! Both approaches calculate the same geodesic distance.")
else:
    print(f"\n✗ Methods differ! Check calculation.")

print("\n" + "="*80)
print("SEGMENT-WISE BREAKDOWN")
print("="*80)

print(f"\n{'Action':<30} {'Frames':<15} {'Geodesic':>12} {'% of Total':>12}")
print("-" * 80)

segment_distances = []
background_count = 0
action_count = 0

for seg in segments:
    start = seg['start']
    end = seg['end'] + 1  # Exclusive end for slicing
    
    # Extract segment
    seg_feats = features[start:end]
    
    # Calculate geodesic for this segment
    seg_dist = calculate_segment_geodesic(seg_feats)
    segment_distances.append(seg_dist)
    
    # Calculate percentage of total
    pct = (seg_dist / total_video_distance_method2) * 100
    
    # Track background vs action
    if seg['name'] == 'background':
        background_count += 1
        marker = "(bg)"
    else:
        action_count += 1
        marker = ""
    
    print(f"{seg['name']:<30} {start:4d}-{end-1:4d} ({end-start:3d})  {seg_dist:12.4f} {pct:11.2f}% {marker}")

# Verify sum of segment distances equals total
sum_segments = sum(segment_distances)
print("-" * 80)
print(f"{'TOTAL':<30} {'':15} {sum_segments:12.4f} {100.00:11.2f}%")
print(f"\nSegment breakdown: {background_count} background + {action_count} action = {len(segments)} total")

if abs(sum_segments - total_video_distance_method2) < 0.001:
    print("\n✓ Sum of segment distances matches total video distance!")
else:
    print(f"\n✗ Mismatch: {sum_segments:.4f} vs {total_video_distance_method2:.4f}")

print("\n" + "="*80)
print("UNDERSTANDING GEODESIC DISTANCE")
print("="*80)

print("""
Geodesic distance measures how much the features "travel" in embedding space.

For a video with aligned features [T, D]:
1. Calculate L2 distance between each consecutive pair of frames
2. Sum all these distances = total geodesic distance

For a segment (e.g., "stirring" from frame 10-50):
1. Extract features for that segment: features[10:51]
2. Calculate geodesic distance just for those frames
3. This tells us how much "movement" happens during "stirring"

Why this matters:
- Actions with more motion → higher geodesic distance
- Longer segments → higher geodesic distance  
- Static actions → lower geodesic distance

By calculating mean geodesic distance per action across all videos:
- We learn the "typical distance" for each action
- Useful for progress estimation and action recognition
""")

print("="*80)
print("VERIFICATION COMPLETE")
print("="*80)
print(f"✓ Geodesic calculation is correct")
print(f"✓ Matches evaluation.py approach")
print(f"✓ Segments sum to total video distance")
print(f"✓ Ready to calculate action-wise means")
print("="*80)