import os
import numpy as np
from tqdm import tqdm

# --- PATH CONFIGURATION ---
GTCC_FEAT_DIR = '/vision/anishn/GTCC_Data_Processed_1fps/egoprocel/frames/'
PROTAS_BASE = '/vision/anishn/ProTAS/data_1fps/'
SUBSETS = [
    'egoprocel_subset1_S', 'egoprocel_subset2_OP_P', 'egoprocel_subset3_tent', 
    'egoprocel_subset4_numbers', 'egoprocel_subset5_head'
]

def check_frame_alignment():
    feat_files = sorted([f for f in os.listdir(GTCC_FEAT_DIR) if f.endswith('.npy')])
    print(f"[DEBUG] Found {len(feat_files)} feature files")
    print(f"[DEBUG] PROTAS_BASE: {PROTAS_BASE}")
    print(f"[DEBUG] Checking first file to diagnose issue...\n")

    # Debug first file in detail
    if feat_files:
        first_file = feat_files[0]
        video_name = os.path.splitext(first_file)[0]
        print(f"[DEBUG] First video: {video_name}")
        
        for s in SUBSETS:
            # Check what folders exist in this subset
            subset_path = os.path.join(PROTAS_BASE, s)
            print(f"\n[DEBUG] Checking subset: {s}")
            print(f"[DEBUG] Subset path: {subset_path}")
            print(f"[DEBUG] Subset exists: {os.path.exists(subset_path)}")
            
            if os.path.exists(subset_path):
                folders = [d for d in os.listdir(subset_path) if os.path.isdir(os.path.join(subset_path, d))]
                print(f"[DEBUG] Folders in subset: {folders}")
                
                # Try the old path (labels)
                old_path = os.path.join(PROTAS_BASE, s, 'labels', f"{video_name}.txt")
                print(f"[DEBUG] Looking for (old): {old_path}")
                print(f"[DEBUG] Exists: {os.path.exists(old_path)}")
                
                # Try the correct path (groundTruth)
                correct_path = os.path.join(PROTAS_BASE, s, 'groundTruth', f"{video_name}.txt")
                print(f"[DEBUG] Looking for (correct): {correct_path}")
                print(f"[DEBUG] Exists: {os.path.exists(correct_path)}")
                
                # List files in groundTruth if it exists
                gt_dir = os.path.join(PROTAS_BASE, s, 'groundTruth')
                if os.path.exists(gt_dir):
                    gt_files = [f for f in os.listdir(gt_dir) if f.endswith('.txt')]
                    print(f"[DEBUG] Found {len(gt_files)} .txt files in groundTruth")
                    if gt_files:
                        print(f"[DEBUG] First 3 files: {gt_files[:3]}")
    
    print("\n" + "="*80)
    print("Now running full alignment check with CORRECTED path...\n")
    
    stats = {"aligned": 0, "mismatch": 0, "no_label": 0}
    mismatch_details = []

    for f_name in tqdm(feat_files, desc="Verifying Frames"):
        video_name = os.path.splitext(f_name)[0]
        
        # 1. Search for labels in subsets (CORRECTED PATH)
        label_path = None
        for s in SUBSETS:
            p = os.path.join(PROTAS_BASE, s, 'groundTruth', f"{video_name}.txt")  # FIXED!
            if os.path.exists(p):
                label_path = p
                break
        
        if not label_path:
            stats["no_label"] += 1
            continue

        # 2. Compare frame counts
        feats = np.load(os.path.join(GTCC_FEAT_DIR, f_name))
        t_feat = feats.shape[0]

        with open(label_path, 'r') as f:
            t_gt = len([line.strip() for line in f if line.strip()])

        if t_feat == t_gt:
            stats["aligned"] += 1
        else:
            stats["mismatch"] += 1
            mismatch_details.append(f"{video_name}: Feat={t_feat}, GT={t_gt}")

    print("\n--- Alignment Report ---")
    print(f"Perfectly Aligned: {stats['aligned']}")
    print(f"Missing Labels:    {stats['no_label']}")
    print(f"Mismatched:        {stats['mismatch']}")
    
    if mismatch_details:
        print(f"\nTop {min(10, len(mismatch_details))} Mismatches:")
        for m in mismatch_details[:10]: 
            print(f"  {m}")

if __name__ == "__main__":
    check_frame_alignment()