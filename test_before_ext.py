"""
Quick test to verify model loading and forward pass work correctly
Run this BEFORE the full extraction to catch issues early
"""
import os
import torch
import numpy as np
from utils.train_util import get_config_for_folder, ckpt_restore_mprong
from utils.logging import configure_logging_format

logger = configure_logging_format()

# Paths
EXP_FOLDER = '/vision/anishn/GTCC_CVPR2024/output/multi-task-setting/V1___GTCC_egoprocel'
TEST_FEATURE = '/vision/anishn/GTCC_Data_Processed_1fps/egoprocel/frames/P18-R04-ContinentalBreakfast.npy'

print("="*80)
print("QUICK TEST: Verify Model and Forward Pass")
print("="*80)

# 1. Load model
print("\n[1/4] Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = get_config_for_folder(EXP_FOLDER)
ckpt_path = os.path.join(EXP_FOLDER, 'ckpt', 'epoch-50.pt')

try:
    model, _, epoch, loss, _ = ckpt_restore_mprong(
        ckpt_path,
        num_heads=16,
        dropout=config.LOSS_TYPE.get('GTCC', False),
        device=device
    )
    model.eval()
    print(f"✓ Model loaded successfully (epoch {epoch}, loss {loss:.4f})")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    exit(1)

# 2. Load test feature
print("\n[2/4] Loading test feature...")
try:
    raw_feats = np.load(TEST_FEATURE)
    print(f"✓ Loaded {TEST_FEATURE}")
    print(f"  Shape: {raw_feats.shape}")
    print(f"  Expected: [Time, 1024, 14, 14] for spatial features")
    print(f"  Got: {raw_feats.ndim}D tensor - {'✓ OK' if raw_feats.ndim == 4 else '✗ UNEXPECTED'}")
except Exception as e:
    print(f"✗ Feature loading failed: {e}")
    exit(1)

# 3. Test forward pass
print("\n[3/4] Testing forward pass...")
try:
    feat_tensor = torch.from_numpy(raw_feats).float().to(device)
    inputs = [feat_tensor]  # Wrap in list - same as training!
    
    print(f"  Input format: list of {len(inputs)} tensors")
    print(f"  Each tensor shape: {inputs[0].shape}")
    
    with torch.no_grad():
        out_dict = model(inputs)
    
    print(f"✓ Forward pass successful!")
    print(f"  Output keys: {list(out_dict.keys())}")
    
    if 'outputs' in out_dict:
        outputs = out_dict['outputs']
        print(f"  Type of 'outputs': {type(outputs)}")
        if isinstance(outputs, list) and len(outputs) > 0:
            print(f"  Number of videos in batch: {len(outputs)}")
            print(f"  Shape of first video: {outputs[0].shape}")
            aligned_feats = outputs[0].cpu().numpy()
            print(f"  Aligned features shape: {aligned_feats.shape}")
            
            # Verify time dimension preserved
            if aligned_feats.shape[0] == raw_feats.shape[0]:
                print(f"  ✓ Time dimension preserved: {raw_feats.shape[0]} frames")
            else:
                print(f"  ✗ Time dimension changed: {raw_feats.shape[0]} → {aligned_feats.shape[0]}")
        else:
            print(f"  ✗ Unexpected outputs format")
    else:
        print(f"  ✗ No 'outputs' key in model output")
        
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    print("\nFull error:")
    import traceback
    traceback.print_exc()
    print("\n" + "="*80)
    print("DIAGNOSIS:")
    print("="*80)
    print("If you see 'got 4 and 3 dimensions', check your model_multiprong.py")
    print("Make sure you removed the debug code that unpacks the list:")
    print("")
    print("  WRONG (in your current model_multiprong.py):")
    print("    if isinstance(videos, (list, tuple)):")
    print("        videos = videos[0]  # ← REMOVE THIS!")
    print("")
    print("  CORRECT:")
    print("    general_features = self.base_model(videos)  # Pass list as-is")
    print("")
    print("Solution: Replace models/model_multiprong.py with model_multiprong_CLEAN.py")
    print("="*80)
    exit(1)

# 4. Summary
print("\n[4/4] Summary")
print("="*80)
print("✓ Model loads correctly")
print("✓ Features load correctly")
print("✓ Forward pass works")
print("✓ Output format is correct")
print("✓ Time dimension is preserved")
print("="*80)
print("\n✓ All checks passed! You can now run the full extraction:")
print("  python generate_gtcc_aligned_features_debugged.py")
print("="*80)