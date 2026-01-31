import numpy as np
import os

# --- CONFIGURATION ---
# Storing in a new folder to avoid overwriting your real data
DUMMY_DATA_ROOT = "/vision/anishn/GTCC_Dummy_Data"
DSET_PATH = os.path.join(DUMMY_DATA_ROOT, "egoprocel/frames")
os.makedirs(DSET_PATH, exist_ok=True)

# Correct shape for ResNet-50 Layer 3 spatiotemporal features
DUMMY_SHAPE = (100, 1024, 14, 14) 

# Handles identified from your Image 1, Image 2, and logs
# These MUST match the filenames in your Dataset (minus the extension)
dummy_handles = [
    "OP01-R03-BaconAndEggs",
    "OP02-R03-BaconAndEggs",
    "OP03-R03-BaconAndEggs",
    "OP04-R03-BaconAndEggs",
    "OP05-R03-BaconAndEggs",
    "OP06-R03-BaconAndEggs",
    "P02-R03-BaconAndEggs",
    "P16-R03-BaconAndEggs",
    "P17-R03-BaconAndEggs",
    "P18-R03-BaconAndEggs",
    "P19-R03-BaconAndEggs",
    "P20-R03-BaconAndEggs",
    "P21-R03-BaconAndEggs",
    "P22-R03-BaconAndEggs",
    "P23-R03-BaconAndEggs",
    "P24-R03-BaconAndEggs"
]

for handle in dummy_handles:
    print(f"Creating dummy 4D feature file for handle: {handle}...")
    # Creating a float32 tensor of shape (100, 1024, 14, 14)
    dummy_data = np.random.rand(*DUMMY_SHAPE).astype(np.float32)
    np.save(os.path.join(DSET_PATH, f"{handle}.npy"), dummy_data)

print(f"\nSuccess! Dummy files are ready at: {DSET_PATH}")
print("IMPORTANT: Run 'export DATASET_PATH=/vision/anishn/GTCC_Dummy_Data' before training.")