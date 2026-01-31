import json

JSON_PATH = "/vision/anishn/GTCC_CVPR2024/dset_jsons/egoprocel.json"
OUTPUT_TXT = "/vision/anishn/GTCC_CVPR2024/egoprocel_task_handle_map.txt"

with open(JSON_PATH, "r") as f:
    data = json.load(f)

lines = []

for task_name, task_data in data.items():
    for handle in task_data["handles"]:
        lines.append((task_name, handle))

# Sort for stable, readable output
lines.sort()

with open(OUTPUT_TXT, "w") as f:
    for task_name, handle in lines:
        f.write(f"{task_name}\t{handle}\n")

print(f"Saved {len(lines)} task-handle pairs to {OUTPUT_TXT}")
