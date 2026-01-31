import os
import requests

out_dir = "/vision/anishn/Egocentric/Datasets/EGTEA-Gaze+"
os.makedirs(out_dir, exist_ok=True)

with open("/vision/anishn/GTCC_CVPR2024/egtea_gaze_links.txt") as f:
    urls = [l.strip() for l in f if l.strip()]

for url in urls:
    dl_url = url.replace("?dl=0", "?dl=1")
    fname = dl_url.split("/")[-1].split("?")[0]
    out_path = os.path.join(out_dir, fname)

    if os.path.exists(out_path):
        print(f"[SKIP] {fname}")
        continue

    print(f"[DOWNLOADING] {fname}")
    r = requests.get(dl_url, stream=True)
    r.raise_for_status()

    with open(out_path, "wb") as f_out:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f_out.write(chunk)
