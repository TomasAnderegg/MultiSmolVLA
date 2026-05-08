import os
import io
import pandas as pd
import requests
from tqdm import tqdm

DATASET_NAME = "TomasAnderegg/libero_10_thermal"
API_URL = f"https://huggingface.co/api/datasets/{DATASET_NAME}"
OUTPUT_DIR = os.environ.get("DATA_DIR", "/scratch/izar/apapadat/data/parquet_thermal")
HF_TOKEN = os.environ.get("HF_TOKEN")

os.makedirs(OUTPUT_DIR, exist_ok=True)

headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
if not HF_TOKEN:
    print("Warning: HF_TOKEN not set, requests may be rate limited")

response = requests.get(API_URL, headers=headers)
response.raise_for_status()
data = response.json()

urls = [
    f"https://huggingface.co/datasets/{DATASET_NAME}/resolve/main/{f['rfilename']}"
    for f in data.get("siblings", [])
    if f["rfilename"].endswith(".parquet")
]

print(f"Found {len(urls)} parquet shards")

for i, url in enumerate(tqdm(urls, desc="Downloading shards")):
    shard_name = url.split("/")[-1]
    output_path = os.path.join(OUTPUT_DIR, shard_name)

    if os.path.exists(output_path):
        continue

    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        df = pd.read_parquet(io.BytesIO(resp.content))
        df.to_parquet(output_path, index=False)
    except Exception as e:
        print(f"Failed on shard {i}: {e}")

print(f"Done. Shards saved to {OUTPUT_DIR}")
