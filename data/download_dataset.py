import os
import pandas as pd
import requests
from tqdm import tqdm

# Dataset API endpoint
DATASET_NAME = "binhng/libero_10_lerobot_mask_depth"
API_URL = f"https://huggingface.co/api/datasets/{DATASET_NAME}"

OUTPUT_DIR = os.environ.get("DATA_DIR", "/scratch/izar/garate/data/parquet")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fetch dataset file list
response = requests.get(API_URL)
response.raise_for_status()
data = response.json()

# Extract parquet shard URLs
urls = [
    f"https://huggingface.co/datasets/{DATASET_NAME}/resolve/main/{f['rfilename']}"
    for f in data.get("siblings", [])
    if f["rfilename"].endswith(".parquet")
]

print(f"Found {len(urls)} parquet shards")

# Download + save each shard
for i, url in enumerate(tqdm(urls, desc="Processing shards")):
    shard_name = url.split("/")[-1]
    output_path = os.path.join(OUTPUT_DIR, shard_name)

    # Skip if already exists (resume capability)
    if os.path.exists(output_path):
        continue

    try:
        df = pd.read_parquet(url)
        df.to_parquet(output_path, index=False)
    except Exception as e:
        print(f"Failed on shard {i}: {e}")

print("All shards downloaded and saved individually.")
