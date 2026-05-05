"""
Upload data/parquet_scratch/parquet_thermal/ to HuggingFace as a dataset.
Usage:
    python scripts/upload_thermal_hf.py --repo <username>/<repo-name> [--private]
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

TOKEN_FILE = "/scratch/izar/garate/huggingface_cache/token"
DATA_DIR = "/scratch/izar/garate/data/parquet_thermal"
FALLBACK_DATA_DIR = "data/parquet_scratch/parquet_thermal"


def get_token():
    # HF_TOKEN env var takes priority
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    # Explicit token file on this cluster
    token_path = Path(TOKEN_FILE)
    if token_path.exists():
        t = token_path.read_text().strip()
        if t:
            return t
    # Fall back to credentials saved by `hf auth login`
    return None


def get_data_dir():
    for path in [DATA_DIR, FALLBACK_DATA_DIR]:
        p = Path(path)
        if p.exists() and any(p.iterdir()):
            return p
    raise RuntimeError(f"Data directory not found at {DATA_DIR} or {FALLBACK_DATA_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="username/repo-name on HuggingFace")
    parser.add_argument("--private", action="store_true", help="Make the dataset private")
    parser.add_argument("--path-in-repo", default="data/train", help="Folder path inside the HF repo")
    args = parser.parse_args()

    token = get_token()
    data_dir = get_data_dir()

    api = HfApi(token=token)  # token=None → uses `hf auth login` cached credentials

    user = api.whoami()
    print(f"Logged in as: {user['name']}")

    print(f"Creating repo: {args.repo} (private={args.private})")
    api.create_repo(
        repo_id=args.repo,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    n_files = len(list(data_dir.glob("*.parquet")))
    print(f"Uploading {n_files} parquet files from {data_dir} → {args.repo}/{args.path_in_repo}")

    api.upload_folder(
        folder_path=str(data_dir),
        repo_id=args.repo,
        repo_type="dataset",
        path_in_repo=args.path_in_repo,
    )

    print(f"\nDone! Dataset at: https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
