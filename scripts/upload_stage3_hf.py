#!/usr/bin/env python3
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo = "TomasAnderegg/multismolvla-stage3-joint"

api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
print(f"Repo ready: https://huggingface.co/{repo}")

for step in [50000, 95000]:
    src = f"/scratch/checkpoints/stage3_joint/pipeline_step{step}.pt"
    dst = f"pipeline_step{step}.pt"
    print(f"Uploading {src} ...")
    api.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=repo, repo_type="model")
    print(f"Done: {dst}")

print("All done!")
