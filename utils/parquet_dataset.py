import glob
import os
from io import BytesIO

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SIZE = 224

# LIBERO-10 task index → language instruction
# task_index maps to the 10 manipulation tasks in binhng/libero_10_lerobot_mask_depth
LIBERO10_TASKS = {
    0: "pick up the alphabet soup and place it in the basket",
    1: "pick up the cream cheese box and place it in the basket",
    2: "pick up the salad dressing and place it in the basket",
    3: "pick up the bbq sauce and place it in the basket",
    4: "pick up the ketchup and place it in the basket",
    5: "pick up the milk and place it in the basket",
    6: "pick up the tomato sauce and place it in the basket",
    7: "pick up the butter and place it in the basket",
    8: "pick up the chocolate pudding and place it in the basket",
    9: "pick up the cream cheese and place it in the basket",
}


def _decode_image(img_dict, mode: str = "RGB") -> torch.Tensor:
    """Decode a parquet image dict {"bytes": ..., "path": ...} → float tensor [0,1]."""
    img = Image.open(BytesIO(img_dict["bytes"])).convert(mode)
    img = img.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.BILINEAR if mode == "RGB" else Image.NEAREST,
    )
    arr = np.array(img, dtype=np.float32) / 255.0
    if mode == "RGB":
        return torch.from_numpy(arr).permute(2, 0, 1)   # (3, H, W)
    else:
        return torch.from_numpy(arr).unsqueeze(0)        # (1, H, W)


class ParquetThermalDataset(Dataset):
    """
    Dataset reading from pre-generated parquet shards that contain all 4 modalities:
    RGB, depth, segmentation, and thermal (pre-computed by thermal_pipeline.py).

    Each shard is one episode. For each timestep we return a chunk of `chunk_size`
    consecutive actions from that episode (padded at episode end) so SmolVLA can
    train on action sequences rather than single steps.

    Each shard is loaded on demand with a LRU-1 per-worker cache to limit I/O.
    """

    def __init__(self, data_dir: str, chunk_size: int = 50, task_ids: list[int] | None = None):
        self.chunk_size = chunk_size
        all_shards = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
        if not all_shards:
            raise ValueError(f"No parquet files found in {data_dir}")

        # Filter shards by task_id if requested
        if task_ids is not None:
            task_id_set = set(task_ids)
            filtered = []
            for shard in all_shards:
                tid = pd.read_parquet(shard, columns=["task_index"])["task_index"].iloc[0]
                if int(tid) in task_id_set:
                    filtered.append(shard)
            self.shards = filtered
            if not self.shards:
                raise ValueError(f"No shards found for task_ids={task_ids} in {data_dir}")
        else:
            self.shards = all_shards

        # Build (shard_idx, row_idx) index and record each shard's length
        self._index = []
        self._shard_lengths = []
        for i, shard in enumerate(self.shards):
            n = len(pd.read_parquet(shard, columns=["index"]))
            self._shard_lengths.append(n)
            for j in range(n):
                self._index.append((i, j))

        # LRU-1 cache — one shard loaded at a time per worker instance
        self._cached_shard_idx = None
        self._cached_df = None

    def __len__(self) -> int:
        return len(self._index)

    def _load_shard(self, shard_idx: int) -> pd.DataFrame:
        if self._cached_shard_idx != shard_idx:
            self._cached_df = pd.read_parquet(self.shards[shard_idx])
            self._cached_shard_idx = shard_idx
        return self._cached_df

    def __getitem__(self, idx: int) -> dict:
        shard_idx, row_idx = self._index[idx]
        df = self._load_shard(shard_idx)
        row = df.iloc[row_idx]

        rgb     = _decode_image(row["observation.images.image"],       mode="RGB")
        depth   = _decode_image(row["observation.images.image_depth"], mode="L")
        seg     = _decode_image(row["observation.images.image_mask"],  mode="L")
        thermal = _decode_image(row["observation.images.thermal"],     mode="L")
        thermal = thermal.repeat(3, 1, 1)  # (1,H,W) → (3,H,W) for ImageBind

        state = torch.tensor(np.asarray(row["observation.state"], dtype=np.float32))

        # Action chunk: fetch chunk_size consecutive actions from the same episode.
        # If we are near the end, clamp to the last row and mark those steps as pad.
        ep_len = self._shard_lengths[shard_idx]
        actions = []
        is_pad = []
        for k in range(self.chunk_size):
            r = min(row_idx + k, ep_len - 1)
            actions.append(torch.tensor(np.asarray(df.iloc[r]["action"], dtype=np.float32)))
            is_pad.append(row_idx + k >= ep_len)

        action        = torch.stack(actions)                      # (chunk_size, 7)
        action_is_pad = torch.tensor(is_pad, dtype=torch.bool)   # (chunk_size,)

        task_idx = int(row["task_index"])
        lang = LIBERO10_TASKS.get(task_idx, "perform the manipulation task")

        return {
            "rgb":                    rgb,           # (3, 224, 224) [0,1]
            "depth":                  depth,         # (1, 224, 224) [0,1]
            "seg":                    seg,           # (1, 224, 224) [0,1]
            "thermal":                thermal,       # (3, 224, 224) [0,1]
            "observation.state":      state,
            "action":                 action,        # (chunk_size, 7)
            "action_is_pad":          action_is_pad, # (chunk_size,)
            "language_instruction":   lang,
        }
