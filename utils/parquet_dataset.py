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
# Source: libero/libero/benchmark/libero_suite_task_map.py, "libero_10" entry.
# Language = filename after stripping scene prefix (e.g. KITCHEN_SCENE3_), underscores→spaces.
LIBERO10_TASKS = {
    0: "put both the alphabet soup and the tomato sauce in the basket",
    1: "put both the cream cheese box and the butter in the basket",
    2: "turn on the stove and put the moka pot on it",
    3: "put the black bowl in the bottom drawer of the cabinet and close it",
    4: "put the white mug on the left plate and put the yellow and white mug on the right plate",
    5: "pick up the book and place it in the back compartment of the caddy",
    6: "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    7: "put both the alphabet soup and the cream cheese box in the basket",
    8: "put both moka pots on the stove",
    9: "put the yellow and white mug in the microwave and close it",
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

    def __init__(self, data_dir: str, chunk_size: int = 50):
        self.chunk_size = chunk_size
        self.shards = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
        if not self.shards:
            raise ValueError(f"No parquet files found in {data_dir}")

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
