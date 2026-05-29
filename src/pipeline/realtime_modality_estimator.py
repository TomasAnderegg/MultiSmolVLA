"""
Real-time depth and segmentation estimator for the 4M eval pipeline.

Depth  : Depth Anything V2 Small (24M params, ~30fps on GPU)
Seg    : k-means clustering on (R,G,B,D) features → pseudo-instance segmentation
         Produces a [0,1] grayscale map compatible with the 4M semseg tokenizer
         (which was trained on parquet image_mask — also grayscale [0,1]).
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


class RealtimeModalityEstimator:
    """
    Estimates depth and pseudo-segmentation from RGB in real-time.

    Usage:
        estimator = RealtimeModalityEstimator(device="cuda")
        result = estimator.estimate(rgb)   # rgb: (B,3,H,W) float [0,1]
        depth = result["depth"]            # (B,1,H,W) float [0,1]
        seg   = result["seg"]              # (B,1,H,W) float [0,1]
    """

    def __init__(
        self,
        device: str = "cuda",
        depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf",
        n_seg_clusters: int = 16,
        use_depth: bool = True,
        use_seg: bool = True,
    ):
        self.device = device
        self.use_depth = use_depth
        self.use_seg = use_seg
        self.n_seg_clusters = n_seg_clusters

        self._depth_processor = None
        self._depth_model = None

        if use_depth or use_seg:
            self._load_depth_model(depth_model)

        print(f"[RealtimeModalityEstimator] Ready  "
              f"depth={use_depth}  seg={use_seg}  clusters={n_seg_clusters}")

    # ─────────────────────────────────────────────────────────────────────────
    # Model loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_depth_model(self, model_id: str):
        print(f"[RealtimeModalityEstimator] Loading depth model {model_id} ...")
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        try:
            self._depth_processor = AutoImageProcessor.from_pretrained(
                model_id, local_files_only=True
            )
            self._depth_model = AutoModelForDepthEstimation.from_pretrained(
                model_id, local_files_only=True
            )
        except Exception:
            self._depth_processor = AutoImageProcessor.from_pretrained(model_id)
            self._depth_model = AutoModelForDepthEstimation.from_pretrained(model_id)

        self._depth_model.to(self.device).eval()
        for p in self._depth_model.parameters():
            p.requires_grad = False
        print("[RealtimeModalityEstimator] Depth model loaded")

    # ─────────────────────────────────────────────────────────────────────────
    # Depth estimation
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _estimate_depth(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        rgb : (B, 3, H, W) float [0,1]  on any device
        returns (B, 1, H, W) float [0,1]  depth map  (larger = further)
        """
        B, C, H, W = rgb.shape
        rgb_cpu = (rgb.cpu() * 255).byte()

        pil_images = [
            Image.fromarray(rgb_cpu[b].permute(1, 2, 0).numpy())
            for b in range(B)
        ]
        inputs = self._depth_processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._depth_model(**inputs)
        # predicted_depth: (B, H', W')  — raw metric depth
        depth_raw = outputs.predicted_depth   # (B, H', W')

        # Resize to input resolution
        depth = F.interpolate(
            depth_raw.unsqueeze(1).float(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )  # (B, 1, H, W)

        # Normalize per-sample to [0, 1]
        for b in range(B):
            d = depth[b]
            dmin, dmax = d.min(), d.max()
            if dmax > dmin:
                depth[b] = (d - dmin) / (dmax - dmin)
            else:
                depth[b] = torch.zeros_like(d)

        return depth.to(rgb.device)

    # ─────────────────────────────────────────────────────────────────────────
    # Pseudo-segmentation via k-means on (R,G,B,D) features
    # ─────────────────────────────────────────────────────────────────────────

    def _estimate_seg(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        rgb   : (B, 3, H, W) float [0,1]
        depth : (B, 1, H, W) float [0,1]
        returns (B, 1, H, W) float [0,1]  pseudo-seg (cluster_id / n_clusters)
        """
        from sklearn.cluster import MiniBatchKMeans

        B, _, H, W = rgb.shape
        # Stack (R,G,B,D) → (B, 4, H, W) → (B, H*W, 4)
        rgbd = torch.cat([rgb, depth], dim=1)  # (B,4,H,W)
        rgbd_np = rgbd.cpu().numpy()           # (B,4,H,W)

        seg_maps = []
        for b in range(B):
            feats = rgbd_np[b].reshape(4, H * W).T  # (H*W, 4)
            km = MiniBatchKMeans(
                n_clusters=self.n_seg_clusters,
                random_state=0,
                n_init=3,
                max_iter=50,
            )
            labels = km.fit_predict(feats).reshape(H, W)  # (H, W)  int [0, K-1]
            seg_norm = labels.astype(np.float32) / (self.n_seg_clusters - 1)  # [0,1]
            seg_maps.append(torch.from_numpy(seg_norm))

        seg = torch.stack(seg_maps, dim=0).unsqueeze(1).to(rgb.device)  # (B,1,H,W)
        return seg

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def estimate(self, rgb: torch.Tensor) -> dict:
        """
        rgb : (B, 3, H, W) float [0,1]
        returns dict with keys "depth" and/or "seg", each (B,1,H,W) float [0,1]
        """
        result = {}

        depth = None
        if self.use_depth or self.use_seg:
            depth = self._estimate_depth(rgb)

        if self.use_depth:
            result["depth"] = depth

        if self.use_seg:
            result["seg"] = self._estimate_seg(rgb, depth)

        return result
