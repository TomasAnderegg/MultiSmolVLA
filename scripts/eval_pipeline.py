#!/usr/bin/env python
"""
Evaluate the MultiSmolVLA pipeline on LIBERO environments.

Replaces the baseline SmolVLA policy with our full multimodal pipeline:
  RGB → Block1 (ThermalGen + ImageBind) → Block2 (4M + MLP + SmolVLA) → Actions

Depth and segmentation are obtained from the LIBERO/robosuite simulator by subclassing
LiberoEnv to override _ensure_env() (which hardcodes the OffScreenRenderEnv kwargs) and
_format_raw_obs() (which drops all non-RGB keys).  The subclass is injected into the
vec_env slot list after make_env() but before the first reset(), which is safe because
LiberoEnv creates its OffScreenRenderEnv lazily on first reset().

Usage:
  # Evaluate all tasks in a suite (10 episodes each):
  python scripts/eval_pipeline.py \\
      --checkpoint checkpoints/full_pipeline/pipeline_final.pt \\
      --task libero_object \\
      --n_episodes 10 \\
      --output_dir ./eval_logs/multismolvla

  # Specific task ids only:
  python scripts/eval_pipeline.py \\
      --checkpoint checkpoints/full_pipeline/pipeline_final.pt \\
      --task libero_10 \\
      --task_ids 0 1 2 \\
      --n_episodes 5

  # No checkpoint (base untrained pipeline, useful for sanity-check):
  python scripts/eval_pipeline.py --task libero_spatial --n_episodes 2
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# ── repo / lerobot paths ─────────────────────────────────────────────────────
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "third_party", "lerobot", "src"))

from lerobot.envs import make_env, preprocess_observation
from lerobot.envs.configs import LiberoEnv as LiberoEnvCfg  # the dataclass config

from src.pipeline.full_pipeline import VLAPipeline
from src.pipeline.modality_dropout import ModalityDropout, AVAILABLE_MODALITIES, AVAILABLE_CORRUPTIONS

# ---------------------------------------------------------------------------
# Robosuite 1.4.x segmentation overflow fix
# ---------------------------------------------------------------------------
# binding_utils.read_pixels() encodes the segmentation ID as:
#   seg = rgb[:,:,0] + rgb[:,:,1]*256 + rgb[:,:,2]*65536
# on a uint8 ndarray, which raises OverflowError.  The fix is to cast to
# int32 before the arithmetic.  We must apply this BEFORE any OffScreenRenderEnv
# is constructed because robosuite validates the segmentation sensor during
# __init__ by calling read_pixels(), and that is what triggers the crash.
def _apply_robosuite_seg_patch():
    """Fix robosuite 1.4.x OverflowError in binding_utils.read_pixels().

    The crash is:
        seg_img = rgb_img[:,:,0] + rgb_img[:,:,1]*(2**8) + rgb_img[:,:,2]*(2**16)
    where rgb_img is uint8, so *256 overflows.

    The call chain is:
        robot_env.py camera_segmentation sensor
          -> self.sim.render(..., segmentation=True)
             -> read_pixels(..., segmentation=True)   # <-- crashes here
          -> seg = result[::convention, :, 1]         # expects (H,W,3) uint8

    So read_pixels must return a plain (H,W,3) uint8 array — the packed render
    buffer — with seg IDs encoded across the three colour channels.  We cannot
    return a decoded seg ID array or a tuple; the caller does its own slicing.

    Fix: call _orig with segmentation=False to get the raw RGB buffer without
    triggering the overflow, decode the IDs with int32 arithmetic, then RE-PACK
    them back into a (H,W,3) uint8 array in the same channel layout so the
    caller's slicing (channel index 1) still extracts the correct values.
    """
    import robosuite.utils.binding_utils as _bu

    ctx_cls = getattr(_bu, "MjRenderContextOffscreen",
               getattr(_bu, "MjRenderContext", None))
    if ctx_cls is None:
        log.warning("Could not locate MjRenderContext class -- seg patch skipped.")
        return

    _orig = ctx_cls.read_pixels

    def _patched_read_pixels(self, width, height, *, depth=False, segmentation=False):
        if not segmentation:
            return _orig(self, width, height, depth=depth, segmentation=segmentation)

        # Get the raw packed-RGB render buffer without the crashing decode step.
        base    = _orig(self, width, height, depth=depth, segmentation=False)
        rgb_img = base[0] if depth else base   # (H, W, 3) uint8

        # Decode with int32 — no overflow.
        rgb32   = rgb_img.astype(np.int32)
        seg_ids = (rgb32[:, :, 0]
                   + rgb32[:, :, 1] * (2**8)
                   + rgb32[:, :, 2] * (2**16))   # (H, W) int32

        # Re-pack into (H, W, 3) uint8 so robot_env.py:
        #   seg = result[::convention, :, 1]
        # still reads the correct mid-byte channel.
        packed = np.zeros((*rgb_img.shape[:2], 3), dtype=np.uint8)
        packed[:, :, 0] = (seg_ids & 0x0000FF).astype(np.uint8)
        packed[:, :, 1] = ((seg_ids & 0x00FF00) >> 8).astype(np.uint8)
        packed[:, :, 2] = ((seg_ids & 0xFF0000) >> 16).astype(np.uint8)

        if depth:
            return packed, base[1]
        return packed

    ctx_cls.read_pixels = _patched_read_pixels
    log.info("Applied robosuite seg patch to %s.read_pixels (int32 decode + repack)",
             ctx_cls.__name__)


# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_apply_robosuite_seg_patch()

_IMAGE_SIZE   = 224    # ThermalGen and ImageBind both expect 224x224
_DEPTH_NEAR   = 0.01   # metres -- must match robosuite camera near-clip
_DEPTH_FAR    = 10.0   # metres -- must match robosuite camera far-clip
_CAMERA_NAME  = "agentview"


def depth_buf_to_meters(depth_buf: torch.Tensor) -> torch.Tensor:
    """Convert a raw MuJoCo non-linear depth buffer ([0,1]) to metric depth in metres."""
    return _DEPTH_NEAR * _DEPTH_FAR / (_DEPTH_FAR - (_DEPTH_FAR - _DEPTH_NEAR) * depth_buf)


# ---- LiberoEnv subclass that adds depth + segmentation ----------------------
#
# LiberoEnv (lerobot/envs/libero.py) has two problems that prevent depth/seg:
#
#   1. _ensure_env() hardcodes the OffScreenRenderEnv constructor kwargs and
#      never passes camera_depths or camera_segmentations.
#
#   2. _format_raw_obs() explicitly cherry-picks only RGB, eef, gripper, and
#      joint keys from the raw robosuite observation, so depth/seg are dropped
#      even if robosuite produces them.
#
# We fix both by subclassing and overriding just those two methods.
# The subclass instances are injected into the vec_env slot list after
# make_env() builds it, but before the first reset() triggers _ensure_env(),
# so the lazy init works in our favour.

class LiberoEnvWithDepthSeg:
    """Wraps a LiberoEnv to add depth and segmentation observations.

    Two problems in LiberoEnv prevent depth/seg from reaching the policy:

      1. _ensure_env() hardcodes the OffScreenRenderEnv constructor with no
         camera_depths / camera_segmentations flags.

      2. gymnasium's SyncVectorEnv stacks observations by iterating over
         observation_space keys only, so any extra keys added to the obs dict
         are silently dropped before obs_raw reaches our code.

    Solution: override reset() and step() to read depth/seg directly from the
    underlying robosuite env via _get_observations() after each call, and store
    them as instance attributes (_last_depth, _last_seg).  extract_depth_seg()
    reads from those attributes on each slot, bypassing the vec_env stacking.
    """

    def __init__(self, libero_env):
        from libero.libero.envs import OffScreenRenderEnv as _OSRE
        self._wrapped = libero_env
        self._OSRE = _OSRE
        self._last_depth = None   # (H,W) float32 after each reset/step
        self._last_seg   = None   # (H,W,1) int32 after each reset/step
        # Patch _ensure_env on the instance so depth/seg flags are passed.
        self._wrapped._ensure_env = self._patched_ensure_env

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def _patched_ensure_env(self):
        """Like LiberoEnv._ensure_env but enables depth and segmentation."""
        if self._wrapped._env is not None:
            return
        env = self._OSRE(
            bddl_file_name=self._wrapped._task_bddl_file,
            camera_heights=self._wrapped.observation_height,
            camera_widths=self._wrapped.observation_width,
            camera_depths=True,
            camera_segmentations="instance",
        )
        env.reset()
        self._wrapped._env = env

    def _cache_depth_seg(self):
        """Read depth and seg directly from the live robosuite env and cache."""
        rs_env = self._wrapped._env
        if rs_env is None:
            return
        # _get_observations() returns the full robosuite obs dict including all
        # camera modalities enabled at construction time (depth, seg, rgb).
        raw = rs_env.env._get_observations()
        self._last_depth = raw.get(f"{_CAMERA_NAME}_depth")
        self._last_seg   = raw.get(f"{_CAMERA_NAME}_segmentation_instance")

    def reset(self, **kwargs):
        result = self._wrapped.reset(**kwargs)
        self._cache_depth_seg()
        return result

    def step(self, action):
        result = self._wrapped.step(action)
        self._cache_depth_seg()
        return result

    def close(self):
        return self._wrapped.close()

    def render(self):
        return self._wrapped.render()


def inject_depth_seg_envs(vec_env) -> None:
    """Swap every LiberoEnv slot in vec_env for a LiberoEnvWithDepthSeg wrapper.

    Must be called AFTER make_env() and BEFORE the first reset().
    LiberoEnv._ensure_env() is lazy (runs on first reset()), so the swap is
    always safe as long as no reset has happened yet.
    """
    slots = getattr(vec_env, "envs", None)
    if slots is None:
        log.warning("inject_depth_seg_envs: could not find .envs on vec_env -- "
                    "depth/seg will be zeros.")
        return
    for i, env in enumerate(slots):
        if isinstance(env, LiberoEnvWithDepthSeg):
            log.debug("Slot %d: already wrapped, skipping.", i)
            continue
        slots[i] = LiberoEnvWithDepthSeg(env)
        log.info("Slot %d: injected LiberoEnvWithDepthSeg", i)


def extract_depth_seg(vec_env, device: str):
    """Extract depth and seg from LiberoEnvWithDepthSeg slot caches.

    After each reset()/step(), every LiberoEnvWithDepthSeg slot calls
    _cache_depth_seg() which reads from the live robosuite env directly,
    bypassing gymnasium's SyncVectorEnv observation stacking (which only
    copies keys defined in observation_space and drops everything else).

    We iterate over vec_env.envs, read _last_depth and _last_seg from each
    LiberoEnvWithDepthSeg slot, and stack them into (B,1,H,W) tensors.

    Returns:
        depth : (B,1,H,W) float32 tensor, metric metres
        seg   : (B,1,H,W) float32 tensor, normalised to [0,1]
    """
    slots = getattr(vec_env, "envs", [])
    depth_frames, seg_frames = [], []

    for i, slot in enumerate(slots):
        if not isinstance(slot, LiberoEnvWithDepthSeg):
            log.warning("Slot %d is not LiberoEnvWithDepthSeg (%s) -- zeros.", i, type(slot))
            depth_frames.append(None)
            seg_frames.append(None)
            continue
        depth_frames.append(slot._last_depth)
        seg_frames.append(slot._last_seg)

    n = len(slots) or 1

    def _to_tensor(arr):
        if arr is None:
            return None
        if isinstance(arr, torch.Tensor):
            return arr.float()
        return torch.from_numpy(np.asarray(arr, dtype=np.float32))

    # ---- depth ---------------------------------------------------------------
    if any(d is not None for d in depth_frames):
        H, W = next(d for d in depth_frames if d is not None).shape[:2]
        stacked = torch.zeros(n, 1, H, W)
        for i, d in enumerate(depth_frames):
            if d is not None:
                t = _to_tensor(d)
                if t.dim() == 3:        # (H,W,1) -> (H,W)
                    t = t.squeeze(-1)
                stacked[i, 0] = t
        depth = depth_buf_to_meters(stacked.to(device))
        if depth.shape[-2:] != (_IMAGE_SIZE, _IMAGE_SIZE):
            depth = F.interpolate(depth, (_IMAGE_SIZE, _IMAGE_SIZE), mode="nearest")
    else:
        log.warning("extract_depth_seg: no depth data in any slot -- zeros.")
        depth = torch.zeros(n, 1, _IMAGE_SIZE, _IMAGE_SIZE, device=device)

    # ---- segmentation --------------------------------------------------------
    if any(s is not None for s in seg_frames):
        first = next(s for s in seg_frames if s is not None)
        H, W  = first.shape[:2]
        stacked = torch.zeros(n, 1, H, W)
        for i, s in enumerate(seg_frames):
            if s is not None:
                t = _to_tensor(s)
                if t.dim() == 3:        # (H,W,1) -> (H,W)
                    t = t.squeeze(-1)
                stacked[i, 0] = t
        seg = stacked.to(device)
        seg_max = seg.amax(dim=(-3,-2,-1), keepdim=True).clamp(min=1.)
        seg = seg / seg_max
        if seg.shape[-2:] != (_IMAGE_SIZE, _IMAGE_SIZE):
            seg = F.interpolate(seg, (_IMAGE_SIZE, _IMAGE_SIZE), mode="nearest")
    else:
        log.warning("extract_depth_seg: no seg data in any slot -- zeros.")
        seg = torch.zeros(n, 1, _IMAGE_SIZE, _IMAGE_SIZE, device=device)

    return depth, seg


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate MultiSmolVLA on LIBERO")

    # Checkpoint / model
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to pipeline .pt checkpoint from train_full_pipeline.py. "
                        "Omit to use the base (untrained) weights.")
    p.add_argument("--smolvla_checkpoint", type=str, default="lerobot/smolvla_libero",
                   help="HF hub or local path for the SmolVLA base checkpoint.")
    p.add_argument("--fourm_checkpoint", type=str, default="EPFL-VILAB/4M-21_XL",
                   help="HF hub or local path for the 4M-21 encoder checkpoint.")
    p.add_argument("--fourm_dim", type=int, default=1024,
                   help="Hidden dim of the 4M model (768 for B, 1024 for L/XL).")

    # Environment
    p.add_argument("--task", type=str, default="libero_object",
                   help="LIBERO suite: libero_spatial | libero_object | libero_goal | "
                        "libero_10 | libero_90")
    p.add_argument("--task_ids", type=int, nargs="*", default=None,
                   help="Subset of task ids to evaluate (0-indexed). Default: all tasks in suite.")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Number of parallel envs per task (same as --eval.batch_size in lerobot-eval).")

    # Eval
    p.add_argument("--n_episodes", type=int, default=10,
                   help="Number of episodes per task to evaluate.")
    p.add_argument("--max_lang_tokens", type=int, default=48)
    p.add_argument("--output_dir", type=str, default="./eval_logs/multismolvla")

    # Device
    p.add_argument("--device", type=str, default=None,
                   help="cuda or cpu (auto-detected if omitted).")

    # Eval-time modality corruption (robustness testing)
    p.add_argument("--corrupt_modalities", nargs="+", default=[], choices=AVAILABLE_MODALITIES,
                   help="Modalities to corrupt at eval time. E.g. --corrupt_modalities rgb depth. "
                        "By default no corruption is applied (clean eval).")
    p.add_argument("--corrupt_alpha", type=float, default=0.0,
                   help="Corruption intensity: 0.0=zeroed out (hard), 1.0=clean, 0.5=moderate soft corruption. "
                        "Only used when --corrupt_modalities is set.")
    p.add_argument("--corrupt_type", type=str, default=None, choices=AVAILABLE_CORRUPTIONS,
                   help="Corruption type to apply: gaussian | blur | occlusion. "
                        "Omit to pick randomly at each step.")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Eval-time corruption helper
# ─────────────────────────────────────────────────────────────────────────────

def build_eval_corruptor(args) -> ModalityDropout | None:
    """Return a ModalityDropout configured for deterministic eval-time corruption, or None."""
    if not args.corrupt_modalities:
        return None
    corruptor = ModalityDropout(
        modalities=args.corrupt_modalities,
        corruption_types=[args.corrupt_type] if args.corrupt_type else AVAILABLE_CORRUPTIONS,
    )
    corruptor.eval()
    return corruptor


def apply_eval_corruption(inputs: dict, corruptor: ModalityDropout, alpha: float) -> dict:
    """Corrupt the requested modalities in `inputs` at a fixed alpha intensity."""
    corrupted = dict(inputs)
    for modality in corruptor.modalities:
        if modality in corrupted:
            corrupted[modality] = corruptor.corrupt(corrupted[modality], alpha)
    return corrupted


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(args, device: str) -> VLAPipeline:
    log.info("Building VLAPipeline ...")
    pipeline = VLAPipeline(
        smolvla_checkpoint=args.smolvla_checkpoint,
        fourm_checkpoint=args.fourm_checkpoint,
        fourm_dim=args.fourm_dim,
        use_4m=True,
        freeze_4m=True,
        freeze_mlp=False,
        device=device,
    )

    if args.checkpoint is not None:
        log.info(f"Loading checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        pipeline.load_state_dict(state)
        log.info("Checkpoint loaded ✅")
    else:
        log.info("No checkpoint — using base (untrained) pipeline weights")

    pipeline.to(device)
    pipeline.eval()
    return pipeline


def reset_pipeline(pipeline: VLAPipeline):
    """Clear the SmolVLA action-chunk queue between episodes."""
    pipeline.block2.smolvla.policy.reset()


# ─────────────────────────────────────────────────────────────────────────────
# Observation → pipeline inputs
# ─────────────────────────────────────────────────────────────────────────────

def obs_to_pipeline_inputs(
    obs: dict,
    vec_env,
    task_descriptions: list[str],
    tokenizer,
    device: str,
    max_lang_tokens: int,
) -> tuple[dict, dict]:
    """
    Convert a lerobot-format observation (after preprocess_observation + LiberoProcessorStep)
    into the (inputs, batch) pair expected by VLAPipeline.forward().

    obs     : preprocessed lerobot observation -- used for RGB and robot state.
    vec_env : the vector env whose LiberoEnvWithDepthSeg slots hold the most
              recent depth/seg arrays (cached directly from robosuite after each
              reset/step, bypassing gymnasium's observation stacking).

    Expected obs keys:
      "observation.images.image"  : (B, 3, H, W) float32 [0, 1]
      "observation.state"         : (B, 8) float32
    """
    rgb = obs["observation.images.image"]  # (B, 3, H, W) [0, 1]

    # Resize to 224x224 (ThermalGen + ImageBind requirement)
    if rgb.shape[-2] != _IMAGE_SIZE or rgb.shape[-1] != _IMAGE_SIZE:
        rgb = F.interpolate(rgb, size=(_IMAGE_SIZE, _IMAGE_SIZE), mode="bilinear", align_corners=False)

    B = rgb.shape[0]
    rgb = rgb.to(device)

    # Depth and seg come directly from the raw robosuite observation.
    # lerobot's preprocess_observation / LiberoProcessorStep only handle RGB and
    # robot_state and will silently discard any other keys, so we bypass it here.
    depth, seg = extract_depth_seg(vec_env, device)

    inputs = {
        "rgb":   rgb,
        "depth": depth,
        "seg":   seg,
    }

    # Robot state: LiberoProcessorStep produces "observation.state" (B, 8)
    state = obs.get("observation.state", torch.zeros(B, 8)).to(device)

    # Tokenise task descriptions
    tok = tokenizer(
        task_descriptions,
        padding="max_length",
        truncation=True,
        max_length=max_lang_tokens,
        return_tensors="pt",
    )

    batch = {
        "observation.state":                   state,
        "observation.language.tokens":         tok["input_ids"].to(device),
        "observation.language.attention_mask": tok["attention_mask"].to(device),
    }

    return inputs, batch


# ─────────────────────────────────────────────────────────────────────────────
# Episode / eval loop
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(
    vec_env,
    pipeline: VLAPipeline,
    env_preprocessor,
    tokenizer,
    device: str,
    args,
    corruptor: ModalityDropout | None = None,
) -> list[bool]:
    """Run one batched episode; return per-env success flags."""
    # Inject LiberoEnvWithDepthSeg wrappers before the first reset.
    # inject_depth_seg_envs checks whether the slot is already wrapped, so
    # calling this at the start of every episode is safe (idempotent).
    inject_depth_seg_envs(vec_env)

    reset_pipeline(pipeline)
    obs_raw, _ = vec_env.reset()

    obs = preprocess_observation(obs_raw)
    obs = env_preprocessor(obs)

    try:
        task_descs = list(vec_env.get_attr("task_description"))
    except (AttributeError, NotImplementedError):
        task_descs = ["perform the manipulation task"] * vec_env.num_envs

    max_steps = vec_env.get_attr("_max_episode_steps")[0]
    done    = np.zeros(vec_env.num_envs, dtype=bool)
    success = np.zeros(vec_env.num_envs, dtype=bool)

    for _ in range(max_steps):
        if done.all():
            break

        # obs_raw is the unprocessed dict/list straight from the vec_env -- it
        # still contains depth and seg keys that preprocess_observation drops.
        inputs, batch = obs_to_pipeline_inputs(obs, vec_env, task_descs, tokenizer, device, args.max_lang_tokens)

        if corruptor is not None:
            inputs = apply_eval_corruption(inputs, corruptor, args.corrupt_alpha)

        with torch.inference_mode():
            action = pipeline.forward(inputs, batch, epoch=0)  # (B, action_dim)

        obs_raw, _reward, terminated, truncated, info = vec_env.step(action.cpu().numpy())

        obs = preprocess_observation(obs_raw)
        obs = env_preprocessor(obs)

        if "final_info" in info:
            ep_success = np.asarray(info["final_info"].get("is_success", np.zeros(vec_env.num_envs)), dtype=bool)
        elif "is_success" in info:
            ep_success = np.asarray(info["is_success"], dtype=bool)
        else:
            ep_success = np.zeros(vec_env.num_envs, dtype=bool)

        success |= ep_success
        done    |= np.asarray(terminated | truncated, dtype=bool)

    return success.tolist()


def eval_task(task_id, vec_env, pipeline, env_preprocessor, tokenizer, device, args,
              corruptor: ModalityDropout | None = None) -> dict:
    """Evaluate one task for n_episodes; return success list and rate."""
    successes: list[bool] = []
    while len(successes) < args.n_episodes:
        ep = run_episode(vec_env, pipeline, env_preprocessor, tokenizer, device, args, corruptor)
        successes.extend(ep)
    successes = successes[: args.n_episodes]
    sr = float(np.mean(successes)) * 100
    log.info(f"  task_id={task_id} → {sr:.1f}%  ({args.n_episodes} episodes)")
    return {"success_rate": sr, "successes": successes}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = args.device
    log.info(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Build pipeline
    pipeline = build_pipeline(args, device)

    # Eval-time corruptor (optional)
    corruptor = build_eval_corruptor(args)
    if corruptor is not None:
        log.info(f"Eval corruption: modalities={args.corrupt_modalities}  "
                 f"alpha={args.corrupt_alpha}  type={args.corrupt_type or 'random'}")

    # Tokeniser (same as training)
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

    # Build LIBERO envs  →  {suite_name: {task_id: vec_env}}
    env_cfg = LiberoEnvCfg(
        task=args.task,
        task_ids=args.task_ids,
        obs_type="pixels_agent_pos",  # gives robot_state so LiberoProcessorStep can build observation.state
    )
    log.info(f"Creating LIBERO envs  task={args.task}  task_ids={args.task_ids or 'all'}  "
             f"batch_size={args.batch_size}")
    envs = make_env(env_cfg, n_envs=args.batch_size)

    # LiberoProcessorStep: flips images 180° and converts robot_state → observation.state (8-dim)
    env_preprocessor, _ = env_cfg.get_env_processors()

    all_results: dict = {}
    all_successes: list[bool] = []

    for suite_name, task_envs in envs.items():
        log.info(f"\n── Suite: {suite_name}  ({len(task_envs)} tasks) ──────────────────────────")
        suite_results: dict = {}
        for task_id, vec_env in task_envs.items():
            result = eval_task(task_id, vec_env, pipeline, env_preprocessor, tokenizer, device, args, corruptor)
            suite_results[task_id] = result
            all_successes.extend(result["successes"])
            vec_env.close()
        all_results[suite_name] = suite_results

    overall_sr = float(np.mean(all_successes)) * 100 if all_successes else 0.0
    log.info(f"\nOverall success rate: {overall_sr:.1f}%  ({len(all_successes)} total episodes)")

    output = {
        "overall_success_rate": overall_sr,
        "per_suite": all_results,
        "corruption": {
            "modalities": args.corrupt_modalities,
            "alpha": args.corrupt_alpha,
            "type": args.corrupt_type,
        },
    }
    out_path = os.path.join(args.output_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
