#!/usr/bin/env python
"""
Evaluate the MultiSmolVLA pipeline on LIBERO environments.

Replaces the baseline SmolVLA policy with our full multimodal pipeline:
  RGB → Block1 (ThermalGen + ImageBind) → Block2 (4M + MLP + SmolVLA) → Actions

Depth and segmentation are not available from LIBERO at eval time, so zero tensors
are passed for those modalities — ThermalGen only needs the RGB image to produce thermal.

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_IMAGE_SIZE = 224  # ThermalGen and ImageBind both expect 224×224


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
        state = torch.load(args.checkpoint, map_location=device, weights_only=False)
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
    task_descriptions: list[str],
    tokenizer,
    device: str,
    max_lang_tokens: int,
) -> tuple[dict, dict]:
    """
    Convert a lerobot-format observation (after preprocess_observation + LiberoProcessorStep)
    into the (inputs, batch) pair expected by VLAPipeline.forward().

    Expected obs keys:
      "observation.images.image"  : (B, 3, H, W) float32 [0, 1]  — agentview (flipped by LiberoProcessorStep)
      "observation.state"         : (B, 8) float32  — [eef_pos(3), eef_axisangle(3), gripper_qpos(2)]
    """
    rgb = obs["observation.images.image"]  # (B, 3, H, W) [0, 1]

    # Resize to 224×224 (ThermalGen + ImageBind requirement)
    if rgb.shape[-2] != _IMAGE_SIZE or rgb.shape[-1] != _IMAGE_SIZE:
        rgb = F.interpolate(rgb, size=(_IMAGE_SIZE, _IMAGE_SIZE), mode="bilinear", align_corners=False)

    B = rgb.shape[0]
    rgb = rgb.to(device)

    # Depth and seg are not produced by LIBERO — zeros are fine:
    # Block1.forward() skips ThermalGen when "thermal" is already in inputs,
    # but here we just provide the RGB and let ThermalGen generate thermal.
    inputs = {
        "rgb":   rgb,
        "depth": torch.zeros(B, 1, _IMAGE_SIZE, _IMAGE_SIZE, device=device),
        "seg":   torch.zeros(B, 1, _IMAGE_SIZE, _IMAGE_SIZE, device=device),
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

        inputs, batch = obs_to_pipeline_inputs(obs, task_descs, tokenizer, device, args.max_lang_tokens)

        if corruptor is not None:
            inputs = apply_eval_corruption(inputs, corruptor, args.corrupt_alpha)

        with torch.inference_mode():
            # Block1: RGB → ThermalGen → ImageBind embedding
            # Block2: {rgb, depth, seg, thermal} → 4M → MLP → SmolVLA → action
            # SmolVLA manages an internal action-chunk queue; one action is returned per call.
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
