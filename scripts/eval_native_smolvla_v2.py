#!/usr/bin/env python3
"""
Evaluate native SmolVLA (lerobot/smolvla_libero) on LIBERO — v2 architecture.

Optionally routes observations through a FourMImageProcessor before SigLIP:
  RGB (env) → 4M encoder → 4M decoder (single-pass) → DiVAE decode → RGB* → SigLIP → action

Phase 1 (--use_fourm, RGB only): validates the full pipeline end-to-end.
Phase 2/3: add --use_depth / --use_seg once validated.
"""

import json
import os
import sys
import traceback

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "third_party", "lerobot", "src"))

from lerobot.envs import make_env, preprocess_observation
from lerobot.envs.configs import LiberoEnv as LiberoEnvCfg
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def _extract_success(info, n_envs):
    if "is_success" in info:
        return np.asarray(info["is_success"], dtype=bool).reshape(n_envs)
    if "final_info" in info:
        fi = info["final_info"]
        result = np.zeros(n_envs, dtype=bool)
        for i, d in enumerate(fi):
            if isinstance(d, dict) and d.get("is_success", False):
                result[i] = True
        return result
    return np.zeros(n_envs, dtype=bool)


def _save_video(frames: list, path: str, fps: int) -> None:
    try:
        import imageio
        imageio.mimwrite(path, frames, fps=fps, codec="libx264", quality=8)
        print(f"[video] saved → {path}", flush=True)
    except Exception as e:
        print(f"[video] imageio failed ({e}), trying cv2 ...", flush=True)
        try:
            import cv2
            H, W, _ = frames[0].shape
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
            for f in frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"[video] saved → {path}", flush=True)
        except Exception as e2:
            print(f"[video] failed: {e2}", flush=True)


def _extract_rgb_frame(obs_raw: dict) -> np.ndarray | None:
    for key in obs_raw:
        v = obs_raw[key]
        if isinstance(v, np.ndarray) and v.ndim >= 3 and v.shape[-1] == 3:
            arr = v[0] if v.ndim == 4 else v
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
            return arr
    return None


def _find_image_key(obs: dict) -> str | None:
    """Return the first obs key that contains a (B, 3, H, W) float tensor."""
    for key, val in obs.items():
        if isinstance(val, torch.Tensor) and val.ndim == 4 and val.shape[1] == 3:
            return key
    return None


def _find_depth_key(obs: dict) -> str | None:
    """Return the depth key for the main camera (image_depth before image2_depth)."""
    candidates = [k for k in obs if "depth" in k.lower() and isinstance(obs[k], torch.Tensor)]
    if not candidates:
        return None
    # prefer the key without "2" (main agentview over wrist camera)
    main = [k for k in candidates if "image2" not in k]
    return main[0] if main else candidates[0]


def _find_seg_key(obs: dict) -> str | None:
    """Return the first obs key that looks like a segmentation map."""
    for key in obs:
        if "seg" in key.lower():
            return key
    return None


def _tensor_from_np(arr, ndim_target=4, channels=1) -> torch.Tensor:
    """Convert (H,W), (B,H,W), or (B,H,W,C) numpy array to (B,C,H,W) tensor."""
    t = torch.from_numpy(np.asarray(arr))
    if t.ndim == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.ndim == 3:
        t = t.unsqueeze(1)
    elif t.ndim == 4 and t.shape[-1] == channels:
        t = t.permute(0, 3, 1, 2)
    return t


def _inject_aux(obs: dict, obs_raw: dict) -> dict:
    """Inject depth/seg from obs_raw["aux"] as (B,1,H,W) float tensors.

    Depth and seg live in obs_raw["aux"] (not "pixels") so preprocess_observation
    never touches them. We inject them here with standard observation keys.
    """
    aux = obs_raw.get("aux", {})
    if not aux:
        return obs
    obs = dict(obs)
    for key, arr in aux.items():
        if arr is None:
            continue
        t = _tensor_from_np(arr, channels=1).float()
        obs[f"observation.images.{key}"] = t
    return obs


def _apply_fourm(obs: dict, fourm_processor, debug: bool = False,
                 estimator=None, modality_dropout=None) -> dict:
    """Replace the RGB observation with the 4M-reconstructed image.

    estimator        : RealtimeModalityEstimator — computes depth/seg from RGB when
                       they are not available in the env observation.
    modality_dropout : ModalityDropout — randomly corrupts/drops modalities before 4M
                       to test robustness (used at eval time to simulate obstructions).
    """
    img_key = _find_image_key(obs)
    if img_key is None:
        print("[fourm] WARNING: no (B,3,H,W) image found in obs — skipping 4M", flush=True)
        return obs

    rgb_in = obs[img_key].to(fourm_processor.device)

    depth_in = None
    if fourm_processor.use_depth:
        dep_key = _find_depth_key(obs)
        if dep_key is not None:
            depth_in = obs[dep_key].to(fourm_processor.device)
            if debug:
                print(f"  [fourm] depth key={dep_key}  shape={depth_in.shape}"
                      f" [{depth_in.min():.3f}, {depth_in.max():.3f}]", flush=True)
        elif estimator is not None and estimator.use_depth:
            estimated = estimator.estimate(rgb_in)
            depth_in = estimated.get("depth")
            if debug and depth_in is not None:
                print(f"  [fourm] depth estimated  shape={depth_in.shape}"
                      f" [{depth_in.min():.3f}, {depth_in.max():.3f}]", flush=True)
        else:
            print("[fourm] WARNING: use_depth=True but no depth in obs and no estimator", flush=True)

    seg_in = None
    if fourm_processor.use_seg:
        seg_key = _find_seg_key(obs)
        if seg_key is not None:
            seg_in = obs[seg_key].to(fourm_processor.device)
            if debug:
                print(f"  [fourm] seg key={seg_key}  shape={seg_in.shape}"
                      f" unique={seg_in.unique().numel()}", flush=True)
        elif estimator is not None and estimator.use_seg:
            if depth_in is not None:
                seg_in = estimator._estimate_seg(rgb_in, depth_in)
            else:
                estimated = estimator.estimate(rgb_in)
                seg_in = estimated.get("seg")
            if debug and seg_in is not None:
                print(f"  [fourm] seg estimated  shape={seg_in.shape}"
                      f" unique={seg_in.unique().numel()}", flush=True)
        else:
            print("[fourm] WARNING: use_seg=True but no seg in obs and no estimator", flush=True)

    # ── ModalityDropout : simulate obstructions ───────────────────────────────
    if modality_dropout is not None:
        modality_inputs = {"rgb": rgb_in}
        if depth_in is not None:
            modality_inputs["depth"] = depth_in
        if seg_in is not None:
            modality_inputs["seg"] = seg_in

        modality_inputs = modality_dropout(modality_inputs)

        rgb_in   = modality_inputs["rgb"]
        depth_in = modality_inputs.get("depth")
        seg_in   = modality_inputs.get("seg")

        if debug:
            print(f"  [dropout] rgb=[{rgb_in.min():.3f},{rgb_in.max():.3f}]"
                  f"  depth={depth_in is not None}  seg={seg_in is not None}", flush=True)

    if getattr(fourm_processor, '_divae_only', False):
        rgb_out = fourm_processor.divae_roundtrip(
            rgb_in,
            ddim_steps=getattr(fourm_processor, '_ddim_steps', 50),
            input_size=getattr(fourm_processor, '_divae_input_size', 448),
            bypass_vq=getattr(fourm_processor, '_bypass_vq', True),
        )
    else:
        rgb_out = fourm_processor.process(rgb_in, depth=depth_in, seg=seg_in)

    if debug:
        print(f"  [fourm] key={img_key}  in={rgb_in.shape} [{rgb_in.min():.3f}, {rgb_in.max():.3f}]"
              f"  out={rgb_out.shape} [{rgb_out.min():.3f}, {rgb_out.max():.3f}]", flush=True)

    obs = dict(obs)
    obs[img_key] = rgb_out.to(rgb_in.device)
    return obs


def run_episode(vec_env, policy, env_preprocessor, preprocessor, postprocessor,
                device, fourm_processor=None, estimator=None, modality_dropout=None,
                debug=False, save_video_path: str | None = None) -> list[bool]:
    policy.reset()
    obs_raw, _ = vec_env.reset()
    frames: list[np.ndarray] = []
    if save_video_path:
        f = _extract_rgb_frame(obs_raw)
        if f is not None:
            frames.append(f)

    obs = preprocess_observation(obs_raw)
    obs = env_preprocessor(obs)
    obs = _inject_aux(obs, obs_raw)

    try:
        task_descs = list(vec_env.call("task_description"))
    except (AttributeError, NotImplementedError):
        try:
            task_descs = list(vec_env.call("task"))
        except Exception:
            task_descs = [""] * vec_env.num_envs
    obs["task"] = task_descs

    try:
        max_steps = vec_env.call("_max_episode_steps")[0]
    except Exception:
        max_steps = 300

    if debug:
        print(f"  [dbg] task_desc='{task_descs[0]}'  max_steps={max_steps}", flush=True)
        pixels = obs_raw.get("pixels", {})
        if isinstance(pixels, dict):
            print(f"  [dbg] raw obs pixels keys: {list(pixels.keys())}", flush=True)
            for k, v in pixels.items():
                arr = np.asarray(v)
                print(f"  [dbg]   {k}: shape={arr.shape} dtype={arr.dtype}", flush=True)
        print(f"  [dbg] processed obs keys: {[k for k in obs.keys() if 'image' in k]}", flush=True)

    done    = np.zeros(vec_env.num_envs, dtype=bool)
    success = np.zeros(vec_env.num_envs, dtype=bool)

    for step_i in range(max_steps):
        if done.all():
            break

        # ── Optional: route RGB through 4M before SmolVLA's SigLIP ──────────
        step_obs = obs
        if fourm_processor is not None:
            step_obs = _apply_fourm(obs, fourm_processor, debug=(debug and step_i < 3),
                                    estimator=estimator, modality_dropout=modality_dropout)

        batch = preprocessor(step_obs)

        if debug and step_i < 3:
            img_key = _find_image_key(step_obs)
            if img_key:
                t = step_obs[img_key]
                print(f"  [dbg] step={step_i} rgb shape={t.shape} [{t.min():.3f}, {t.max():.3f}]", flush=True)
            if "observation.state" in batch:
                s = batch["observation.state"]
                print(f"  [dbg] step={step_i} state min={s.min():.3f} max={s.max():.3f}", flush=True)

        with torch.inference_mode():
            action = policy.select_action(batch)

        action = postprocessor(action)

        if debug and step_i < 3:
            print(f"  [dbg] step={step_i} action={np.round(action.cpu().numpy()[0], 3)}", flush=True)

        action_np = action.cpu().numpy()
        obs_raw, _reward, terminated, truncated, info = vec_env.step(action_np)
        if save_video_path:
            f = _extract_rgb_frame(obs_raw)
            if f is not None:
                frames.append(f)

        obs = preprocess_observation(obs_raw)
        obs = env_preprocessor(obs)
        obs = _inject_aux(obs, obs_raw)
        obs["task"] = task_descs

        step_success = _extract_success(info, vec_env.num_envs)
        success |= step_success
        done    |= np.asarray(terminated | truncated, dtype=bool)

    if debug:
        print(f"  [dbg] episode done  step={step_i}  success={success.tolist()}", flush=True)

    if save_video_path and frames:
        _save_video(frames, save_video_path, fps=10)

    return success.tolist()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",          default="libero_10")
    parser.add_argument("--task_ids",      type=int, nargs="+", default=None)
    parser.add_argument("--n_episodes",    type=int, default=5)
    parser.add_argument("--batch_size",    type=int, default=1)
    parser.add_argument("--output_dir",    default="eval_results/smolvla_native_v2")
    parser.add_argument("--model_id",      default="lerobot/smolvla_libero")
    parser.add_argument("--record_video",  action="store_true")
    # 4M options
    parser.add_argument("--use_fourm",     action="store_true",
                        help="Route observations through 4M before SigLIP.")
    parser.add_argument("--fourm_model",   default="EPFL-VILAB/4M-21_XL")
    parser.add_argument("--divae_steps",      type=int,   default=50,
                        help="Number of DiVAE diffusion steps for RGB decode (default 50).")
    parser.add_argument("--tok_temperature", type=float, default=3.0,
                        help="Token sampling temperature. 0=argmax, >0=multinomial sampling (default 3.0).")
    parser.add_argument("--cfg_scale",      type=float, default=1.0,
                        help="Classifier-Free Guidance scale (1.0=off, 2.0=recommended, official 4M default).")
    parser.add_argument("--use_depth",    action="store_true",
                        help="Add depth as additional 4M encoder input (tok_depth@224).")
    parser.add_argument("--use_seg",      action="store_true",
                        help="Add segmentation as additional 4M encoder input (tok_semseg@224).")
    parser.add_argument("--maskgit_steps",  type=int,   default=4,
                        help="MaskGIT iterative decoding steps (default 4).")
    parser.add_argument("--use_tok_encoder", action="store_true",
                        help="Phase 3: encode via DiVAE tokenize → tok_rgb@224 (patch-level, domain-robust).")
    parser.add_argument("--divae_only", action="store_true",
                        help="Diagnostic: pure DiVAE tokenize→decode, skip 4M transformer entirely.")
    parser.add_argument("--rgb_passthrough", action="store_true",
                        help="4M encoder runs with all inputs (RGB+depth+seg), but MaskGIT is skipped "
                             "and original RGB tokens are decoded directly — perfect reconstruction "
                             "while depth/seg still appear in debug images.")
    parser.add_argument("--ddim_steps",  type=int, default=50,
                        help="DDIM steps for DiVAE decode in --divae_only mode (default 50).")
    parser.add_argument("--divae_input_size", type=int, default=448,
                        help="Input resolution fed to DiVAE in --divae_only mode: 224→14×14 tokens, 448→28×28 (default 448).")
    parser.add_argument("--no_bypass_vq", action="store_true",
                        help="Disable VQ bypass: use discrete codebook quantization (default: bypass for sharper output).")
    parser.add_argument("--lora_checkpoint", type=str, default=None,
                        help="Path to LoRA weights (.pt) from finetune_4m_libero.py — "
                             "injects and loads LoRA adapters into the 4M transformer.")
    # Real-time modality estimation (when env doesn't provide depth/seg)
    parser.add_argument("--use_estimator", action="store_true",
                        help="Estimate depth and seg from RGB in real-time using "
                             "Depth Anything V2 Small + k-means clustering.")
    parser.add_argument("--depth_model", type=str,
                        default="depth-anything/Depth-Anything-V2-Small-hf",
                        help="HuggingFace model ID for depth estimation.")
    parser.add_argument("--n_seg_clusters", type=int, default=16,
                        help="Number of k-means clusters for pseudo-segmentation (default 16).")
    # Modality dropout — simulate obstructions at eval time
    parser.add_argument("--p_drop_rgb",   type=float, default=0.0,
                        help="Probability to corrupt/drop RGB before 4M (0=off, 1=always drop).")
    parser.add_argument("--p_drop_depth", type=float, default=0.0,
                        help="Probability to corrupt/drop depth before 4M.")
    parser.add_argument("--p_drop_seg",   type=float, default=0.0,
                        help="Probability to corrupt/drop seg before 4M.")
    parser.add_argument("--dropout_alpha", type=float, default=0.0,
                        help="Corruption strength: 0=hard zero, 1=clean. "
                             "Controls gaussian/blur/occlusion intensity.")
    args = parser.parse_args()

    try:
        _main(args)
    except Exception:
        print("FATAL ERROR:", flush=True)
        print(traceback.format_exc(), flush=True)
        raise


def _main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] Device: {device}", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── SmolVLA ──────────────────────────────────────────────────────────────
    print(f"[eval] Loading {args.model_id} ...", flush=True)
    policy = SmolVLAPolicy.from_pretrained(args.model_id)
    policy.eval().to(device)
    print("[eval] Policy loaded", flush=True)

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=args.model_id
    )

    # ── Optional: 4M image processor ─────────────────────────────────────────
    fourm_processor = None
    if args.use_fourm:
        from src.pipeline.fourm_image_processor_v2 import FourMImageProcessor
        # rgb_passthrough needs discrete tokens from DiVAE encoder
        use_tok_enc = args.use_tok_encoder or args.rgb_passthrough
        fourm_processor = FourMImageProcessor(
            fourm_checkpoint=args.fourm_model,
            divae_steps=args.divae_steps,
            tok_temperature=args.tok_temperature,
            cfg_scale=args.cfg_scale,
            use_depth=args.use_depth,
            use_seg=args.use_seg,
            n_maskgit_steps=args.maskgit_steps,
            use_tok_encoder=use_tok_enc,
            lora_checkpoint=args.lora_checkpoint,
            device=device,
        )
        fourm_processor._debug_save_dir    = os.path.join(args.output_dir, "debug_images")
        fourm_processor._debug_counter     = 0
        fourm_processor._divae_only        = args.divae_only
        fourm_processor._rgb_passthrough   = args.rgb_passthrough
        fourm_processor._ddim_steps        = args.ddim_steps
        fourm_processor._divae_input_size  = args.divae_input_size
        fourm_processor._bypass_vq         = not args.no_bypass_vq
        inputs = "RGB" + (" + depth" if args.use_depth else "") + (" + seg" if args.use_seg else "")
        mode   = "divae_only" if args.divae_only else ("rgb_passthrough" if args.rgb_passthrough else "full_4M")
        print(f"[eval] FourMImageProcessor ready  inputs={inputs}  mode={mode}  "
              f"ddim_steps={args.ddim_steps}  bypass_vq={not args.no_bypass_vq}", flush=True)
    else:
        print("[eval] Running without 4M (native SmolVLA baseline)", flush=True)

    # ── Optional: real-time depth/seg estimator ───────────────────────────────
    estimator = None
    if args.use_estimator and args.use_fourm and (args.use_depth or args.use_seg):
        from src.pipeline.realtime_modality_estimator import RealtimeModalityEstimator
        estimator = RealtimeModalityEstimator(
            device=device,
            depth_model=args.depth_model,
            n_seg_clusters=args.n_seg_clusters,
            use_depth=args.use_depth,
            use_seg=args.use_seg,
        )

    # ── Optional: modality dropout for robustness testing ─────────────────────
    modality_dropout = None
    any_drop = args.p_drop_rgb > 0 or args.p_drop_depth > 0 or args.p_drop_seg > 0
    if any_drop and args.use_fourm:
        from src.pipeline.modality_dropout import ModalityDropout

        class _FixedDropout(ModalityDropout):
            """ModalityDropout with per-modality drop probabilities."""
            def __init__(self, p_rgb, p_depth, p_seg, alpha):
                super().__init__(p_drop=0.0, alpha_min=alpha, total_epochs=1)
                self._p = {"rgb": p_rgb, "depth": p_depth, "seg": p_seg}
                self._alpha = alpha

            def forward(self, inputs: dict, epoch: int = 0) -> dict:
                import random
                outputs = {}
                for mod, x in inputs.items():
                    p = self._p.get(mod, 0.0)
                    if p > 0 and random.random() < p:
                        outputs[mod] = self.corrupt(x, self._alpha)
                    else:
                        outputs[mod] = x
                return outputs

        modality_dropout = _FixedDropout(
            p_rgb=args.p_drop_rgb,
            p_depth=args.p_drop_depth,
            p_seg=args.p_drop_seg,
            alpha=args.dropout_alpha,
        )
        modality_dropout.eval()  # no random drop at eval unless p_drop > 0
        print(f"[eval] ModalityDropout  p_rgb={args.p_drop_rgb}  "
              f"p_depth={args.p_drop_depth}  p_seg={args.p_drop_seg}  "
              f"alpha={args.dropout_alpha}", flush=True)

    # ── LIBERO envs ───────────────────────────────────────────────────────────
    env_cfg = LiberoEnvCfg(
        task=args.task,
        task_ids=args.task_ids,
        obs_type="pixels_agent_pos",
        enable_depth=False,   # always False — we use the estimator instead
        enable_seg=False,
    )
    envs = make_env(env_cfg, n_envs=args.batch_size)
    env_preprocessor, _ = env_cfg.get_env_processors()

    all_results  = {}
    all_successes = []
    video_base = os.path.join(args.output_dir, "videos") if args.record_video else None

    for suite_name, task_envs in envs.items():
        print(f"\n── Suite: {suite_name} ({len(task_envs)} tasks) ──", flush=True)
        for task_id, vec_env in task_envs.items():
            successes = []
            ep_idx = 0
            while len(successes) < args.n_episodes:
                debug      = (ep_idx == 0)  # debug first episode of every task
                video_path = None
                if video_base and ep_idx == 0:
                    os.makedirs(video_base, exist_ok=True)
                    video_path = os.path.join(video_base, f"task{task_id}_ep0.mp4")
                ep = run_episode(
                    vec_env, policy, env_preprocessor, preprocessor, postprocessor,
                    device, fourm_processor=fourm_processor,
                    estimator=estimator, modality_dropout=modality_dropout,
                    debug=debug, save_video_path=video_path,
                )
                successes.extend(ep)
                ep_idx += 1
            successes = successes[:args.n_episodes]
            sr = float(np.mean(successes)) * 100
            print(f"  task_id={task_id} → {sr:.1f}% ({args.n_episodes} episodes)  successes={successes}", flush=True)
            all_results[str(task_id)] = {"success_rate": sr, "successes": successes}
            all_successes.extend(successes)

    overall = float(np.mean(all_successes)) * 100
    print(f"\n[eval] Overall: {overall:.1f}%", flush=True)

    result = {"overall_success_rate": overall, "per_task": all_results,
              "config": {"use_fourm": args.use_fourm, "divae_steps": args.divae_steps,
                         "maskgit_steps": args.maskgit_steps, "fourm_model": args.fourm_model}}
    out_path = os.path.join(args.output_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[eval] Results → {out_path}", flush=True)


if __name__ == "__main__":
    main()
