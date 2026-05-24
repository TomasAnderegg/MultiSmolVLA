#!/usr/bin/env python3
"""
Evaluate native SmolVLA (lerobot/smolvla_libero) on LIBERO.
Uses lerobot's own preprocessor/postprocessor pipeline for correct normalization.
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
    """Robustly extract is_success from vectorized env info."""
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


def run_episode(vec_env, policy, env_preprocessor, preprocessor, postprocessor,
                device, debug=False) -> list[bool]:
    policy.reset()
    obs_raw, _ = vec_env.reset()
    obs = preprocess_observation(obs_raw)
    obs = env_preprocessor(obs)

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
        rgb_key = "observation.images.camera1"
        if rgb_key in obs:
            t = obs[rgb_key]
            print(f"  [dbg] before-preproc rgb shape={t.shape} dtype={t.dtype} min={t.min():.3f} max={t.max():.3f}", flush=True)

    done = np.zeros(vec_env.num_envs, dtype=bool)
    success = np.zeros(vec_env.num_envs, dtype=bool)

    for step_i in range(max_steps):
        if done.all():
            break

        # Apply policy preprocessor (normalizes state, tokenizes language, moves to device)
        batch = preprocessor(obs)

        if debug and step_i == 0:
            print(f"  [dbg] batch_keys={list(batch.keys())}", flush=True)
            if "observation.state" in batch:
                s = batch["observation.state"]
                print(f"  [dbg] state after norm shape={s.shape} min={s.min():.3f} max={s.max():.3f}", flush=True)

        with torch.inference_mode():
            action = policy.select_action(batch)

        # Apply policy postprocessor (unnormalizes actions)
        action = postprocessor(action)

        if debug and step_i < 3:
            print(f"  [dbg] step={step_i} action={np.round(action.cpu().numpy()[0], 3)}", flush=True)

        action_np = action.cpu().numpy()
        obs_raw, _reward, terminated, truncated, info = vec_env.step(action_np)
        obs = preprocess_observation(obs_raw)
        obs = env_preprocessor(obs)
        obs["task"] = task_descs

        step_success = _extract_success(info, vec_env.num_envs)
        success |= step_success
        done |= np.asarray(terminated | truncated, dtype=bool)

        if debug and step_i == 0:
            print(f"  [dbg] info_keys={list(info.keys())}", flush=True)
            print(f"  [dbg] is_success={info.get('is_success', 'N/A')}", flush=True)

    if debug:
        print(f"  [dbg] episode done step={step_i} success={success.tolist()}", flush=True)

    return success.tolist()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",        default="libero_10")
    parser.add_argument("--n_episodes",  type=int, default=5)
    parser.add_argument("--batch_size",  type=int, default=1)
    parser.add_argument("--output_dir",  default="eval_results/smolvla_native_libero_10")
    parser.add_argument("--model_id",    default="lerobot/smolvla_libero")
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

    print(f"[eval] Loading {args.model_id} ...", flush=True)
    policy = SmolVLAPolicy.from_pretrained(args.model_id)
    policy.eval().to(device)
    print("[eval] Policy loaded", flush=True)

    print("[eval] Loading pre/post processors ...", flush=True)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=args.model_id
    )
    print(f"[eval] preprocessor={type(preprocessor).__name__}  postprocessor={type(postprocessor).__name__}", flush=True)

    env_cfg = LiberoEnvCfg(task=args.task, obs_type="pixels_agent_pos")
    print("[eval] Calling make_env ...", flush=True)
    envs = make_env(env_cfg, n_envs=args.batch_size)
    env_preprocessor, _ = env_cfg.get_env_processors()
    print(f"[eval] envs keys={list(envs.keys())}  env_preprocessor={type(env_preprocessor).__name__}", flush=True)

    all_results = {}
    all_successes = []

    for suite_name, task_envs in envs.items():
        print(f"\n── Suite: {suite_name} ({len(task_envs)} tasks) ──", flush=True)
        for task_id, vec_env in task_envs.items():
            successes = []
            while len(successes) < args.n_episodes:
                debug = (task_id == 0 and len(successes) == 0)
                ep = run_episode(vec_env, policy, env_preprocessor, preprocessor,
                                 postprocessor, device, debug=debug)
                successes.extend(ep)
            successes = successes[:args.n_episodes]
            sr = float(np.mean(successes)) * 100
            print(f"  task_id={task_id} → {sr:.1f}% ({args.n_episodes} episodes)", flush=True)
            all_results[str(task_id)] = {"success_rate": sr, "successes": successes}
            all_successes.extend(successes)

    overall = float(np.mean(all_successes)) * 100
    print(f"\n[eval] Overall success rate: {overall:.1f}%", flush=True)

    result = {"overall_success_rate": overall, "per_task": all_results}
    out_path = os.path.join(args.output_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[eval] Results saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
