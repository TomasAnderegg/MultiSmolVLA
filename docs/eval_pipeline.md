# Evaluating MultiSmolVLA on LIBERO

`scripts/eval_pipeline.py` runs the full multimodal pipeline on LIBERO simulation environments
and reports per-task and overall success rates.

The pipeline at eval time:

```
RGB (from env) ──► ThermalGen ──► ImageBind ──► Block1 embedding
                                                      │
depth = zeros ──────────────────────────────────────► 4M-21 encoder
seg   = zeros ──────────────────────────────────────► MLP connector
                                                      │
                                              SmolVLA (action head)
                                                      │
                                                  action (7-D)
```

Depth and segmentation are not available from LIBERO at eval time, so zero tensors are used.
ThermalGen synthesises the thermal modality from RGB only.

---

## Quick start

**Smoke test (no checkpoint, 1 episode):**
```bash
python scripts/eval_pipeline.py \
    --task libero_spatial \
    --task_ids 0 \
    --n_episodes 1 \
    --batch_size 1 \
    --output_dir ./eval_logs/smoke_test
```

**Full eval with a trained checkpoint:**
```bash
python scripts/eval_pipeline.py \
    --checkpoint checkpoints/full_pipeline/pipeline_final.pt \
    --task libero_object \
    --n_episodes 10 \
    --output_dir ./eval_logs/multismolvla_libero_object
```

**Via SLURM (edits `run_eval_pipeline.sh` config section first):**
```bash
sbatch scripts/run_eval_pipeline.sh
```

Results are written to `<output_dir>/eval_results.json`.

---

## All flags

### Checkpoint / model

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | *(none)* | Path to a `.pt` file from `train_full_pipeline.py`. Omit to use random/base weights (useful for smoke tests). |
| `--smolvla_checkpoint` | `lerobot/smolvla_libero` | HF Hub id or local path for the SmolVLA base weights. |
| `--fourm_checkpoint` | `EPFL-VILAB/4M-21_XL` | HF Hub id or local path for the 4M-21 encoder. |
| `--fourm_dim` | `1024` | Hidden dim of the 4M model. Use `768` for the B variant, `1024` for L/XL. |

### Environment

| Flag | Default | Description |
|---|---|---|
| `--task` | `libero_object` | LIBERO suite to evaluate: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`. |
| `--task_ids` | *(all)* | Space-separated list of task indices to run (0-indexed). Omit to run the full suite. |
| `--batch_size` | `1` | Number of parallel environments per task. |

### Evaluation

| Flag | Default | Description |
|---|---|---|
| `--n_episodes` | `10` | Number of episodes per task. |
| `--max_lang_tokens` | `48` | Max token length for task description tokenisation. |
| `--output_dir` | `./eval_logs/multismolvla` | Directory where `eval_results.json` is written. |
| `--device` | *(auto)* | `cuda` or `cpu`. Auto-detected if omitted. |

---

## Modality corruption flags

These flags let you corrupt specific input modalities at eval time to measure robustness.
Corruption is applied **after** `obs_to_pipeline_inputs` (i.e. on the tensors going into Block1)
and uses the same corruption kernels as training-time `ModalityDropout`.

| Flag | Default | Description |
|---|---|---|
| `--corrupt_modalities` | *(none)* | Which modalities to corrupt. Space-separated subset of `rgb depth seg thermal`. Omit for a clean eval. |
| `--corrupt_alpha` | `0.0` | Corruption intensity. `1.0` = clean (no effect), `0.5` = moderate soft corruption, `0.0` = modality zeroed out entirely (hard dropout). |
| `--corrupt_type` | *(random)* | Corruption kernel: `gaussian`, `blur`, or `occlusion`. Omit to pick randomly at each step. |

### Corruption semantics

The corruption blends the clean signal with a corrupted version:

```
output = alpha * x_clean + (1 - alpha) * x_corrupted
```

| `alpha` | Effect |
|---|---|
| `1.0` | Identity — clean signal, no change |
| `0.5` | 50 % clean / 50 % corrupted |
| `0.0` | Hard dropout — tensor zeroed out |

Corruption types:
- **`gaussian`** — additive Gaussian noise, sigma = `0.5 * (1 - alpha)`
- **`blur`** — horizontal motion blur, kernel size proportional to `1 - alpha` (up to 21 px)
- **`occlusion`** — centred black square, side proportional to `1 - alpha` (up to full image)

### Examples

Zero out RGB entirely (simulates camera failure):
```bash
python scripts/eval_pipeline.py \
    --checkpoint checkpoints/full_pipeline/pipeline_final.pt \
    --task libero_spatial \
    --n_episodes 10 \
    --corrupt_modalities rgb \
    --corrupt_alpha 0.0
```

Moderate Gaussian noise on RGB and depth:
```bash
python scripts/eval_pipeline.py \
    --checkpoint checkpoints/full_pipeline/pipeline_final.pt \
    --task libero_spatial \
    --n_episodes 10 \
    --corrupt_modalities rgb depth \
    --corrupt_alpha 0.5 \
    --corrupt_type gaussian
```

Heavy occlusion on RGB only:
```bash
python scripts/eval_pipeline.py \
    --checkpoint checkpoints/full_pipeline/pipeline_final.pt \
    --task libero_object \
    --n_episodes 10 \
    --corrupt_modalities rgb \
    --corrupt_alpha 0.2 \
    --corrupt_type occlusion
```

---

## Output format

`eval_results.json`:
```json
{
  "overall_success_rate": 72.0,
  "per_suite": {
    "libero_object": {
      "0": {"success_rate": 80.0, "successes": [true, true, false, ...]},
      "1": {"success_rate": 60.0, "successes": [true, false, true, ...]}
    }
  },
  "corruption": {
    "modalities": ["rgb"],
    "alpha": 0.5,
    "type": "gaussian"
  }
}
```
