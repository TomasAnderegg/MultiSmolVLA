# Training Guide — MultiSmolVLA

## Overview

Training uses `scripts/train_full_pipeline.py`.

**Block 1**: pre-computed thermal (from parquet) → ModalityDropout → ImageBind → thermal embedding (B, 1024). ThermalGen is skipped when using `--data_dir`.
**Block 2**: all 4 modalities + language → 4M-21 → MLP connector → SmolVLA → action chunk (B, 50, 7).

Training objective: flow matching. One observation in, predict the next 50 ground-truth actions.

Always use `--data_dir` pointing to the parquet shards directory. Each shard is one episode and contains RGB, depth, seg, thermal (pre-computed), state, action, and task_index. The dataset builds 50-step action chunks automatically and provides `action_is_pad` to mask padded episode-end steps from the loss.

---

## Dataset

The dataset (`TomasAnderegg/libero_10_thermal`) is on HuggingFace and already contains all 4 modalities pre-computed. Download it once with:

```bash
HF_TOKEN=<your_hf_token> python data/download_training_dataset.py
```

Go into the file "download_training_dataset.py" and update OUTPUT_DIR with your username
Shards are saved to `/scratch/izar/<USER NAME>/data/parquet_thermal`.

---

## Training

Template:
```bash
python scripts/train_full_pipeline.py \
    --data_dir /path/to/parquet_thermal \
    --freeze_thermalgen --freeze_imagebind \
    --lr 1e-5 --lr_mlp 1e-3 \
    --batch_size 4 --steps 10000
    -- OTHER FLAGS
```

Smoke test (5 steps, verifies loss + backward pass, logs to WandB):
```bash
sbatch scripts/run_test_training.sh
```

---

## All flags

### Data
| Flag | Default | Description |
|------|---------|-------------|
| `--data_dir` | `None` | Path to parquet shards directory. **Use this.** Always provides `action_is_pad`. |
| `--dataset` | `lerobot/libero_spatial_no_noops` | HuggingFace dataset fallback (no pre-computed thermal, no `action_is_pad`) |
| `--image_key` | `observation.images.top` | RGB key when using `--dataset` |
| `--chunk_size` | `50` | Action steps per training sample. Must match SmolVLA config. |

### Checkpoints
| Flag | Default | Description |
|------|---------|-------------|
| `--smolvla_checkpoint` | `lerobot/smolvla_libero` | Pretrained SmolVLA weights |
| `--fourm_checkpoint` | `EPFL-VILAB/4M-21_XL` | 4M-21 encoder checkpoint |
| `--fourm_model` | `None` | Shortcut: `B` (768d), `L`/`XL` (1024d). Overrides `--fourm_checkpoint` and `--fourm_dim`. |
| `--fourm_dim` | `1024` | 4M output dim (768 for B, 1024 for L/XL) |
| `--output_dir` | `checkpoints/full_pipeline` | Where to save checkpoints |

### Freeze flags
| Flag | Description |
|------|-------------|
| `--freeze_thermalgen` | Freeze ThermalGen. No-op with `--data_dir` (thermal already pre-computed). |
| `--freeze_imagebind` | Freeze ImageBind. Keep frozen — large pretrained encoder. |
| `--freeze_4m` | Freeze 4M-21 encoder |
| `--freeze_mlp` | Freeze MLP connector |
| `--freeze_smolvlm` | Freeze SmolVLM backbone |
| `--freeze_action_expert` | Freeze action expert (flow matching head) |

### Training hyperparameters
| Flag | Default | Description |
|------|---------|-------------|
| `--lr` | `1e-5` | LR for all trainable params except MLP |
| `--lr_mlp` | `1e-3` | Separate LR for MLP connector (randomly initialized, needs higher LR) |
| `--batch_size` | `4` | Batch size |
| `--steps` | `10000` | Total gradient steps |
| `--save_every` | `1000` | Checkpoint every N steps |
| `--log_every` | `50` | Log average loss every N steps |
| `--num_workers` | `4` | DataLoader workers |
| `--max_lang_tokens` | `48` | Max tokenized length for language instructions |

### ModalityDropout
| Flag | Default | Description |
|------|---------|-------------|
| `--p_drop` | `0.5` | Probability of corrupting each modality per step |
| `--alpha_min` | `0.0` | Min blend factor at end of curriculum (0.0 = full zero-out, 1.0 = no corruption) |
| `--total_epochs` | `100` | Steps over which alpha decays 1.0 → alpha_min |
| `--modalities` | `rgb depth seg thermal` | Modalities to apply dropout to |
| `--corruption_types` | `gaussian blur occlusion` | Corruption types to sample from |
