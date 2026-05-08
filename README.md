# Sensor-Robust Multimodal VLA via Modality-Dropout Adapter Training on SmolVLA

**CS-503 Visual Intelligence — EPFL, 2026**

*Alix Papadatos · Florian Tanguy · Mario Fernández · Tomas Garate Anderegg*

```bash
git clone --recurse-submodules <url_de_ton_repo>
```


---

## Overview

State-of-the-art Vision-Language-Action (VLA) models rely mostly on RGB perception and suffer catastrophic performance degradation under sensor failure. We propose a training strategy that mitigates this dependence by replacing SmolVLA's SigLIP encoder with the frozen **4M-21** multimodal encoder, adding a lightweight **MLP connector**, and training under a **modality-dropout curriculum** to build robustness to sensor failures and full modality dropout.

---

## Pipeline

![Pipeline](general_pipeline.drawio.png)

*Block 1 (Perception): Four input modalities (RGB, depth, segmentation, thermal) pass through a modality-dropout layer during training. Thermal is synthesized from RGB via ThermalGen and embedded through ImageBind into a format natively supported by 4M-21. The frozen 4M-21 encoder fuses all available modalities into a unified token sequence.*

*Block 2 (Action): Multimodal tokens are projected into SmolLM2's embedding space via a MLP connector. SmolLM2, concatenated with text and robot state tokens, conditions the action expert to generate continuous action chunks.*

---

## Method

### Architecture

| Component | Role | Frozen? |
|---|---|---|
| ImageBind | Thermal → embedding compatible with 4M-21 | ✅ |
| 4M-21 encoder | Fuses RGB + depth + seg + thermal → token sequence | ✅ |
| MLP connector | Projects 4M tokens → SmolLM2 token space | ❌ (Stage 1) |
| SmolLM2 | Language decoder, conditions action expert | ❌ (Stage 2, LoRA) |
| Action expert | Generates continuous action chunks | ❌ (Stage 2, LoRA) |

### Training Strategy

**Stage 1 — Connector alignment:** Train only the MLP connector to align 4M-21 features with SmolLM2's expected token distribution. All other components frozen.

**Stage 2 — Robustness fine-tuning:** Fine-tune the full model with LoRA adapters under a modality-dropout curriculum. Each modality is independently zeroed out with probability $p_{\text{drop}}$, increasing linearly from 0 to 0.5 over training.

### Modality Dropout

During training, each modality is independently zeroed out before being passed to 4M-21. This forces the model to learn to fuse all modalities when available, and to compensate for missing ones progressively.

---

## Dataset

We use the [`binhng/original-libero`](https://huggingface.co/datasets/binhng/original-libero) curated dataset on HuggingFace, derived from the [LIBERO benchmark](https://libero-project.github.io/), which provides aligned RGB, semantic segmentation, and depth map modalities. Thermal representations are synthetically generated from RGB using [ThermalGen](https://openreview.net/forum?id=o0JSYq1TQ4).

---

## Evaluation

We evaluate on the four LIBERO task suites: **Spatial**, **Object**, **Goal**, **Long**.

| Condition | Description |
|---|---|
| Clean | All modalities available, no corruption |
| Hard dropout | One or more modalities zeroed at inference time |
| Soft corruption | Gaussian noise, motion blur, centered black-square occlusion |

**Baselines:**
- Vanilla SmolVLA (RGB only, no dropout training): 87.3% avg task completion
- Vanilla π0 (RGB only, no dropout training): 86% avg task completion

**Ablations:**
- (a) w/ vs. w/o additional modalities
- (b) Fixed dropout vs. curriculum dropout schedule

---

## Project Structure

```
MultiSmolVLA/
├── src/
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── encoder_4m.py           # 4M-21 encoder wrapper
│   │   ├── imagebind_encoder.py    # ImageBind thermal encoder
│   │   ├── thermalgen_encoder.py   # ThermalGen RGB→thermal
│   │   ├── modality_dropout.py     # Curriculum modality dropout
│   │   ├── connector.py            # MLP connector (LLaVA-1.5 style)
│   │   ├── smolvla_wrapper.py      # SmolVLA wrapper
│   │   └── full_pipeline.py        # End-to-end pipeline
│   └── utils/
│       ├── __init__.py
│       └── debug.py
├── scripts/
│   ├── test_pipeline.py            # End-to-end sanity check
│   └── test_dropout.py             # Test modality dropout seul
├── notebooks/                      # Debug & visualization
├── data/                           # LIBERO dataset (not tracked)
├── models/                         # Checkpoints (not tracked)
├── third_party/                    # Source deps (not tracked)
│   ├── lerobot/
│   ├── ml-4m/
│   ├── ImageBind/
│   └── ThermalGen/
├── assets/
│   └── pipeline.png                # Pipeline figure
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Create environment

```bash
conda create -n vla python=3.12 -y
conda activate vla
```

### 2. Install PyTorch (CUDA)

```bash
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install source dependencies

```bash
# LeRobot (SmolVLA)
git clone https://github.com/huggingface/lerobot.git third_party/lerobot
cd third_party/lerobot && git checkout -b vla_custom && pip install -e . && cd ../..

# 4M-21
git clone https://github.com/apple/ml-4m.git third_party/ml-4m
pip install --no-deps -e third_party/ml-4m

# ImageBind
git clone https://github.com/facebookresearch/ImageBind.git third_party/ImageBind
pip install --no-deps -e third_party/ImageBind
pip install pytorchvideo types-regex
```

> ⚠️ **Windows fix for pytorchvideo:** In `site-packages/pytorchvideo/transforms/augmentations.py`, replace `import torchvision.transforms.functional_tensor as F_t` with `import torchvision.transforms.functional as F_t`.

### 5. Apply required patch to lerobot

Python 3.12 has a stricter dataclass rule that breaks one file in lerobot. Open `third_party/lerobot/src/lerobot/policies/groot/groot_n1.py` and find the `GR00TN15Config` class (~line 177). Add `default=None` to the four `init=False` fields:

```python
# Before
backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})
action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})
action_horizon: int = field(init=False, metadata={"help": "Action horizon."})
action_dim: int = field(init=False, metadata={"help": "Action dimension."})

# After
backbone_cfg: dict = field(init=False, default=None, metadata={"help": "Backbone configuration."})
action_head_cfg: dict = field(init=False, default=None, metadata={"help": "Action head configuration."})
action_horizon: int = field(init=False, default=None, metadata={"help": "Action horizon."})
action_dim: int = field(init=False, default=None, metadata={"help": "Action dimension."})
```

### 6. Run sanity check

```bash
python scripts/test_pipeline.py # Full pipeline — requires GPU
```

---

## CLI Reference

### `scripts/train_block2.py` — Stage 1 : alignement du connecteur MLP

```bash
python scripts/train_block2.py [OPTIONS]
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--dataset` | str | `lerobot/libero_spatial_no_noops` | Repo HuggingFace du dataset |
| `--image_key` | str | `observation.images.top` | Clé du dataset utilisée comme entrée RGB dans 4M |
| `--smolvla_checkpoint` | str | `lerobot/smolvla_libero` | Checkpoint SmolVLA de base |
| `--output_dir` | str | `checkpoints/block2` | Dossier de sauvegarde des checkpoints |
| `--fourm_model` | `B\|L\|XL` | `None` | Variante 4M-21 (B=768d, L/XL=1024d). Override `--fourm_checkpoint` et `--fourm_dim` |
| `--fourm_checkpoint` | str | `EPFL-VILAB/4M-21_XL` | Checkpoint 4M-21 HuggingFace |
| `--fourm_dim` | int | `1024` | Dimension des features 4M-21 |
| `--freeze_4m` | flag | `False` | Gèle l'encodeur 4M |
| `--freeze_mlp` | flag | `False` | Gèle le connecteur MLP |
| `--freeze_smolvlm` | flag | `False` | Gèle le LLM SmolVLM |
| `--freeze_action_expert` | flag | `False` | Gèle l'action expert |
| `--lr` | float | `1e-4` | Learning rate |
| `--batch_size` | int | `4` | Taille du batch |
| `--steps` | int | `10000` | Nombre de steps d'entraînement |
| `--save_every` | int | `1000` | Fréquence de sauvegarde (en steps) |
| `--log_every` | int | `50` | Fréquence de logging (en steps) |
| `--num_workers` | int | `4` | Workers du DataLoader |
| `--max_lang_tokens` | int | `48` | Longueur max des tokens de langage |
| `--dummy` | flag | `False` | Utilise des inputs aléatoires (smoke-test sans dataset) |
| `--dummy_state_dim` | int | `7` | Dimension de l'état robot en mode dummy |
| `--dummy_action_dim` | int | `7` | Dimension des actions en mode dummy |
| `--dummy_action_steps` | int | `50` | Chunk size en mode dummy (doit matcher `chunk_size=50`) |

**Exemples :**
```bash
# Stage 1 standard
python scripts/train_block2.py --fourm_model XL --lr 1e-4 --batch_size 8 --steps 20000

# Smoke-test rapide sans GPU ni dataset
python scripts/train_block2.py --dummy --steps 10 --fourm_model B

# Geler tout sauf le connecteur MLP
python scripts/train_block2.py --freeze_4m --freeze_smolvlm --freeze_action_expert
```

---

### `scripts/train_full_pipeline.py` — Stage 2 : fine-tuning robustesse

```bash
python scripts/train_full_pipeline.py [OPTIONS]
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--dataset` | str | `lerobot/libero_spatial_no_noops` | Repo HuggingFace du dataset |
| `--image_key` | str | `observation.images.top` | Clé du dataset utilisée comme entrée RGB |
| `--smolvla_checkpoint` | str | `lerobot/smolvla_libero` | Checkpoint SmolVLA de base |
| `--fourm_model` | `B\|L\|XL` | `None` | Variante 4M-21 |
| `--fourm_checkpoint` | str | `EPFL-VILAB/4M-21_XL` | Checkpoint 4M-21 |
| `--fourm_dim` | int | `1024` | Dimension des features 4M-21 |
| `--output_dir` | str | `checkpoints/full_pipeline` | Dossier de sauvegarde |
| `--freeze_thermalgen` | flag | `False` | Gèle ThermalGen (générateur RGB→thermal) |
| `--freeze_imagebind` | flag | `False` | Gèle l'encodeur thermal ImageBind |
| `--freeze_4m` | flag | `False` | Gèle l'encodeur 4M |
| `--freeze_mlp` | flag | `False` | Gèle le connecteur MLP |
| `--freeze_smolvlm` | flag | `False` | Gèle le LLM SmolVLM |
| `--freeze_action_expert` | flag | `False` | Gèle l'action expert |
| `--lr` | float | `1e-5` | Learning rate global |
| `--lr_mlp` | float | `1e-3` | Learning rate spécifique au connecteur MLP |
| `--batch_size` | int | `4` | Taille du batch |
| `--steps` | int | `10000` | Nombre de steps d'entraînement |
| `--save_every` | int | `1000` | Fréquence de sauvegarde (en steps) |
| `--log_every` | int | `50` | Fréquence de logging (en steps) |
| `--num_workers` | int | `4` | Workers du DataLoader |
| `--max_lang_tokens` | int | `48` | Longueur max des tokens de langage |

**Exemples :**
```bash
# Stage 2 complet — tout entraînable
python scripts/train_full_pipeline.py --fourm_model XL --lr 1e-5 --lr_mlp 1e-3 --steps 30000

# Geler Block 1, fine-tuner seulement Block 2
python scripts/train_full_pipeline.py --freeze_thermalgen --freeze_imagebind --freeze_4m
```

---

### `src/pipeline/modality_dropout.py` — Test / debug du dropout

```bash
python src/pipeline/modality_dropout.py [OPTIONS]
```

| Flag | Type | Default | Choix | Description |
|---|---|---|---|---|
| `--modalities` | str+ | `rgb depth seg thermal` | `rgb` `depth` `seg` `thermal` | Modalités à inclure dans le dropout |
| `--p_drop` | float | `0.5` | `[0, 1]` | Probabilité de corrompre chaque modalité à chaque step |
| `--alpha_min` | float | `0.0` | `[0, 1]` | Alpha minimum en fin de curriculum (`0.0` = hard dropout, `1.0` = pas de corruption) |
| `--total_epochs` | int | `100` | — | Nombre d'epochs pour le curriculum schedule |
| `--corruption_types` | str+ | `gaussian blur occlusion` | `gaussian` `blur` `occlusion` | Types de corruption soft appliqués |
| `--epoch` | int | `0` | — | Epoch courante (affiche l'alpha correspondant) |
| `--img_size` | int | `64` | — | Taille spatiale des tenseurs de test |

**Exemples :**
```bash
# Tester le dropout à mi-curriculum avec seulement RGB et thermal
python src/pipeline/modality_dropout.py --modalities rgb thermal --epoch 50 --total_epochs 100

# Hard dropout agressif, seulement bruit gaussien
python src/pipeline/modality_dropout.py --p_drop 0.9 --alpha_min 0.0 --corruption_types gaussian

# Voir l'alpha à différentes epochs
python src/pipeline/modality_dropout.py --epoch 0 --total_epochs 200   # alpha=1.0
python src/pipeline/modality_dropout.py --epoch 100 --total_epochs 200  # alpha=0.5
python src/pipeline/modality_dropout.py --epoch 200 --total_epochs 200  # alpha=0.0
```

---

### `scripts/upload_thermal_hf.py` — Upload dataset sur HuggingFace

```bash
python scripts/upload_thermal_hf.py --repo <username>/<repo-name> [OPTIONS]
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--repo` | str | *(requis)* | Identifiant du repo HuggingFace (ex: `TomasAnderegg/libero_10_thermal`) |
| `--private` | flag | `False` | Rendre le dataset privé |
| `--path_in_repo` | str | `data/train` | Chemin dans le repo HuggingFace où déposer les fichiers |

**Exemples :**
```bash
# Upload privé
python scripts/upload_thermal_hf.py --repo TomasAnderegg/libero_10_thermal --private

# Upload public dans un sous-dossier custom
python scripts/upload_thermal_hf.py --repo TomasAnderegg/libero_10_thermal --path_in_repo data/parquet
```

> Le token HuggingFace est lu depuis `$HF_TOKEN`, puis `/scratch/izar/garate/huggingface_cache/token`, puis les credentials sauvegardés par `hf auth login`. Le token doit avoir la permission **Write**.

---

### `scripts/test_block2.py` / `scripts/test_pipeline.py` — Sanity checks

```bash
python scripts/test_block2.py [--fourm_model B|L|XL]
python scripts/test_pipeline.py [--fourm_model B|L|XL]
```

| Flag | Type | Default | Choix | Description |
|---|---|---|---|---|
| `--fourm_model` | str | `XL` | `B` `L` `XL` | Variante 4M-21 à tester (B=768d, L/XL=1024d) |

---

## Dependencies

| Package | Version | Notes |
|---|---|---|
| Python | 3.12 | Required by LeRobot 0.5.2+ |
| PyTorch | 2.7.0+cu128 | Required by LeRobot 0.5.2+ |
| LeRobot | 0.5.2 | Contains SmolVLA |
| fourm | 1.0.0 | 4M-21, installed `--no-deps` |
| ImageBind | 0.1.0 | Thermal encoder, installed `--no-deps` |

---

## References

1. Fei et al., *LIBERO-Plus: In-depth robustness analysis of VLA models*, arXiv:2510.13626, 2025.
2. Ma et al., *A survey on VLA models for embodied AI*, arXiv:2405.14093, 2026.
3. Bachmann et al., *4M-21: An any-to-any vision model*, arXiv:2406.09406, 2024.
4. Shukor et al., *SmolVLA: A VLA model for affordable and efficient robotics*, arXiv:2506.01844, 2025.
5. Guo et al., *On robustness of VLA against multi-modal perturbations*, arXiv:2510.00037, 2026.
6. Xiao et al., *ThermalGen*, NeurIPS 2025.
7. Girdhar et al., *ImageBind: One embedding space to bind them all*, arXiv:2305.05665, 2023.
8. Liu et al., *Improved baselines with visual instruction tuning (LLaVA-1.5)*, arXiv:2310.03744, 2024.
