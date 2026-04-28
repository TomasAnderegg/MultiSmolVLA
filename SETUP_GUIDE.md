# Environment Setup Guide — MultiSmolVLA (Block 2)

This guide sets up the working environment for **Block 2** (SmolVLA + 4M-21 action pipeline).
Pass this file to Claude along with the repo to get help debugging install issues.

---

## Context

The project uses a dedicated conda environment called `lerobot` (Python 3.12.13, CUDA 12.8).
Two key packages are installed as local editable installs from the repo's submodules:

| Package | Source | How installed |
|---|---|---|
| `lerobot` 0.5.2 | `third_party/lerobot/` (submodule) | `pip install -e .` |
| `fourm` 1.0.0 | `third_party/ml_4m/` (symlink → `ml-4m/`) | `pip install --no-deps -e .` |

> **ImageBind is NOT needed for Block 2.** Skip it entirely.

---

## Step-by-step install

### 1. Clone the repo with submodules

```bash
git clone --recurse-submodules <repo_url>
cd MultiSmolVLA
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### 2. Create the conda environment

```bash
conda create -n lerobot python=3.12 -y
conda activate lerobot
```

### 3. Install PyTorch (CUDA 12.8)

```bash
pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install lerobot from the submodule

```bash
cd third_party/lerobot
pip install -e .
cd ../..
```

> This installs lerobot 0.5.2 which **already has the Python 3.12 patch** applied
> (`groot_n1.py` fields have `default=None`). No manual patch needed.

### 5. Create the ml_4m symlink and install 4M

The `ml-4m` submodule directory name has a hyphen, which Python can't import.
A symlink with an underscore is required:

```bash
# Only needed if the symlink doesn't exist yet
ln -s ml-4m third_party/ml_4m
```

Then install:

```bash
pip install --no-deps -e third_party/ml_4m
```

The `--no-deps` flag is important — 4M's declared deps conflict with lerobot's versions.
The packages already installed by lerobot are sufficient.

### 6. Verify

```bash
conda activate lerobot
python -c "import lerobot; import fourm; print('OK')"
```

Expected output: `OK`

---

## Key gotchas

- **Python must be 3.12**, not 3.13. LeRobot 0.5.x requires `>=3.12` but some deps break on 3.13.
- **`ml_4m` symlink must exist** before the editable install. If you get `ModuleNotFoundError: No module named 'fourm'`, check the symlink: `ls -la third_party/` should show `ml_4m -> ml-4m`.
- **`--no-deps` on 4M is mandatory.** Without it, pip will downgrade `diffusers` and break lerobot.
- **lerobot must be installed from `third_party/lerobot/`** (the submodule), not from a separate clone, because the submodule already has the groot dataclass patch needed by Python 3.12.
- The active Python in this project is from the `lerobot` conda env. Always run `conda activate lerobot` before any script.

---

## Installed package versions (reference)

The working environment has:

```
torch              2.10.0+cu128
torchvision        0.25.0
torchaudio         2.10.0
lerobot            0.5.2   (editable: third_party/lerobot)
fourm              1.0.0   (editable: third_party/ml_4m)
transformers       5.6.2
diffusers          0.35.2
einops             0.8.2
timm               1.0.26
gymnasium          1.2.3
gym-aloha          0.1.3
accelerate         1.13.0
wandb              0.24.2
datasets           4.1.1
```
