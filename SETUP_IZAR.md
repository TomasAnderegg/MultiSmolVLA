# Setup on IZAR (EPFL cluster)

This guide covers the full setup from scratch on the IZAR cluster. It documents every pitfall encountered during installation.

---

## 1. Create the conda environment

Create the env on scratch (home has limited quota):

```bash
conda create -p /scratch/izar/$USER/envs/multismolvla python=3.12 -y
conda activate /scratch/izar/$USER/envs/multismolvla
```

> The env appears without a name in `conda info --envs`. Always activate it with the full path.

---

## 2. Install PyTorch

**Check your driver version first** (on a compute node, not the login node):

```bash
srun --partition=gpu --gres=gpu:1 --time=00:05:00 --pty bash -c "nvidia-smi"
```

Then install the matching PyTorch build:

| Driver version | CUDA | Command |
|---|---|---|
| ≥ 570 | 12.8 | `--index-url https://download.pytorch.org/whl/cu128` with `+cu128` |
| ≥ 525 | 12.6 | `--index-url https://download.pytorch.org/whl/cu126` with `+cu126` |

Example for CUDA 12.6 (recommended for IZAR):

```bash
pip install torch==2.7.0+cu126 torchvision==0.22.0+cu126 torchaudio==2.7.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126
```

> If you get `RuntimeError: CUDA error: no kernel image is available`, your PyTorch build does not match the GPU driver. Reinstall with a lower CUDA version.

---

## 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Install source dependencies

The submodules should already be present after `git clone --recurse-submodules`. Install them as editable packages:

```bash
# LeRobot (SmolVLA)
cd third_party/lerobot && pip install -e . && cd ../..

# 4M-21 — no-deps to avoid version conflicts with the rest of the env
pip install --no-deps -e third_party/ml-4m

# ImageBind — no-deps for the same reason
pip install --no-deps -e third_party/ImageBind
```

> `--no-deps` installs only the package code, skipping its declared dependencies (which pin old versions of torch/torchvision incompatible with the rest of the env).

---

## 5. Apply the pytorchvideo patch

`pytorchvideo` uses `torchvision.transforms.functional_tensor` which was removed in torchvision ≥ 0.16. Patch it manually:

```bash
sed -i 's/import torchvision.transforms.functional_tensor as F_t/import torchvision.transforms.functional as F_t/' \
    $(python -c "import site; print(site.getsitepackages()[0])")/pytorchvideo/transforms/augmentations.py
```

---

## 6. Apply the lerobot patch (Python 3.12)

Python 3.12 enforces stricter dataclass rules that break one lerobot file. Edit `third_party/lerobot/src/lerobot/policies/groot/groot_n1.py` and add `default=None` to the four `init=False` fields in `GR00TN15Config` (~line 177):

```python
# Before
backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})

# After
backbone_cfg: dict = field(init=False, default=None, metadata={"help": "Backbone configuration."})
```

Apply the same change to `action_head_cfg`, `action_horizon`, and `action_dim`.

---

## 7. Run the sanity check

Create the logs directory and submit the SLURM job:

```bash
mkdir -p logs
sbatch scripts/run_test_block2.sh
```

Monitor the output:

```bash
# Watch job status
watch squeue -u $USER

# Tail the log once the job starts (ST = R)
tail -f logs/test_block2_<JOBID>.out
```

Expected output at the end:

```
Success: actions shape = torch.Size([1, 64, 7])
✅ Block2 pipeline test completed
```

---

## Known dependency conflicts (non-blocking)

When installing torchvision/torchaudio, pip will warn about conflicts with `fourm` and `imagebind`. These are safe to ignore because both were installed with `--no-deps`:

| Package | Declared requirement | Installed | Status |
|---|---|---|---|
| fourm | `diffusers==0.20.0` | 0.37.x | Works |
| fourm | `huggingface_hub<=0.24.0` | 1.x | Works |
| fourm | `numpy<2.0.0` | 2.x | Works |
| imagebind | `pytorchvideo @ git+...` | pip version | Works after patch |
