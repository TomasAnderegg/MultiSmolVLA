FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
# Allow pip to install into system Python on Ubuntu 24.04 (PEP 668)
ENV PIP_BREAK_SYSTEM_PACKAGES=1
WORKDIR /workspace

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3-pip \
    git git-lfs curl wget ffmpeg \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libglu1-mesa-dev libglfw3-dev libglew-dev \
    libosmesa6-dev patchelf \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

# ── PyTorch 2.7.0 for CUDA 12.8 ──────────────────────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.7.0 \
    torchvision==0.22.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# ── Clone MultiSmolVLA ────────────────────────────────────────────────────────
RUN git clone --recurse-submodules https://github.com/TomasAnderegg/MultiSmolVLA.git /workspace/MultiSmolVLA
WORKDIR /workspace/MultiSmolVLA

# ── Core Python deps ──────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    huggingface_hub>=1.0.0 \
    transformers>=4.40.0 \
    diffusers>=0.37.0 \
    timm>=1.0.0 \
    torchdiffeq>=0.2.3 \
    einops>=0.7.0 \
    ftfy>=6.1.0 \
    numpy==1.26.4 \
    Pillow>=10.0.0 \
    wandb \
    scikit-learn \
    matplotlib \
    pyarrow \
    tqdm \
    boto3 \
    webdataset \
    albumentations \
    braceexpand \
    pandas \
    datasets \
    "torchmetrics[image,multimodal]" \
    num2words \
    opencv-python-headless

# ── LeRobot (SmolVLA) ─────────────────────────────────────────────────────────
RUN pip install --no-cache-dir -e third_party/lerobot

# ── 4M-21 ─────────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir --no-deps -e third_party/ml-4m

# ── ImageBind ─────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir --no-deps -e third_party/ImageBind && \
    pip install --no-cache-dir pytorchvideo iopath types-regex && \
    sed -i 's|import torchvision.transforms.functional_tensor as F_t|from torchvision.transforms import functional as F_t|g' \
        /usr/local/lib/python3.12/dist-packages/pytorchvideo/transforms/augmentations.py

# ── ThermalGen ────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir --no-deps -e third_party/ThermalGen 2>/dev/null || true

# ── Pin cmake 3.x (lerobot installs cmake 4.x which breaks egl_probe build) ──
RUN pip install --no-cache-dir "cmake>=3.29,<4.0"

# ── LIBERO (simulation environment) ──────────────────────────────────────────
RUN git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git /workspace/LIBERO && \
    pip install --no-cache-dir -e /workspace/LIBERO && \
    pip install --no-cache-dir \
        robosuite==1.4.0 \
        bddl \
        easydict==1.9 \
        robomimic==0.2.0 \
        "gym==0.25.2" \
        mujoco

# ── Fix lerobot GR00T Python 3.12 issue ──────────────────────────────────────
RUN python -c "\
import re, pathlib; \
f = pathlib.Path('third_party/lerobot/src/lerobot/policies/groot/groot_n1.py'); \
content = f.read_text() if f.exists() else ''; \
content = content.replace('field(init=False, metadata=', 'field(init=False, default=None, metadata='); \
f.write_text(content) if f.exists() else None"

# ── Fix robosuite macros ───────────────────────────────────────────────────────
RUN python -c "\
import shutil, robosuite, pathlib; \
site = pathlib.Path(robosuite.__file__).parent; \
src = site / 'macros.py'; \
dst = site / 'macros_private.py'; \
shutil.copy(src, dst) if src.exists() and not dst.exists() else None"

# ── Environment variables ─────────────────────────────────────────────────────
ENV PYTHONPATH="/workspace/LIBERO:/workspace/MultiSmolVLA/third_party/lerobot/src:/workspace/MultiSmolVLA"
ENV HF_HOME="/workspace/hf_cache"
ENV HUGGINGFACE_HUB_CACHE="/workspace/hf_cache/hub"
ENV ROBOSUITE_LOG_FILE="/tmp/robosuite.log"

# ── Entrypoint ────────────────────────────────────────────────────────────────
WORKDIR /workspace/MultiSmolVLA
CMD ["/bin/bash"]
