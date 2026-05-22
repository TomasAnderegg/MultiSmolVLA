FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    git git-lfs curl wget ffmpeg \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libglu1-mesa-dev libglfw3-dev libglew-dev \
    libosmesa6-dev patchelf \
    && rm -rf /var/lib/apt/lists/*

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
    tqdm

# ── LeRobot (SmolVLA) ─────────────────────────────────────────────────────────
RUN pip install --no-cache-dir -e third_party/lerobot

# ── 4M-21 ─────────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir --no-deps -e third_party/ml-4m

# ── ImageBind ─────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir --no-deps -e third_party/ImageBind && \
    pip install --no-cache-dir pytorchvideo iopath types-regex

# ── ThermalGen ────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir --no-deps -e third_party/ThermalGen 2>/dev/null || true

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
import shutil, pathlib; \
src = pathlib.Path('/opt/conda/lib/python3.12/site-packages/robosuite/macros.py'); \
dst = pathlib.Path('/opt/conda/lib/python3.12/site-packages/robosuite/macros_private.py'); \
shutil.copy(src, dst) if src.exists() and not dst.exists() else None"

# ── Environment variables ─────────────────────────────────────────────────────
ENV PYTHONPATH="/workspace/LIBERO:/workspace/MultiSmolVLA/third_party/lerobot/src:/workspace/MultiSmolVLA"
ENV HF_HOME="/workspace/hf_cache"
ENV HUGGINGFACE_HUB_CACHE="/workspace/hf_cache/hub"
ENV ROBOSUITE_LOG_FILE="/tmp/robosuite.log"

# ── Entrypoint ────────────────────────────────────────────────────────────────
WORKDIR /workspace/MultiSmolVLA
CMD ["/bin/bash"]
