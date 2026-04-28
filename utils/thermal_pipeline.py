import os

import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)


import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import torch
import torch.nn as nn
import torchvision.transforms.v2 as v2
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from third_party.ThermalGen.models.generative_models.sit_networks import sit_networks
from third_party.ThermalGen.models.generative_models.sit_networks.transport import create_transport, Sampler
import yaml
from huggingface_hub import PyTorchModelHubMixin

INPUT_DIR = "data/parquet"
OUTPUT_DIR = "data/parquet_thermal"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_COLUMN = "observation.images.image"
THERMAL_COLUMN = "observation.images.thermal"

class ThermalGenSIT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, model_config):
        
        super().__init__()
        self.model_config = model_config
        num_classes = 1000

        # --- Build SiT backbone (KL-VAE only)
        assert model_config['vae_model'] == "klvae", "Only klvae is supported"
        self.model = sit_networks.SiT_models[
            f"SiT-{model_config['arch']}/{model_config['patch_size']}"
        ](
            in_channels=4,
            num_classes=num_classes,
            injection_args=model_config['injection_args'],
            learn_sigma=False,
        )

        # Thermal + RGB VAEs
        self.thermal_vae = AutoencoderKL(**model_config['vae_config'])
        self.RGB_vae = AutoencoderKL.from_pretrained(
            f"stabilityai/sd-vae-ft-{model_config['vae']}"
        )

        # Transport & Sampler
        self.transport = create_transport(**model_config['transport_config'])
        self.sampler = Sampler(self.transport)

        # EMA copy for eval
        self.ema = self.model
        self.use_cfg = model_config['cfg_scale'] > 1.0
        self.thermal_normalizer = model_config.get('thermal_normalizer', None)
        self.RGB_normalizer = model_config.get('RGB_normalizer', None)

    @torch.no_grad()
    def forward(self, RGB, dataset_idx):
        # latent sampling size is tied to VAE downsample ratio (8 for KL-VAE)
        latent_size = RGB.shape[2] // 8, RGB.shape[3] // 8
        zs = torch.randn(RGB.shape[0], 4, *latent_size, device=RGB.device)

        # Encode RGB with KL-VAE
        x_RGB = self.RGB_vae.encode(RGB).latent_dist.sample()
        if self.RGB_normalizer is not None:
            x_RGB = x_RGB * self.RGB_normalizer

        # Conditional sampling
        ys = dataset_idx
        sample_fn = self.sampler.sample_ode()
        if self.use_cfg:
            zs = torch.cat([zs, zs], 0)
            y_null = torch.tensor([1000] * len(ys), device=RGB.device)
            ys = torch.cat([ys, y_null], 0)
            x_RGB = torch.cat([x_RGB, x_RGB], 0)
            model_eval = self.ema.forward_with_cfg
            kwargs = dict(y=ys, x_RGB=x_RGB, cfg_scale=self.model_config['cfg_scale'])
        else:
            model_eval = self.ema.forward
            kwargs = dict(y=ys, x_RGB=x_RGB)

        samples = sample_fn(zs, model_eval, **kwargs)[-1]
        if self.use_cfg:
            samples, _ = samples.chunk(2, dim=0)

        # Decode thermal latent
        if self.thermal_normalizer is not None:
            samples = samples / self.thermal_normalizer
        Pred_Thermal = self.thermal_vae.decode(samples).sample

        return Pred_Thermal[:, :, :RGB.shape[2], :RGB.shape[3]]

def decode_image(img_dict):
    """Decode {'bytes':..., 'path':...} ? PIL.Image"""
    return Image.open(BytesIO(img_dict["bytes"])).convert("RGB"), img_dict["path"]

def encode_image(img: Image.Image, path):
    """Encode PIL.Image {'bytes':..., 'path':...}"""
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return {
        "bytes": buffer.getvalue(),
        "path": "thermal"+path.split("_")[1],    }

def preprocess(img: Image.Image):
    tensor = eval_transform(img).unsqueeze(0).to(DEVICE)
    return tensor

def postprocess(tensor):
    tensor = tensor.squeeze().cpu().numpy()
    tensor = (tensor * 255).astype("uint8")
    return Image.fromarray(tensor)

BATCH_SIZE = 16


def generate_thermal_batch(imgs: list):
    tensors = torch.cat([preprocess(img) for img in imgs], dim=0).to(DEVICE)
    with torch.no_grad():
        dataset_idx = torch.ones(len(imgs), dtype=torch.long, device=DEVICE) * 1000
        pred = model(tensors, dataset_idx)
        pred = torch.clamp(pred * 0.5 + 0.5, 0, 1)
    return [postprocess(pred[i].unsqueeze(0)) for i in range(len(imgs))]


def process_shard(input_path, output_path):
    df = pd.read_parquet(input_path)
    rows = df[IMAGE_COLUMN].tolist()
    thermal_column = [None] * len(rows)

    for start in tqdm(range(0, len(rows), BATCH_SIZE), desc=os.path.basename(input_path)):
        batch_dicts = rows[start:start + BATCH_SIZE]
        imgs, paths = [], []
        indices = []
        for i, img_dict in enumerate(batch_dicts):
            try:
                img, path = decode_image(img_dict)
                imgs.append(img)
                paths.append(path)
                indices.append(start + i)
            except Exception as e:
                print(f"Failed decode at {start + i}: {e}")

        if not imgs:
            continue

        try:
            thermal_imgs = generate_thermal_batch(imgs)
            for idx, thermal_img, path in zip(indices, thermal_imgs, paths):
                thermal_column[idx] = encode_image(thermal_img, path)
        except Exception as e:
            print(f"Failed batch at {start}: {e}")

    df[THERMAL_COLUMN] = thermal_column

    df.to_parquet(output_path, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=int, default=0, help="Index of this job (0-indexed)")
    parser.add_argument("--n_jobs", type=int, default=1, help="Total number of parallel jobs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    BATCH_SIZE = args.batch_size

    print("Loading model...")
    model = ThermalGenSIT.from_pretrained("xjh19972/ThermalGen-XL-2").to(DEVICE)
    print("Model loaded successfully")

    eval_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    all_shards = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".parquet")])
    shards = all_shards[args.job_id::args.n_jobs]
    print(f"Job {args.job_id}/{args.n_jobs}: processing {len(shards)}/{len(all_shards)} shards")

    for shard in tqdm(shards, desc="Shards"):
        input_path = os.path.join(INPUT_DIR, shard)
        output_path = os.path.join(OUTPUT_DIR, shard)

        if os.path.exists(output_path):
            continue

        try:
            process_shard(input_path, output_path)
        except Exception as e:
            print(f"Failed shard {shard}: {e}")

    print("Thermal augmentation complete.")