import os
import sys

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_fourm_root = os.path.join(_repo_root, "third_party", "ml-4m")
if _fourm_root not in sys.path:
    sys.path.insert(0, _fourm_root)

from fourm.models.fm import FM
from fourm.vq.vqvae import DiVAE, VQ
from fourm.models.generate import GenerationSampler

# ImageNet normalization used by 4M for continuous rgb@224 inputs
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)

NUM_PATCHES = 196
PATCH_GRID  = 14   # 224 / 16


def _imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) float [0,1] → ImageNet-normalized float."""
    mean = torch.tensor(_IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std  = torch.tensor(_IMAGENET_STD,  device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


class FourMImageProcessor(nn.Module):
    """
    Wraps 4M-21 as an image-to-image processor with MaskGIT iterative decoding.

    Phase 1: RGB only  → 4M encode → MaskGIT decode → DiVAE → RGB*
    Phase 2: RGB+depth → 4M encode (rgb@224 + tok_depth@224) → MaskGIT → DiVAE → RGB*

    MaskGIT iterative decoding schedule (n_maskgit_steps=4, linear):
      Step 1: predict all 196 tokens, keep top 49 most confident
      Step 2: 49 known as context, predict 147, keep top 98 total
      Step 3: keep top 147
      Step 4: reveal remaining 49 → fully determined
    """

    def __init__(
        self,
        fourm_checkpoint:         str         = "EPFL-VILAB/4M-21_XL",
        divae_checkpoint:         str         = "EPFL-VILAB/4M_tokenizers_rgb_16k_224-448",
        depth_tok_checkpoint:     str         = "EPFL-VILAB/4M_tokenizers_depth_8k_224-448",
        seg_tok_checkpoint:       str         = "EPFL-VILAB/4M_tokenizers_semseg_4k_224-448",
        lora_checkpoint:          str | None  = None,
        divae_steps:              int         = 50,
        tok_temperature:          float       = 3.0,
        cfg_scale:                float       = 1.0,
        use_depth:                bool        = False,
        use_seg:                  bool        = False,
        n_maskgit_steps:          int         = 4,
        use_tok_encoder:          bool        = False,
        device:                   str         = "cuda",
    ):
        super().__init__()
        self.divae_steps      = divae_steps
        self.tok_temperature  = tok_temperature
        self.cfg_scale        = cfg_scale
        self.use_depth        = use_depth
        self.use_seg          = use_seg
        self.n_maskgit_steps  = n_maskgit_steps
        self.use_tok_encoder  = use_tok_encoder

        # ── 4M backbone ───────────────────────────────────────────────────────
        print(f"[FourMImageProcessor] Loading 4M from {fourm_checkpoint} ...")
        try:
            self.fm = FM.from_pretrained(fourm_checkpoint, local_files_only=True)
        except Exception:
            self.fm = FM.from_pretrained(fourm_checkpoint, local_files_only=False)
        self.fm.to(device).eval()
        for p in self.fm.parameters():
            p.requires_grad = False

        # ── Optional LoRA weights (fine-tuned on LIBERO) ──────────────────────
        if lora_checkpoint is not None:
            from fourm.models.lora_utils import inject_trainable_LoRA, get_LoRA_module_names
            ckpt = torch.load(lora_checkpoint, map_location=device)
            rank        = ckpt.get("lora_rank",   8)
            scale       = ckpt.get("lora_scale",  1.0)
            lora_target = ckpt.get("lora_target", "attention")
            lora_mods   = get_LoRA_module_names(lora_target)
            inject_trainable_LoRA(self.fm, rank=rank, scale=scale, target_replace_modules=lora_mods)
            self.fm.to(device)  # LoRA layers created on CPU — move to GPU before loading weights
            missing, unexpected = self.fm.load_state_dict(ckpt["lora_state_dict"], strict=False)
            for p in self.fm.parameters():
                p.requires_grad = False
            print(f"[FourMImageProcessor] LoRA loaded from {lora_checkpoint}  "
                  f"(rank={rank}, target={lora_target}, "
                  f"missing={len(missing)}, unexpected={len(unexpected)})")

        self.sampler = GenerationSampler(self.fm)
        print("[FourMImageProcessor] 4M loaded")

        # ── RGB DiVAE (decoder) ───────────────────────────────────────────────
        print(f"[FourMImageProcessor] Loading RGB DiVAE from {divae_checkpoint} ...")
        try:
            self.divae = DiVAE.from_pretrained(divae_checkpoint, local_files_only=True)
        except Exception:
            self.divae = DiVAE.from_pretrained(divae_checkpoint, local_files_only=False)
        self.divae.to(device).eval()
        for p in self.divae.parameters():
            p.requires_grad = False
        print("[FourMImageProcessor] RGB DiVAE loaded")

        # ── Depth tokenizer ───────────────────────────────────────────────────
        self.depth_tok = None
        if use_depth:
            print(f"[FourMImageProcessor] Loading depth tokenizer from {depth_tok_checkpoint} ...")
            try:
                self.depth_tok = VQ.from_pretrained(depth_tok_checkpoint, local_files_only=True)
            except Exception:
                try:
                    self.depth_tok = VQ.from_pretrained(depth_tok_checkpoint, local_files_only=False)
                except Exception:
                    try:
                        self.depth_tok = DiVAE.from_pretrained(depth_tok_checkpoint, local_files_only=True)
                    except Exception:
                        self.depth_tok = DiVAE.from_pretrained(depth_tok_checkpoint, local_files_only=False)
            self.depth_tok.to(device).eval()
            for p in self.depth_tok.parameters():
                p.requires_grad = False
            print("[FourMImageProcessor] Depth tokenizer loaded")

        # ── Seg tokenizer (COCO semseg, 4k codebook, VQ) ─────────────────────
        self.seg_tok = None
        if use_seg:
            print(f"[FourMImageProcessor] Loading seg tokenizer from {seg_tok_checkpoint} ...")
            try:
                self.seg_tok = VQ.from_pretrained(seg_tok_checkpoint, local_files_only=True)
            except Exception:
                self.seg_tok = VQ.from_pretrained(seg_tok_checkpoint, local_files_only=False)
            self.seg_tok.to(device).eval()
            for p in self.seg_tok.parameters():
                p.requires_grad = False
            print("[FourMImageProcessor] Seg tokenizer loaded")

        n_inputs = 1 + int(use_depth) + int(use_seg)
        enc_mode  = "tok_rgb@224 (DiVAE tokenize)" if use_tok_encoder else "rgb@224 (continuous)"
        print(f"[FourMImageProcessor] Ready — inputs={n_inputs}  encoder={enc_mode}  "
              f"(maskgit_steps={n_maskgit_steps}, divae_steps={divae_steps}, temperature={tok_temperature})")

    @property
    def device(self) -> torch.device:
        return next(self.fm.parameters()).device

    @torch.no_grad()
    def divae_roundtrip(self, rgb: torch.Tensor,
                        ddim_steps: int = 50,
                        input_size: int = 448,
                        bypass_vq: bool = True) -> torch.Tensor:
        """Pure DiVAE encode → decode, bypassing 4M entirely.

        Quality levers:
          1. bypass_vq=True  → skip discrete VQ bottleneck, feed continuous latent directly
                               to diffusion decoder (no codebook snapping → sharper output)
          2. DDIM scheduler (50 steps ≈ DDPM 1000 steps quality, 20× faster)
          3. 448×448 input → 28×28 patches (4× more spatial detail than 224 → 14×14)
        """
        from fourm.vq.scheduling import DDIMScheduler as FourmDDIM

        device = self.device
        rgb = rgb.to(device)

        if rgb.shape[-2:] != (input_size, input_size):
            rgb = TF.resize(rgb, [input_size, input_size],
                            interpolation=TF.InterpolationMode.BILINEAR, antialias=True)

        rgb_m1_1 = rgb * 2.0 - 1.0   # [0,1] → [-1,1]

        # Mirror the DiVAE training scheduler exactly — mismatches cause residual noise
        ddim = FourmDDIM(
            num_train_timesteps=self.divae.num_train_timesteps,
            beta_schedule=self.divae.beta_schedule,
            clip_sample=self.divae.clip_sample,              # False for DiVAE
            prediction_type=self.divae.prediction_type,      # v_prediction
            thresholding=self.divae.thresholding,            # True — dynamic thresholding (Imagen)
            zero_terminal_snr=self.divae.zero_terminal_snr,  # True
        )

        if bypass_vq:
            # Continuous path: encoder → quant_proj → diffusion decoder (no VQ snap)
            # The decoder conditioning shape is identical whether we pass quant or h,
            # so this is a drop-in replacement that removes all quantization error.
            x_prep = self.divae.prepare_input(rgb_m1_1)
            h = self.divae.encoder(x_prep)
            h = self.divae.quant_proj(h)   # (B, D_Q, H_Q, W_Q) continuous latent
            pipeline = self.divae._get_pipeline(ddim)
            rgb_out = pipeline(
                h,
                timesteps=ddim_steps,
                image_size=input_size,
                scheduler_timesteps_mode='trailing',
                verbose=False,
            )
        else:
            rgb_out = self.divae.autoencode(
                rgb_m1_1,
                timesteps=ddim_steps,
                scheduler=ddim,
                scheduler_timesteps_mode='trailing',
                verbose=False,
            )

        rgb_out = (rgb_out.clamp(-1.0, 1.0) + 1.0) / 2.0

        if getattr(self, '_debug_save_dir', None) and getattr(self, '_debug_counter', 0) < 5:
            self._save_debug_pair(rgb, rgb_out)
        return rgb_out.to(dtype=rgb.dtype)

    @torch.no_grad()
    def process(self,
                rgb:   torch.Tensor,
                depth: torch.Tensor | None = None,
                seg:   torch.Tensor | None = None) -> torch.Tensor:
        """
        rgb   : (B, 3, H, W) float [0, 1]
        depth : (B, 1, H, W) float [0, 1]  — optional
        seg   : (B, 1, H, W) float/int      — optional (instance IDs or normalized)
        returns (B, 3, H_out, W_out) float [0, 1]  — reconstructed RGB
        """
        B      = rgb.shape[0]
        device = self.device

        rgb = rgb.to(device)
        if rgb.shape[-2:] != (224, 224):
            rgb = TF.resize(rgb, [224, 224], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)

        # ── 1. RGB tokenization (always needed for passthrough / debug) ────────
        rgb_m1_1 = rgb * 2.0 - 1.0
        rgb_token_ids = self.divae.tokenize(rgb_m1_1)   # (B, 14, 14) long

        # ── Passthrough: skip 4M, decode original DiVAE tokens directly ───────
        if getattr(self, '_rgb_passthrough', False):
            current_tokens = rgb_token_ids
        else:
            # ── 2. Build mod dict for GenerationSampler ───────────────────────
            # Encoder modalities: input_mask=False (visible), target_mask=True (not a decoder target)
            # Decoder target RGB: input_mask=True (hidden), target_mask=False (all 196 to generate)
            gen_mod_dict = {}

            if self.use_depth and depth is not None and self.depth_tok is not None:
                depth = depth.to(device)
                if depth.shape[-2:] != (224, 224):
                    depth = TF.resize(depth, [224, 224],
                                      interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
                depth_tokens = self.depth_tok.tokenize(depth)   # (B, 14, 14)
                gen_mod_dict["tok_depth@224"] = {
                    "tensor":                 depth_tokens.reshape(B, -1),
                    "input_mask":             torch.zeros(B, NUM_PATCHES, dtype=torch.bool, device=device),
                    "target_mask":            torch.ones( B, NUM_PATCHES, dtype=torch.bool, device=device),
                    "decoder_attention_mask": torch.zeros(B, NUM_PATCHES, dtype=torch.bool, device=device),
                }
                if getattr(self, '_debug_save_dir', None) and getattr(self, '_debug_counter', 0) < 5:
                    print(f"[fourm_debug] depth tokens unique={depth_tokens.unique().numel()}/8192"
                          f"  depth range=[{depth.min():.3f},{depth.max():.3f}]", flush=True)

            if self.use_seg and seg is not None and self.seg_tok is not None:
                seg = seg.to(device)
                if seg.shape[-2:] != (224, 224):
                    seg = TF.resize(seg, [224, 224],
                                    interpolation=TF.InterpolationMode.NEAREST)
                if seg.dtype in (torch.float32, torch.float16):
                    n_cls = self.seg_tok.cls_emb.num_embeddings
                    seg = (seg.squeeze(1) * (n_cls - 1)).round().long().clamp(0, n_cls - 1)
                else:
                    n_cls = self.seg_tok.cls_emb.num_embeddings
                    seg = seg.squeeze(1).long().clamp(0, n_cls - 1)
                seg_tokens = self.seg_tok.tokenize(seg)   # (B, 14, 14)
                gen_mod_dict["tok_semseg@224"] = {
                    "tensor":                 seg_tokens.reshape(B, -1),
                    "input_mask":             torch.zeros(B, NUM_PATCHES, dtype=torch.bool, device=device),
                    "target_mask":            torch.ones( B, NUM_PATCHES, dtype=torch.bool, device=device),
                    "decoder_attention_mask": torch.zeros(B, NUM_PATCHES, dtype=torch.bool, device=device),
                }
                if getattr(self, '_debug_save_dir', None) and getattr(self, '_debug_counter', 0) < 5:
                    print(f"[fourm_debug] seg tokens unique={seg_tokens.unique().numel()}/4096"
                          f"  seg range=[{seg.min():.3f},{seg.max():.3f}]", flush=True)

            # RGB decoder target: all 196 tokens to generate
            gen_mod_dict["tok_rgb@224"] = {
                "tensor":                 torch.zeros(B, NUM_PATCHES, dtype=torch.long,  device=device),
                "input_mask":             torch.ones( B, NUM_PATCHES, dtype=torch.bool,  device=device),
                "target_mask":            torch.zeros(B, NUM_PATCHES, dtype=torch.bool,  device=device),
                "decoder_attention_mask": torch.zeros(B, NUM_PATCHES, dtype=torch.bool,  device=device),
            }

            # ── 3. MaskGIT via GenerationSampler ──────────────────────────────
            tokens_per_step, already = [], 0
            for s in range(self.n_maskgit_steps):
                target = int(NUM_PATCHES * (s + 1) / self.n_maskgit_steps)
                tokens_per_step.append(max(1, target - already))
                already = target

            cond_mods = [k for k in gen_mod_dict if k != "tok_rgb@224"]

            for step_idx, num_select in enumerate(tokens_per_step):
                if self.cfg_scale > 1.0:
                    self.sampler.guided_maskgit_step_batched(
                        gen_mod_dict, "tok_rgb@224",
                        num_select=num_select,
                        temperature=self.tok_temperature,
                        top_k=0, top_p=1.0,
                        conditioning=cond_mods,
                        guidance_scale=self.cfg_scale,
                    )
                else:
                    self.sampler.maskgit_step_batched(
                        gen_mod_dict, "tok_rgb@224",
                        num_select=num_select,
                        temperature=self.tok_temperature,
                        top_k=0, top_p=1.0,
                    )
                if getattr(self, '_debug_save_dir', None) and getattr(self, '_debug_counter', 0) < 5:
                    tok = gen_mod_dict["tok_rgb@224"]["tensor"]
                    print(f"[fourm_debug] maskgit step={step_idx+1}/{self.n_maskgit_steps}  "
                          f"token_ids unique={tok.unique().numel()}", flush=True)

            current_tokens = gen_mod_dict["tok_rgb@224"]["tensor"].reshape(B, PATCH_GRID, PATCH_GRID)

        # ── 4. DiVAE decode: (B, 14, 14) token ids → (B, 3, 224, 224) pixels ──
        from fourm.vq.scheduling import DDIMScheduler as FourmDDIM
        ddim = FourmDDIM(
            num_train_timesteps=self.divae.num_train_timesteps,
            beta_schedule=self.divae.beta_schedule,
            clip_sample=self.divae.clip_sample,
            prediction_type=self.divae.prediction_type,
            thresholding=self.divae.thresholding,
            zero_terminal_snr=self.divae.zero_terminal_snr,
        )
        rgb_out = self.divae.decode_tokens(
            current_tokens,
            timesteps=self.divae_steps,
            scheduler=ddim,
            scheduler_timesteps_mode='trailing',
        )
        rgb_out = (rgb_out.clamp(-1.0, 1.0) + 1.0) / 2.0

        if getattr(self, '_debug_save_dir', None) and getattr(self, '_debug_counter', 0) < 5:
            self._save_debug_pair(rgb, rgb_out, depth=depth, seg=seg)

        return rgb_out.to(dtype=rgb.dtype)

    def _save_debug_pair(self,
                         rgb_in:  torch.Tensor,
                         rgb_out: torch.Tensor,
                         depth:   torch.Tensor | None = None,
                         seg:     torch.Tensor | None = None) -> None:
        import torchvision
        os.makedirs(self._debug_save_dir, exist_ok=True)

        def to_rgb_224(t: torch.Tensor) -> torch.Tensor:
            """(B,C,H,W) → (1,3,224,224) cpu, values in [0,1]."""
            t = t[:1].cpu().float()
            if t.shape[1] == 1:
                t = t.expand(-1, 3, -1, -1)  # grayscale → 3-channel
            t = t - t.min()
            if t.max() > 0:
                t = t / t.max()
            return TF.resize(t, [224, 224])

        tiles = [to_rgb_224(rgb_in), to_rgb_224(rgb_out)]
        labels = ["rgb_in", "rgb_out"]
        if depth is not None:
            tiles.append(to_rgb_224(depth))
            labels.append("depth")
        if seg is not None:
            tiles.append(to_rgb_224(seg.float()))
            labels.append("seg")

        grid = torchvision.utils.make_grid(torch.cat(tiles, dim=0), nrow=len(tiles))
        path = os.path.join(self._debug_save_dir, f"recon_{self._debug_counter:03d}.png")
        torchvision.utils.save_image(grid, path)
        print(f"[fourm_debug] saved {path}  [{' | '.join(labels)}]  "
              f"out=[{rgb_out.min():.3f},{rgb_out.max():.3f}]", flush=True)
        self._debug_counter += 1
