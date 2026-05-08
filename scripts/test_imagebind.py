import sys
sys.path.append("third_party/ImageBind")
sys.path.append("src")

import torch
from PIL import Image
from torchvision.transforms import v2
from pipeline.imagebind_encoder import ImageBindThermalEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"

# Charger l'image thermique générée par ThermalGen
img = Image.open("third_party/ThermalGen/test_thermal_uncond.png").convert("RGB")
transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, scale=True),  # → [0, 1]
])
thermal = transform(img).unsqueeze(0)  # (1, 3, 224, 224)

# Test
encoder = ImageBindThermalEncoder(device=device)
feat = encoder(thermal)

print(f"Shape : {feat.shape}")    # attendu : torch.Size([1, 1024])
print(f"Norm  : {feat.norm():.4f}")  # attendu : ~1.0

# Charger le RGB original
img_rgb = Image.open("third_party/ThermalGen/pic_og.png").convert("RGB")
rgb = transform(img_rgb).unsqueeze(0)  # même transform, (1, 3, 224, 224)
feat_rgb = encoder(rgb)

# Similarité cosinus
cos = torch.nn.functional.cosine_similarity(feat, feat_rgb)
print(f"Similarité thermique/RGB : {cos.item():.4f}")
