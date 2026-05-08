import sys
sys.path.append("src")
sys.path.append("third_party/ImageBind")

import torch
from PIL import Image
from torchvision.transforms import v2
from pipeline.block1 import Block1

device = "cuda" if torch.cuda.is_available() else "cpu"

# Charger le RGB original
img = Image.open("third_party/ThermalGen/pic_og.png").convert("RGB")
transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, scale=True),  # → [0, 1]
])
rgb = transform(img).unsqueeze(0).to(device)   # (1, 3, 224, 224)

# Depth et seg simulés (zeros) — on n'a pas de vrais capteurs
depth = torch.zeros(1, 1, 224, 224, device=device)
seg   = torch.zeros(1, 1, 224, 224, device=device)

inputs = {"rgb": rgb, "depth": depth, "seg": seg}

# Instancier Block1 et passer en mode inférence
block1 = Block1(device=device)
block1.eval()

with torch.no_grad():
    outputs = block1(inputs, epoch=0)

print("=== Block1 output ===")
print(f"rgb     : {outputs['rgb'].shape}")      # (1, 3, 224, 224)
print(f"depth   : {outputs['depth'].shape}")    # (1, 1, 224, 224)
print(f"seg     : {outputs['seg'].shape}")      # (1, 1, 224, 224)
print(f"thermal : {outputs['thermal'].shape}")  # (1, 1024)
print(f"thermal norm : {outputs['thermal'].norm():.4f}")  # ~1.0
