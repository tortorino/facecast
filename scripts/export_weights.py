"""Exports the generators trained by facecast.ipynb (models/*.pt) into the package (src/facecast/assets).

Only the weights are exported, as .safetensors files. Run it again after retraining:

    uv run python scripts/export_weights.py
"""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
# name in the package -> checkpoint written by the notebook
CHECKPOINTS = {
    "l1": "generator_l1.pt",
    "gan": "generator_gan.pt",
    "gan_unet": "generator_gan_unet.pt",
    "gan_unet_slow": "generator_gan_unet_slow.pt",
}
# The architecture used in the notebook (scale, num_blocks, num_feats)
CONFIG = {"scale": 8, "num_blocks": 16, "num_feats": 64}

assets = ROOT / "src" / "facecast" / "assets"
assets.mkdir(parents=True, exist_ok=True)
for name, filename in CHECKPOINTS.items():
    state = torch.load(ROOT / "models" / filename, map_location="cpu")["model"]
    state = {k: v.contiguous() for k, v in state.items()}
    target = assets / (name + ".safetensors")
    save_file(state, str(target), metadata={"config": json.dumps(CONFIG)})
    target.chmod(0o644)
    print("%-14s <- models/%s (%d tensors)" % (name, filename, len(state)))
