"""The trained generators that are bundled with the package."""

import json
from importlib import resources
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file

from .models import Generator

# name -> short description (all of them upscale 16x16 -> 128x128)
MODELS = {
    "l1": "Stage 1, pixel (L1) loss only. The most accurate one (best PSNR / SSIM), but smooth.",
    "gan": "Stage 2, perceptual + adversarial loss (SRGAN, 10 epochs). Sharper, more realistic faces.",
    "gan_unet": "Stage 2 with a U-Net discriminator and a relativistic loss (4 epochs).",
    "gan_unet_slow": "As gan_unet, with a 4 times lower learning rate of the discriminator.",
}

DEFAULT_MODEL = "gan"


def list_models():
    """Returns {name: description} of the bundled generators."""
    return dict(MODELS)


def weights_path(name):
    """Path of the weight file of a bundled generator."""
    if name not in MODELS:
        raise ValueError("Unknown model %r, choose one of: %s" % (name, ", ".join(MODELS)))
    return resources.files("facecast") / "assets" / (name + ".safetensors")


def load_generator(name_or_path=DEFAULT_MODEL, device="cpu"):
    """Loads a bundled generator by its name, or any exported .safetensors file by its path."""
    path = Path(str(name_or_path))
    if path.suffix == ".safetensors":
        if not path.is_file():
            raise FileNotFoundError("Weight file not found: %s" % path)
        return _load(path, device)
    with resources.as_file(weights_path(name_or_path)) as file:
        return _load(file, device)


def _load(file, device):
    # The architecture is stored next to the weights
    with safe_open(str(file), framework="pt") as f:
        config = json.loads((f.metadata() or {}).get("config", "{}"))
    generator = Generator(**config)
    generator.load_state_dict(load_file(str(file), device="cpu"))
    return generator.to(device).eval()
