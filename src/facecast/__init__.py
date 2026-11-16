"""facecast: face super-resolution (16x16 -> 128x128) with generators trained on CelebA.

    from facecast import FaceUpscaler

    upscaler = FaceUpscaler.from_pretrained("gan")
    face = upscaler.upscale("photo.jpg")
    face = upscaler.upscale("group.jpg", roi=(410, 120, 450, 160))
"""

from .models import Generator
from .preprocess import prepare, square_roi
from .upscaler import FaceUpscaler
from .weights import DEFAULT_MODEL, list_models, load_generator

__version__ = "0.1.0"
__all__ = ["FaceUpscaler", "Generator", "prepare", "square_roi", "list_models", "load_generator", "DEFAULT_MODEL"]
