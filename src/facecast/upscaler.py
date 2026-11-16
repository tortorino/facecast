"""The high-level interface: load a generator and upscale faces with it."""

import os

import numpy as np
import torch
from PIL import Image

from .preprocess import prepare
from .weights import DEFAULT_MODEL, load_generator


class FaceUpscaler:
    """Upscales 16x16 faces to 128x128.

        upscaler = FaceUpscaler.from_pretrained("gan")
        face = upscaler.upscale("photo.jpg")      # any square picture -> 128x128 PIL image
        face = upscaler.upscale("group.jpg", roi=(410, 120, 450, 160))   # a small face in a large photo

    Note that the details in the output are invented by the network, they are plausible, but they are not
    the details of the real person. Never use the output to identify somebody.
    """

    def __init__(self, generator, device=None, batch_size=100):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.generator = generator.to(self.device).eval()
        self.batch_size = batch_size
        self.scale = generator.scale

    @classmethod
    def from_pretrained(cls, name_or_path=DEFAULT_MODEL, device=None, batch_size=100):
        """Loads a bundled generator ("l1", "gan", "gan_unet", "gan_unet_slow") or a .safetensors file."""
        return cls(load_generator(name_or_path), device=device, batch_size=batch_size)

    @torch.no_grad()
    def upscale_batch(self, lr_u8):
        """(N,h,w,3) uint8 array of low-resolution faces -> (N,h*scale,w*scale,3) uint8 array."""
        lr_u8 = np.asarray(lr_u8)
        if lr_u8.ndim != 4 or lr_u8.shape[-1] != 3 or lr_u8.dtype != np.uint8:
            raise ValueError("Expected a uint8 array of shape (N,h,w,3), got %s %s" % (lr_u8.dtype, lr_u8.shape))
        out = []
        for ii in range(0, len(lr_u8), self.batch_size):
            lr = torch.from_numpy(np.array(lr_u8[ii:ii + self.batch_size])).to(self.device)
            sr = self.generator(lr.permute(0, 3, 1, 2).float() / 255)
            out.append((sr.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0,) + lr_u8.shape[1:], dtype=np.uint8)

    def upscale(self, image, roi=None, lr_size=16):
        """Any square picture (path, PIL image or uint8 array) -> upscaled PIL image.

        The picture is resized to lr_size x lr_size first, see `facecast.prepare`.

        roi=(left, top, right, bottom) selects a region of the picture, e.g., a small face in a large
        photo. Only this region is upscaled, and the picture does not have to be square then.

        A list or a tuple of pictures gives a list of PIL images (roi is then None or a list of boxes).
        """
        if isinstance(image, (list, tuple)):
            rois = [None] * len(image) if roi is None else list(roi)
            if len(rois) != len(image):
                raise ValueError("Got %d pictures and %d rois" % (len(image), len(rois)))
            if not image:
                return []
            lr = np.stack([prepare(img, lr_size, self.scale, r) for img, r in zip(image, rois)])
            return [Image.fromarray(sr) for sr in self.upscale_batch(lr)]
        if not isinstance(image, (str, os.PathLike, Image.Image, np.ndarray)):
            raise TypeError("Expected a path, a PIL image, a numpy array or a list of them, got %s" % type(image).__name__)
        lr = prepare(image, lr_size, self.scale, roi)
        return Image.fromarray(self.upscale_batch(lr[None])[0])
