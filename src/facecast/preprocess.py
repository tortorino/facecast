"""Turns any square picture into the low-resolution input the generators expect."""

import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def load_image(image):
    """Path, PIL image or (H,W,3) / (H,W) / (H,W,4) uint8 array -> RGB PIL image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, os.PathLike)):
        with Image.open(image) as img:
            return img.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            raise TypeError("Image arrays have to be uint8 (0 - 255), got %s" % image.dtype)
        if image.ndim not in (2, 3):
            raise ValueError("Expected one image of shape (H,W) or (H,W,C), got %s" % (image.shape,))
        return Image.fromarray(image).convert("RGB")
    raise TypeError("Expected a path, a PIL image or a numpy array, got %s" % type(image).__name__)


def block_average(hr_u8, scale):
    """(H,W,3) uint8 -> (H/scale,W/scale,3) uint8. Averages scale x scale blocks and rounds to 8 bits.

    This is the degradation the generators were trained with (the same computation as in the notebook).
    """
    hr = torch.from_numpy(np.array(hr_u8, dtype=np.uint8)).permute(2, 0, 1)[None].float() / 255
    lr = torch.round(F.avg_pool2d(hr, scale) * 255)
    return lr[0].permute(1, 2, 0).byte().numpy()


def square_roi(roi, width, height):
    """(left, top, right, bottom) -> a square box inside a width x height picture.

    The generators need a square input. A box that is not square is grown on its shorter side (around its
    center), so nothing inside the box is lost, and it is shifted if it would stick out of the picture.
    """
    try:
        left, top, right, bottom = [int(round(float(v))) for v in roi]
    except (TypeError, ValueError):
        raise ValueError("roi has to be (left, top, right, bottom) in pixels, got %r" % (roi,)) from None
    if right <= left or bottom <= top:
        raise ValueError("roi is empty: (left, top, right, bottom) = %r" % ((left, top, right, bottom),))
    if left >= width or top >= height or right <= 0 or bottom <= 0:
        raise ValueError("roi %r is outside of the %dx%d picture" % ((left, top, right, bottom), width, height))
    side = max(right - left, bottom - top)
    if side > min(width, height):
        raise ValueError("roi %r can not be made square inside the %dx%d picture"
                         % ((left, top, right, bottom), width, height))
    # Grow around the center, then shift back inside the picture
    left = (left + right - side) // 2
    top = (top + bottom - side) // 2
    left = min(max(left, 0), width - side)
    top = min(max(top, 0), height - side)
    return left, top, left + side, top + side


def prepare(image, lr_size=16, scale=8, roi=None):
    """Any square picture -> (lr_size, lr_size, 3) uint8 array, the input of a generator.

    With roi=(left, top, right, bottom), only this region of the picture is used (e.g., a small face in a
    large photo). The picture itself does not have to be square then, and the region is made square, see
    `square_roi`. The rules below apply to the size of the region.

    * lr_size x lr_size: used as it is.
    * lr_size*scale or larger (128+): resized to lr_size*scale and block-averaged, exactly as in training.
    * in between: area-averaged straight down to lr_size.
    * smaller than lr_size: upscaled with bicubic interpolation (with a warning).

    The picture has to be square. The generators were trained on aligned CelebA faces, so the face should
    be centered, front-facing and fill the picture in a similar way.
    """
    img = load_image(image)
    if roi is not None:
        img = img.crop(square_roi(roi, *img.size))
    w, h = img.size
    if w != h:
        raise ValueError("The picture has to be square, got %dx%d. Crop it or pass the roi of the face." % (w, h))

    hr_size = lr_size * scale
    if w == lr_size:
        pass
    elif w >= hr_size:
        if w != hr_size:
            img = img.resize((hr_size, hr_size), Image.LANCZOS)
        return block_average(np.asarray(img, dtype=np.uint8), scale)
    elif w > lr_size:
        img = img.resize((lr_size, lr_size), Image.BOX)
    else:
        warnings.warn("The picture is only %dx%d, smaller than the %dx%d input of the generator. "
                      "It is upscaled first, so expect a worse result." % (w, h, lr_size, lr_size), stacklevel=2)
        img = img.resize((lr_size, lr_size), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)
