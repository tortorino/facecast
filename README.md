The notebook takes a 16×16 pixel picture of a face and produces a 128×128 picture of it. That is 8× upscaling per side, so the output has 64 times more pixels than the input. Most of the output therefore cannot be *recovered* from the input; it has to be *invented* in a plausible way, which is why the task is called **face hallucination**. The notebook trains a neural network to do this and compares them with the ordinary interpolation methods in OpenCV (`cv2.resize`).

## Using the trained models in another project

The generators trained in `facecast.ipynb` are also available as a small Python package. It only contains what is needed to use them (no training code), and its only dependencies are `torch`, `numpy`, `pillow` and `safetensors`.

Install it from a clone of this repository (or straight from its git URL):

```
pip install /path/to/facecast
```

Then:

```python
from facecast import FaceUpscaler

upscaler = FaceUpscaler.from_pretrained("gan")   # "l1", "gan", "gan_unet" or "gan_unet_slow"
face = upscaler.upscale("photo.jpg")             # any square picture -> 128x128 PIL image
face.save("face_128.png")

# a small face in a large photo: give the region of the face as (left, top, right, bottom)
face = upscaler.upscale("group.jpg", roi=(410, 120, 450, 160))
```

or from the command line:

```
facecast upscale photo.jpg -m gan -o face_128.png
facecast upscale group.jpg --roi 410 120 450 160
facecast models
```

### The models

| Name | What it is |
|---|---|
| `l1` | Stage 1, pixel (L1) loss only. The most accurate one (best PSNR and SSIM), but smooth. |
| `gan` | Stage 2, perceptual + adversarial loss (10 epochs). Sharper, more realistic faces. The default. |
| `gan_unet` | Stage 2 with a U-Net discriminator and a relativistic loss (4 epochs). |
| `gan_unet_slow` | As `gan_unet`, with a 4 times lower learning rate of the discriminator. |

The three `gan` models give very similar results, see the notebook for the comparison.

### The input picture

The generators take a 16×16 picture and return a 128×128 one. `upscale()` accepts a path, a PIL image or a `uint8` NumPy array of **any square size** and resizes it to 16×16 first:

| Size of the picture | What happens |
|---|---|
| 16×16 | used as it is |
| 128×128 or larger | resized to 128×128 and averaged in 8×8 blocks, exactly as in the training |
| between 16 and 128 | averaged straight down to 16×16 |
| smaller than 16×16 | upscaled to 16×16 first, with a warning |
| not square | rejected with an error, crop it to the face first |

### A small face in a large photo

With `roi=(left, top, right, bottom)` (in pixels), only this region of the picture is upscaled. The photo itself does not have to be square then. The rules from the table above apply to the size of the region, so a 40×40 face is averaged down to 16×16 and comes back as a 128×128 one.

The generators need a square input, so a region that is not square is made square: its shorter side is grown around the center of the region (nothing inside the region is lost), and the region is shifted if it would stick out of the photo. `facecast.square_roi(roi, width, height)` returns the region that is really used.

The region should frame the face the way the CelebA faces are framed: the whole head with some hair, the chin and a bit of the background, not only the eyes and the mouth.

`facecast.prepare(picture)` does only this step and returns the 16×16 array. For many faces that are 16×16 already, use `upscaler.upscale_batch(array)` with a `uint8` array of shape (N, 16, 16, 3).

Things to keep in mind:

* There is no face detection. The generators were trained on aligned CelebA faces, so the face should be centered, front-facing and fill the picture in a similar way (with some hair and chin visible). Otherwise the eyes and the mouth end up in the wrong places.
* The details in the output are **invented**. They look plausible, but they are not the details of the real person, so the output must never be used to identify somebody.
* The generators were trained on clean, artificially downscaled pictures. Real tiny pictures with blur, noise or JPEG artifacts give worse results.

### Development

```
uv sync                                  # the package, the notebook's dependencies and pytest
uv run pytest                            # some tests need data/ and models/ and are skipped without them
uv run python scripts/export_weights.py  # after retraining: models/*.pt -> src/facecast/assets
```
