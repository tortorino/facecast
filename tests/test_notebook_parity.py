"""The package has to give the same results as facecast.ipynb.

The expected numbers are the ones printed in the notebook. These tests need the files the notebook
produces or uses (data/, models/), which are not part of the package, so they are skipped without them.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from facecast import FaceUpscaler, Generator, prepare

ROOT = Path(__file__).resolve().parent.parent
TEST_SET = ROOT / "data" / "celeba_test_random_128_2000.npy"
CHECKPOINTS = {"l1": "generator_l1.pt", "gan": "generator_gan.pt",
               "gan_unet": "generator_gan_unet.pt", "gan_unet_slow": "generator_gan_unet_slow.pt"}

# PSNR (dB) printed in the notebook: the final table and the two demo cells
TEST_PSNR = {"l1": 24.3042, "gan": 23.0634, "gan_unet": 22.8635, "gan_unet_slow": 22.6582}
DEMO_PSNR = {
    "Ivan_Ruller_2016.jpg": {"l1": 27.06, "gan": 25.67, "gan_unet": 25.45, "gan_unet_slow": 25.18},
    "creeper_lamp.jpg": {"l1": 24.35, "gan": 22.17, "gan_unet": 21.84, "gan_unet_slow": 22.17},
}


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2, axis=(1, 2, 3))
    return 10 * np.log10(255 ** 2 / np.maximum(mse, 1e-10))


def notebook_preprocess(path):
    # preprocess() of the notebook: center crop to a square and resize to 128x128
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s)).resize((128, 128), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


@pytest.mark.skipif(not TEST_SET.exists(), reason="needs the test faces prepared by the notebook")
@pytest.mark.parametrize("name", list(TEST_PSNR))
def test_test_set_psnr(name):
    hr = np.load(TEST_SET, mmap_mode="r")
    lr = np.stack([prepare(face) for face in hr])
    sr = FaceUpscaler.from_pretrained(name).upscale_batch(lr)
    assert psnr(sr, np.asarray(hr)).mean() == pytest.approx(TEST_PSNR[name], abs=0.002)


@pytest.mark.parametrize("photo", list(DEMO_PSNR))
@pytest.mark.parametrize("name", list(TEST_PSNR))
def test_demo_psnr(photo, name):
    path = ROOT / "data" / photo
    if not path.exists():
        pytest.skip("needs data/%s" % photo)
    hr = notebook_preprocess(path)
    sr = np.asarray(FaceUpscaler.from_pretrained(name).upscale(path))
    assert psnr(sr[None], hr[None])[0] == pytest.approx(DEMO_PSNR[photo][name], abs=0.006)


@pytest.mark.parametrize("name", list(CHECKPOINTS))
def test_same_output_as_the_notebook_pipeline(name):
    checkpoint = ROOT / "models" / CHECKPOINTS[name]
    photo = ROOT / "data" / "Ivan_Ruller_2016.jpg"
    if not checkpoint.exists() or not photo.exists():
        pytest.skip("needs models/%s and data/%s" % (checkpoint.name, photo.name))
    upscaler = FaceUpscaler.from_pretrained(name)
    device = upscaler.device

    # The pipeline of the notebook, with the weights of the notebook
    generator = Generator(8, 16, 64).to(device).eval()
    generator.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
    hr = torch.from_numpy(np.array(notebook_preprocess(photo))[None]).to(device).permute(0, 3, 1, 2).float() / 255
    lr = torch.round(F.avg_pool2d(hr, 8) * 255) / 255
    with torch.no_grad():
        sr = generator(lr)
    expected = (sr.float().clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()[0]

    # The low-resolution input is exactly the same
    assert np.array_equal(prepare(photo), (lr * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()[0])
    # The output can differ by one gray level in a few pixels: two runs of the same network on the same
    # input do not give bit-identical floats, and a value close to .5 is then rounded the other way
    diff = np.abs(np.asarray(upscaler.upscale(photo)).astype(int) - expected.astype(int))
    assert diff.max() <= 1
    assert np.mean(diff > 0) < 0.02
