import numpy as np
import pytest
from PIL import Image

from facecast import FaceUpscaler, list_models, load_generator
from facecast.cli import main
from facecast.weights import weights_path

MODELS = list(list_models())


def random_image(size, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)


def test_four_models_are_bundled():
    assert MODELS == ["l1", "gan", "gan_unet", "gan_unet_slow"]


@pytest.mark.parametrize("name", MODELS)
def test_every_model_upscales(name):
    face = FaceUpscaler.from_pretrained(name).upscale(random_image(200))
    assert isinstance(face, Image.Image) and face.size == (128, 128) and face.mode == "RGB"


def test_models_differ():
    img = random_image(128)
    outputs = [np.asarray(FaceUpscaler.from_pretrained(name).upscale(img)) for name in MODELS]
    for ii in range(len(outputs)):
        for jj in range(ii + 1, len(outputs)):
            assert not np.array_equal(outputs[ii], outputs[jj])


def test_batch_and_list():
    upscaler = FaceUpscaler.from_pretrained("l1", batch_size=2)
    lr = np.stack([random_image(16, seed) for seed in range(5)])
    sr = upscaler.upscale_batch(lr)
    assert sr.shape == (5, 128, 128, 3) and sr.dtype == np.uint8
    faces = upscaler.upscale([img for img in lr])
    assert len(faces) == 5
    # the same result one by one, in a batch and in a list
    assert np.abs(np.asarray(faces[3]).astype(int) - sr[3]).max() <= 1
    assert np.abs(np.asarray(upscaler.upscale(lr[3])).astype(int) - sr[3]).max() <= 1
    assert upscaler.upscale([]) == []


def test_bad_inputs():
    upscaler = FaceUpscaler.from_pretrained("l1")
    with pytest.raises(ValueError):
        upscaler.upscale_batch(np.zeros((16, 16, 3), dtype=np.uint8))
    with pytest.raises(TypeError):
        upscaler.upscale(42)
    with pytest.raises(ValueError, match="Unknown model"):
        FaceUpscaler.from_pretrained("no_such_model")
    with pytest.raises(FileNotFoundError):
        load_generator("missing.safetensors")


def test_cpu_and_weight_file():
    upscaler = FaceUpscaler.from_pretrained(str(weights_path("gan")), device="cpu")
    assert upscaler.device.type == "cpu"
    assert upscaler.upscale(random_image(16)).size == (128, 128)


def test_cli(tmp_path, capsys):
    source = tmp_path / "face.png"
    Image.fromarray(random_image(300)).save(source)
    assert main(["upscale", str(source), "-m", "l1"]) == 0
    with Image.open(tmp_path / "face_facecast.png") as out:
        assert out.size == (128, 128)
    assert main(["upscale", str(source), "-o", str(tmp_path / "out.png")]) == 0
    assert (tmp_path / "out.png").exists()
    assert main(["models"]) == 0
    assert "gan_unet_slow" in capsys.readouterr().out


def test_cli_rejects_non_square(tmp_path, capsys):
    source = tmp_path / "wide.png"
    Image.new("RGB", (200, 100)).save(source)
    assert main(["upscale", str(source)]) == 1
    assert "square" in capsys.readouterr().err
