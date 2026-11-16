import numpy as np
import pytest
from PIL import Image

from facecast import prepare


def random_image(size, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)


def test_16x16_is_used_as_it_is():
    img = random_image(16)
    assert np.array_equal(prepare(img), img)


def test_128x128_is_block_averaged():
    img = random_image(128)
    expected = img.reshape(16, 8, 16, 8, 3).astype(np.float64).mean(axis=(1, 3))
    lr = prepare(img)
    assert lr.shape == (16, 16, 3) and lr.dtype == np.uint8
    # the same up to the rounding of exact .5 values
    assert np.abs(lr - expected).max() <= 0.5 + 1e-3


@pytest.mark.parametrize("size", [17, 100, 127, 129, 600, 917])
def test_other_sizes_are_resized(size):
    lr = prepare(random_image(size))
    assert lr.shape == (16, 16, 3) and lr.dtype == np.uint8


def test_a_flat_picture_stays_flat():
    for size in (40, 128, 300):
        img = np.full((size, size, 3), 77, dtype=np.uint8)
        assert np.all(prepare(img) == 77)


def test_small_picture_warns():
    with pytest.warns(UserWarning, match="smaller than"):
        lr = prepare(random_image(10))
    assert lr.shape == (16, 16, 3)


def test_non_square_is_rejected():
    with pytest.raises(ValueError, match="square"):
        prepare(np.zeros((100, 120, 3), dtype=np.uint8))


def test_input_types(tmp_path):
    img = random_image(128)
    path = tmp_path / "face.png"
    Image.fromarray(img).save(path)
    expected = prepare(img)
    assert np.array_equal(prepare(Image.fromarray(img)), expected)
    assert np.array_equal(prepare(path), expected)
    assert np.array_equal(prepare(str(path)), expected)


def test_grayscale_and_rgba_arrays():
    assert prepare(np.zeros((64, 64), dtype=np.uint8)).shape == (16, 16, 3)
    assert prepare(np.zeros((64, 64, 4), dtype=np.uint8)).shape == (16, 16, 3)


def test_float_arrays_are_rejected():
    with pytest.raises(TypeError, match="uint8"):
        prepare(np.zeros((16, 16, 3), dtype=np.float32))
