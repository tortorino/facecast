import numpy as np
import pytest
from PIL import Image

from facecast import FaceUpscaler, prepare, square_roi
from facecast.cli import main


def random_image(height, width, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


def test_square_roi_is_kept():
    assert square_roi((100, 50, 140, 90), 800, 600) == (100, 50, 140, 90)


def test_roi_is_grown_to_a_square_around_its_center():
    # 20 wide, 40 high -> 40x40 with the same center
    assert square_roi((100, 100, 120, 140), 800, 600) == (90, 100, 130, 140)
    # 40 wide, 20 high
    assert square_roi((100, 100, 140, 120), 800, 600) == (100, 90, 140, 130)


def test_roi_is_shifted_back_into_the_picture():
    assert square_roi((0, 100, 10, 140), 800, 600) == (0, 100, 40, 140)
    assert square_roi((790, 100, 800, 140), 800, 600) == (760, 100, 800, 140)
    assert square_roi((100, 590, 140, 600), 800, 600) == (100, 560, 140, 600)
    # a box that sticks out is moved in as a whole
    assert square_roi((-10, -10, 30, 30), 800, 600) == (0, 0, 40, 40)


def test_roi_accepts_floats():
    assert square_roi((100.4, 50.2, 140.4, 90.2), 800, 600) == (100, 50, 140, 90)


@pytest.mark.parametrize("roi", [(10, 10, 10, 50), (50, 10, 10, 50), (900, 10, 950, 60), (0, 0, 10, 700), "abcd", (1, 2, 3)])
def test_bad_rois_are_rejected(roi):
    with pytest.raises(ValueError):
        square_roi(roi, 800, 600)


def test_prepare_uses_only_the_roi():
    photo = random_image(600, 800)
    # the same as cropping by hand
    assert np.array_equal(prepare(photo, roi=(300, 200, 340, 240)), prepare(photo[200:240, 300:340]))
    assert np.array_equal(prepare(photo, roi=(300, 200, 316, 216)), photo[200:216, 300:316])
    assert np.array_equal(prepare(photo, roi=(100, 100, 400, 400)), prepare(photo[100:400, 100:400]))


def test_a_picture_with_a_roi_does_not_have_to_be_square():
    photo = random_image(600, 800)
    with pytest.raises(ValueError, match="square"):
        prepare(photo)
    assert prepare(photo, roi=(0, 0, 100, 100)).shape == (16, 16, 3)


def test_tiny_roi_warns():
    with pytest.warns(UserWarning, match="smaller than"):
        prepare(random_image(600, 800), roi=(300, 200, 310, 210))


def test_small_face_in_a_large_photo():
    upscaler = FaceUpscaler.from_pretrained("l1")
    face = random_image(40, 40, seed=1)
    photo = random_image(600, 800)
    photo[200:240, 300:340] = face
    sr = upscaler.upscale(photo, roi=(300, 200, 340, 240))
    assert sr.size == (128, 128)
    assert np.array_equal(np.asarray(sr), np.asarray(upscaler.upscale(face)))


def test_list_of_pictures_with_rois():
    upscaler = FaceUpscaler.from_pretrained("l1")
    photo = random_image(600, 800)
    faces = upscaler.upscale([photo, photo], roi=[(0, 0, 50, 50), (100, 100, 150, 150)])
    assert len(faces) == 2 and not np.array_equal(np.asarray(faces[0]), np.asarray(faces[1]))
    with pytest.raises(ValueError, match="rois"):
        upscaler.upscale([photo, photo], roi=[(0, 0, 50, 50)])


def test_cli_roi(tmp_path, capsys):
    source = tmp_path / "photo.png"
    Image.fromarray(random_image(600, 800)).save(source)
    assert main(["upscale", str(source), "-m", "l1", "--roi", "300", "200", "340", "240"]) == 0
    with Image.open(tmp_path / "photo_facecast.png") as out:
        assert out.size == (128, 128)
    assert main(["upscale", str(source), "--roi", "900", "200", "950", "240"]) == 1
    assert "outside" in capsys.readouterr().err
