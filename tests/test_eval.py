import pytest
from PIL import Image

from pixelart.eval import nearest_downscale


def test_nearest_downscale_dimensions() -> None:
    image = Image.new("RGB", (1024, 1024), (255, 255, 255))
    small = nearest_downscale(image, 8)
    assert small.size == (128, 128)


def test_nearest_downscale_factor_validation() -> None:
    image = Image.new("RGB", (1002, 1002), (255, 255, 255))
    with pytest.raises(ValueError):
        nearest_downscale(image, 8)
