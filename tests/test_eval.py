import pytest
from PIL import Image

from pixelart.eval import (
    nearest_downscale,
    normalize_eval_mode_for_device,
    normalize_runtime_settings,
)


def test_nearest_downscale_dimensions() -> None:
    image = Image.new("RGB", (1024, 1024), (255, 255, 255))
    small = nearest_downscale(image, 8)
    assert small.size == (128, 128)


def test_nearest_downscale_factor_validation() -> None:
    image = Image.new("RGB", (1002, 1002), (255, 255, 255))
    with pytest.raises(ValueError):
        nearest_downscale(image, 8)


def test_normalize_runtime_settings_fallback_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pixelart.eval.is_cuda_available", lambda: False)
    device, dtype_name = normalize_runtime_settings("cuda", "bf16")
    assert device == "cpu"
    assert dtype_name == "fp32"


def test_normalize_runtime_settings_preserves_cuda_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("pixelart.eval.is_cuda_available", lambda: True)
    device, dtype_name = normalize_runtime_settings("cuda", "bf16")
    assert device == "cuda"
    assert dtype_name == "bf16"


def test_normalize_runtime_settings_invalid_dtype() -> None:
    with pytest.raises(ValueError, match="Unsupported dtype"):
        normalize_runtime_settings("cpu", "int8")


def test_normalize_eval_mode_for_device_forces_memory_safe_on_cpu() -> None:
    assert normalize_eval_mode_for_device("speed", "cpu") == "memory_safe"


def test_normalize_eval_mode_for_device_keeps_speed_on_cuda() -> None:
    assert normalize_eval_mode_for_device("speed", "cuda") == "speed"
