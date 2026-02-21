from pathlib import Path
from typing import Any

from PIL import Image

from pixelart.eval import evaluate_prompts


class DummyPipe:
    def __init__(self, name: str) -> None:
        self.name = name
        self.device = "cpu"


def make_prompt(prompt_id: str, seed: int) -> dict[str, Any]:
    return {
        "id": prompt_id,
        "category": "prompt_following",
        "prompt": "pixel art, test prompt",
        "negative_prompt": "",
        "seeds": [seed],
        "expected_traits": ["trait"],
        "failure_checks": ["failure"],
    }


def make_dirs(root: Path) -> tuple[Path, Path, Path, Path]:
    baseline_dir = root / "baseline"
    lora_dir = root / "lora"
    downscaled_dir = root / "downscaled"
    grids_dir = root / "grids"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    lora_dir.mkdir(parents=True, exist_ok=True)
    downscaled_dir.mkdir(parents=True, exist_ok=True)
    grids_dir.mkdir(parents=True, exist_ok=True)
    return baseline_dir, lora_dir, downscaled_dir, grids_dir


def test_evaluate_prompts_speed_mode_call_order(tmp_path: Path, monkeypatch) -> None:
    build_calls: list[str | None] = []
    generation_order: list[str] = []

    def fake_build_pipeline(
        base_model: str,
        vae_model: str | None,
        dtype_name: str,
        device: str,
        lora_path: str | None = None,
        lora_scale: float = 1.0,
    ) -> DummyPipe:
        del base_model, vae_model, dtype_name, device, lora_scale
        build_calls.append(lora_path)
        return DummyPipe("baseline" if lora_path is None else "lora")

    def fake_generate_image(
        pipe: DummyPipe,
        prompt: str,
        negative_prompt: str,
        seed: int,
        steps: int,
        guidance_scale: float,
        width: int,
        height: int,
    ) -> Image.Image:
        del prompt, negative_prompt, seed, steps, guidance_scale
        generation_order.append(pipe.name)
        color = (255, 255, 255) if pipe.name == "baseline" else (0, 0, 0)
        return Image.new("RGB", (width, height), color=color)

    monkeypatch.setattr("pixelart.eval.build_pipeline", fake_build_pipeline)
    monkeypatch.setattr("pixelart.eval.generate_image", fake_generate_image)

    baseline_dir, lora_dir, downscaled_dir, grids_dir = make_dirs(tmp_path)
    rows = evaluate_prompts(
        prompts=[make_prompt("one", 1), make_prompt("two", 2)],
        base_model="base",
        vae_model=None,
        dtype_name="fp16",
        device="cpu",
        lora_path="pixelart-lora-output/checkpoint-8000",
        lora_scale=0.65,
        width=8,
        height=8,
        steps=2,
        guidance_scale=1.0,
        downscale_factor=2,
        eval_mode="speed",
        baseline_dir=baseline_dir,
        lora_dir=lora_dir,
        downscaled_dir=downscaled_dir,
        grids_dir=grids_dir,
        checkpoint_used="checkpoint-8000",
    )

    assert build_calls == [None, "pixelart-lora-output/checkpoint-8000"]
    assert generation_order == ["baseline", "lora", "baseline", "lora"]
    assert len(rows) == 2


def test_evaluate_prompts_memory_safe_mode_call_order(tmp_path: Path, monkeypatch) -> None:
    build_calls: list[str | None] = []
    generation_order: list[str] = []

    def fake_build_pipeline(
        base_model: str,
        vae_model: str | None,
        dtype_name: str,
        device: str,
        lora_path: str | None = None,
        lora_scale: float = 1.0,
    ) -> DummyPipe:
        del base_model, vae_model, dtype_name, device, lora_scale
        build_calls.append(lora_path)
        return DummyPipe("baseline" if lora_path is None else "lora")

    def fake_generate_image(
        pipe: DummyPipe,
        prompt: str,
        negative_prompt: str,
        seed: int,
        steps: int,
        guidance_scale: float,
        width: int,
        height: int,
    ) -> Image.Image:
        del prompt, negative_prompt, seed, steps, guidance_scale
        generation_order.append(pipe.name)
        color = (255, 255, 255) if pipe.name == "baseline" else (0, 0, 0)
        return Image.new("RGB", (width, height), color=color)

    monkeypatch.setattr("pixelart.eval.build_pipeline", fake_build_pipeline)
    monkeypatch.setattr("pixelart.eval.generate_image", fake_generate_image)

    baseline_dir, lora_dir, downscaled_dir, grids_dir = make_dirs(tmp_path)
    rows = evaluate_prompts(
        prompts=[make_prompt("one", 1), make_prompt("two", 2)],
        base_model="base",
        vae_model=None,
        dtype_name="fp16",
        device="cpu",
        lora_path="pixelart-lora-output/checkpoint-8000",
        lora_scale=0.65,
        width=8,
        height=8,
        steps=2,
        guidance_scale=1.0,
        downscale_factor=2,
        eval_mode="memory_safe",
        baseline_dir=baseline_dir,
        lora_dir=lora_dir,
        downscaled_dir=downscaled_dir,
        grids_dir=grids_dir,
        checkpoint_used="checkpoint-8000",
    )

    assert build_calls == [None, "pixelart-lora-output/checkpoint-8000"]
    assert generation_order == ["baseline", "baseline", "lora", "lora"]
    assert len(rows) == 2
