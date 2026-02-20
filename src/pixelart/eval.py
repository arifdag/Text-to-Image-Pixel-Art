from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, ImageDraw

from .io_utils import ensure_dir, timestamp_utc

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime
    yaml = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline SDXL vs trained LoRA.")
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if yaml is None:
        raise RuntimeError("PyYAML is required for YAML config files.")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Eval config must be a mapping/object.")
    return data


def load_prompts(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Prompt file must contain a JSON list.")
    return data


def nearest_downscale(image: Image.Image, factor: int) -> Image.Image:
    if factor <= 0:
        raise ValueError("Downscale factor must be > 0")
    width, height = image.size
    if width % factor != 0 or height % factor != 0:
        raise ValueError(f"Image size {image.size} is not divisible by factor {factor}")
    return image.resize((width // factor, height // factor), Image.Resampling.NEAREST)


def draw_label(image: Image.Image, label: str) -> Image.Image:
    labeled = ImageOps.expand(image, border=(0, 24, 0, 0), fill="black")
    draw = ImageDraw.Draw(labeled)
    draw.text((8, 4), label, fill="white")
    return labeled


def concat_horiz(images: list[Image.Image]) -> Image.Image:
    width = sum(img.width for img in images)
    height = max(img.height for img in images)
    canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
    x = 0
    for img in images:
        canvas.paste(img, (x, 0))
        x += img.width
    return canvas


def build_pipeline(
    base_model: str,
    vae_model: str | None,
    dtype_name: str,
    device: str,
    lora_path: str | None = None,
    lora_scale: float = 1.0,
):
    import torch
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline

    if dtype_name == "bf16":
        dtype = torch.bfloat16
    elif dtype_name == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(vae_model, torch_dtype=dtype) if vae_model else None
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
        vae=vae,
    )
    pipe = pipe.to(device)
    if lora_path:
        pipe.load_lora_weights(lora_path)
        if hasattr(pipe, "fuse_lora"):
            pipe.fuse_lora(lora_scale=lora_scale)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe


def generate_image(
    pipe: Any,
    prompt: str,
    negative_prompt: str,
    seed: int,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
) -> Image.Image:
    import torch

    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    )
    return output.images[0]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    prompts = load_prompts(Path(str(config["prompts_file"])))

    output_dir = ensure_dir(Path(str(config.get("output_dir", "artifacts/eval"))))
    baseline_dir = ensure_dir(output_dir / "baseline")
    lora_dir = ensure_dir(output_dir / "lora")
    downscaled_dir = ensure_dir(output_dir / "downscaled")
    grids_dir = ensure_dir(output_dir / "grids")

    base_model = str(config["base_model"])
    lora_path = config.get("lora_path")
    vae_model = config.get("vae_model")
    dtype_name = str(config.get("dtype", "bf16"))
    device = str(config.get("device", "cuda"))
    width = int(config.get("width", 1024))
    height = int(config.get("height", 1024))
    steps = int(config.get("num_inference_steps", 30))
    guidance_scale = float(config.get("guidance_scale", 7.0))
    downscale_factor = int(config.get("downscale_factor", 8))
    lora_scale = float(config.get("lora_scale", 1.0))

    baseline_pipe = build_pipeline(base_model, vae_model, dtype_name, device)
    lora_pipe = build_pipeline(base_model, vae_model, dtype_name, device, lora_path, lora_scale)

    report_rows: list[dict[str, Any]] = []
    for row in prompts:
        prompt_id = str(row.get("id", "prompt"))
        prompt = str(row["prompt"])
        negative_prompt = str(row.get("negative_prompt", ""))
        seed = int(row.get("seed", 42))

        base_img = generate_image(
            pipe=baseline_pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
        )
        lora_img = generate_image(
            pipe=lora_pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
        )

        base_path = baseline_dir / f"{prompt_id}_seed{seed}.png"
        lora_path_file = lora_dir / f"{prompt_id}_seed{seed}.png"
        base_img.save(base_path)
        lora_img.save(lora_path_file)

        base_small = nearest_downscale(base_img, downscale_factor)
        lora_small = nearest_downscale(lora_img, downscale_factor)
        base_small_path = downscaled_dir / f"{prompt_id}_seed{seed}_baseline.png"
        lora_small_path = downscaled_dir / f"{prompt_id}_seed{seed}_lora.png"
        base_small.save(base_small_path)
        lora_small.save(lora_small_path)

        grid = concat_horiz(
            [
                draw_label(base_small, f"{prompt_id} baseline"),
                draw_label(lora_small, f"{prompt_id} lora"),
            ]
        )
        grid_path = grids_dir / f"{prompt_id}_seed{seed}_grid.png"
        grid.save(grid_path)

        report_rows.append(
            {
                "id": prompt_id,
                "prompt": prompt,
                "seed": seed,
                "baseline": base_path.as_posix(),
                "lora": lora_path_file.as_posix(),
                "baseline_downscaled": base_small_path.as_posix(),
                "lora_downscaled": lora_small_path.as_posix(),
                "grid": grid_path.as_posix(),
            }
        )
        print(f"Evaluated {prompt_id} (seed={seed})")

    report_path = output_dir / "report.json"
    report = {
        "created_at": timestamp_utc(),
        "config": config,
        "results": report_rows,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()

