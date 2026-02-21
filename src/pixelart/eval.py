from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from .io_utils import ensure_dir, timestamp_utc

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime
    yaml = None

ALLOWED_PROMPT_CATEGORIES = {
    "prompt_following",
    "pixel_integrity",
    "detail_readability",
    "composition_clarity",
}
ALLOWED_EVAL_MODES = {"speed", "memory_safe"}


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


def _normalize_string_list(field_name: str, value: Any, prompt_id: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Prompt '{prompt_id}' must include a non-empty '{field_name}' list.")
    normalized: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                f"Prompt '{prompt_id}' has invalid {field_name}[{index}]: expected non-empty string."
            )
        normalized.append(item.strip())
    return normalized


def _normalize_seeds(row: dict[str, Any], prompt_id: str) -> list[int]:
    seeds_raw = row.get("seeds")
    if seeds_raw is None:
        # Backward-compatible fallback for v1 prompt files.
        if "seed" in row:
            seeds_raw = [row["seed"]]
        else:
            raise ValueError(f"Prompt '{prompt_id}' must include 'seeds' or legacy 'seed'.")
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise ValueError(f"Prompt '{prompt_id}' must include a non-empty 'seeds' list.")

    seeds: list[int] = []
    for index, seed in enumerate(seeds_raw):
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"Prompt '{prompt_id}' has invalid seeds[{index}]: expected integer.")
        seeds.append(seed)
    return seeds


def normalize_prompt_row(
    row: Any,
    row_index: int,
    allowed_categories: set[str],
) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"Prompt row #{row_index} must be an object.")

    prompt_id_raw = row.get("id")
    if not isinstance(prompt_id_raw, str) or not prompt_id_raw.strip():
        raise ValueError(f"Prompt row #{row_index} must include a non-empty 'id'.")
    prompt_id = prompt_id_raw.strip()

    category_raw = row.get("category")
    if not isinstance(category_raw, str) or not category_raw.strip():
        raise ValueError(f"Prompt '{prompt_id}' must include a non-empty 'category'.")
    category = category_raw.strip()
    if category not in allowed_categories:
        allowed_text = ", ".join(sorted(allowed_categories))
        raise ValueError(
            f"Prompt '{prompt_id}' has unsupported category '{category}'. Allowed: {allowed_text}"
        )

    prompt_raw = row.get("prompt")
    if not isinstance(prompt_raw, str) or not prompt_raw.strip():
        raise ValueError(f"Prompt '{prompt_id}' must include a non-empty 'prompt'.")
    prompt = prompt_raw.strip()

    negative_prompt = row.get("negative_prompt", "")
    if not isinstance(negative_prompt, str):
        raise ValueError(f"Prompt '{prompt_id}' has invalid 'negative_prompt': expected string.")

    seeds = _normalize_seeds(row, prompt_id)
    expected_traits = _normalize_string_list("expected_traits", row.get("expected_traits"), prompt_id)
    failure_checks = _normalize_string_list("failure_checks", row.get("failure_checks"), prompt_id)

    return {
        "id": prompt_id,
        "category": category,
        "prompt": prompt,
        "negative_prompt": negative_prompt.strip(),
        "seeds": seeds,
        "expected_traits": expected_traits,
        "failure_checks": failure_checks,
    }


def load_prompts(
    path: Path,
    allowed_categories: set[str] | None = None,
) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Prompt file must contain a JSON list.")
    categories = allowed_categories or ALLOWED_PROMPT_CATEGORIES
    prompts: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row_index, row in enumerate(data, start=1):
        normalized_row = normalize_prompt_row(row, row_index, categories)
        prompt_id = normalized_row["id"]
        if prompt_id in seen_ids:
            raise ValueError(f"Duplicate prompt id '{prompt_id}' found in prompt file.")
        seen_ids.add(prompt_id)
        prompts.append(normalized_row)
    return prompts


def normalize_eval_mode(mode_raw: Any) -> str:
    mode = str(mode_raw or "speed").strip().lower()
    if mode not in ALLOWED_EVAL_MODES:
        allowed = ", ".join(sorted(ALLOWED_EVAL_MODES))
        raise ValueError(f"Unsupported eval mode '{mode}'. Allowed: {allowed}")
    return mode


def resolve_lora_path(lora_path_raw: Any, checkpoint_subdir_raw: Any) -> str:
    if lora_path_raw is None:
        raise ValueError("Eval config must include 'lora_path'.")
    lora_path = Path(str(lora_path_raw).strip())

    checkpoint_subdir = str(checkpoint_subdir_raw).strip() if checkpoint_subdir_raw else ""
    if checkpoint_subdir:
        lora_path = lora_path / checkpoint_subdir
    return lora_path.as_posix()


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


def build_output_paths(
    prompt_id: str,
    seed: int,
    baseline_dir: Path,
    lora_dir: Path,
    downscaled_dir: Path,
    grids_dir: Path,
) -> dict[str, Path]:
    return {
        "baseline": baseline_dir / f"{prompt_id}_seed{seed}.png",
        "lora": lora_dir / f"{prompt_id}_seed{seed}.png",
        "baseline_downscaled": downscaled_dir / f"{prompt_id}_seed{seed}_baseline.png",
        "lora_downscaled": downscaled_dir / f"{prompt_id}_seed{seed}_lora.png",
        "grid": grids_dir / f"{prompt_id}_seed{seed}_grid.png",
    }


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


def save_downscaled_and_grid(
    base_img: Image.Image,
    lora_img: Image.Image,
    paths: dict[str, Path],
    downscale_factor: int,
    prompt_id: str,
) -> None:
    base_small = nearest_downscale(base_img, downscale_factor)
    lora_small = nearest_downscale(lora_img, downscale_factor)
    base_small.save(paths["baseline_downscaled"])
    lora_small.save(paths["lora_downscaled"])

    grid = concat_horiz(
        [
            draw_label(base_small, f"{prompt_id} baseline"),
            draw_label(lora_small, f"{prompt_id} lora"),
        ]
    )
    grid.save(paths["grid"])


def build_report_row(
    prompt_row: dict[str, Any],
    seed: int,
    paths: dict[str, Path],
    checkpoint_used: str,
    eval_mode: str,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    lora_scale: float,
) -> dict[str, Any]:
    return {
        "id": prompt_row["id"],
        "category": prompt_row["category"],
        "prompt": prompt_row["prompt"],
        "seed": seed,
        "expected_traits": prompt_row["expected_traits"],
        "failure_checks": prompt_row["failure_checks"],
        "checkpoint_used": checkpoint_used,
        "run_params": {
            "seed": seed,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height,
            "lora_scale": lora_scale,
            "eval_mode": eval_mode,
        },
        "baseline": paths["baseline"].as_posix(),
        "lora": paths["lora"].as_posix(),
        "baseline_downscaled": paths["baseline_downscaled"].as_posix(),
        "lora_downscaled": paths["lora_downscaled"].as_posix(),
        "grid": paths["grid"].as_posix(),
    }


def cleanup_pipeline_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover - optional dependency for eval only
        return


def evaluate_prompts(
    prompts: list[dict[str, Any]],
    base_model: str,
    vae_model: str | None,
    dtype_name: str,
    device: str,
    lora_path: str,
    lora_scale: float,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    downscale_factor: int,
    eval_mode: str,
    baseline_dir: Path,
    lora_dir: Path,
    downscaled_dir: Path,
    grids_dir: Path,
    checkpoint_used: str,
) -> list[dict[str, Any]]:
    report_rows: list[dict[str, Any]] = []

    if eval_mode == "speed":
        baseline_pipe = build_pipeline(base_model, vae_model, dtype_name, device)
        lora_pipe = build_pipeline(
            base_model,
            vae_model,
            dtype_name,
            device,
            lora_path=lora_path,
            lora_scale=lora_scale,
        )

        for row in prompts:
            for seed in row["seeds"]:
                prompt_id = row["id"]
                paths = build_output_paths(
                    prompt_id=prompt_id,
                    seed=seed,
                    baseline_dir=baseline_dir,
                    lora_dir=lora_dir,
                    downscaled_dir=downscaled_dir,
                    grids_dir=grids_dir,
                )
                base_img = generate_image(
                    pipe=baseline_pipe,
                    prompt=row["prompt"],
                    negative_prompt=row["negative_prompt"],
                    seed=seed,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                )
                lora_img = generate_image(
                    pipe=lora_pipe,
                    prompt=row["prompt"],
                    negative_prompt=row["negative_prompt"],
                    seed=seed,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                )
                base_img.save(paths["baseline"])
                lora_img.save(paths["lora"])
                save_downscaled_and_grid(base_img, lora_img, paths, downscale_factor, prompt_id)
                report_rows.append(
                    build_report_row(
                        prompt_row=row,
                        seed=seed,
                        paths=paths,
                        checkpoint_used=checkpoint_used,
                        eval_mode=eval_mode,
                        steps=steps,
                        guidance_scale=guidance_scale,
                        width=width,
                        height=height,
                        lora_scale=lora_scale,
                    )
                )
                print(f"Evaluated {prompt_id} (seed={seed})")

        del baseline_pipe
        del lora_pipe
        cleanup_pipeline_cache()
        return report_rows

    baseline_pipe = build_pipeline(base_model, vae_model, dtype_name, device)
    for row in prompts:
        for seed in row["seeds"]:
            prompt_id = row["id"]
            paths = build_output_paths(
                prompt_id=prompt_id,
                seed=seed,
                baseline_dir=baseline_dir,
                lora_dir=lora_dir,
                downscaled_dir=downscaled_dir,
                grids_dir=grids_dir,
            )
            base_img = generate_image(
                pipe=baseline_pipe,
                prompt=row["prompt"],
                negative_prompt=row["negative_prompt"],
                seed=seed,
                steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
            )
            base_img.save(paths["baseline"])
            print(f"Generated baseline {prompt_id} (seed={seed})")

    del baseline_pipe
    cleanup_pipeline_cache()

    lora_pipe = build_pipeline(
        base_model,
        vae_model,
        dtype_name,
        device,
        lora_path=lora_path,
        lora_scale=lora_scale,
    )
    for row in prompts:
        for seed in row["seeds"]:
            prompt_id = row["id"]
            paths = build_output_paths(
                prompt_id=prompt_id,
                seed=seed,
                baseline_dir=baseline_dir,
                lora_dir=lora_dir,
                downscaled_dir=downscaled_dir,
                grids_dir=grids_dir,
            )
            lora_img = generate_image(
                pipe=lora_pipe,
                prompt=row["prompt"],
                negative_prompt=row["negative_prompt"],
                seed=seed,
                steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
            )
            lora_img.save(paths["lora"])
            with Image.open(paths["baseline"]) as baseline_saved:
                base_img = baseline_saved.convert("RGB")
            save_downscaled_and_grid(base_img, lora_img, paths, downscale_factor, prompt_id)
            report_rows.append(
                build_report_row(
                    prompt_row=row,
                    seed=seed,
                    paths=paths,
                    checkpoint_used=checkpoint_used,
                    eval_mode=eval_mode,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    lora_scale=lora_scale,
                )
            )
            print(f"Evaluated {prompt_id} (seed={seed})")

    del lora_pipe
    cleanup_pipeline_cache()
    return report_rows


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
    lora_path = resolve_lora_path(config.get("lora_path"), config.get("checkpoint_subdir"))
    checkpoint_used = str(config.get("checkpoint_subdir") or Path(lora_path).name)
    vae_model = config.get("vae_model")
    dtype_name = str(config.get("dtype", "bf16"))
    device = str(config.get("device", "cuda"))
    width = int(config.get("width", 1024))
    height = int(config.get("height", 1024))
    steps = int(config.get("num_inference_steps", 30))
    guidance_scale = float(config.get("guidance_scale", 7.0))
    downscale_factor = int(config.get("downscale_factor", 8))
    lora_scale = float(config.get("lora_scale", 1.0))
    eval_mode = normalize_eval_mode(config.get("eval_mode", "speed"))
    report_rows = evaluate_prompts(
        prompts=prompts,
        base_model=base_model,
        vae_model=vae_model,
        dtype_name=dtype_name,
        device=device,
        lora_path=lora_path,
        lora_scale=lora_scale,
        width=width,
        height=height,
        steps=steps,
        guidance_scale=guidance_scale,
        downscale_factor=downscale_factor,
        eval_mode=eval_mode,
        baseline_dir=baseline_dir,
        lora_dir=lora_dir,
        downscaled_dir=downscaled_dir,
        grids_dir=grids_dir,
        checkpoint_used=checkpoint_used,
    )

    report_path = output_dir / "report.json"
    config_with_runtime = dict(config)
    config_with_runtime["eval_mode"] = eval_mode
    config_with_runtime["resolved_lora_path"] = lora_path
    report = {
        "created_at": timestamp_utc(),
        "config": config_with_runtime,
        "checkpoint_used": checkpoint_used,
        "results": report_rows,
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
