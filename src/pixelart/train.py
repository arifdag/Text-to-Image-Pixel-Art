from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .io_utils import ensure_dir, timestamp_utc

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime when YAML is required
    yaml = None


ARG_KEYS = [
    "pretrained_model_name_or_path",
    "pretrained_vae_model_name_or_path",
    "train_data_dir",
    "output_dir",
    "resolution",
    "image_interpolation_mode",
    "train_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "max_train_steps",
    "checkpointing_steps",
    "checkpoints_total_limit",
    "rank",
    "seed",
    "snr_gamma",
    "validation_prompt",
    "validation_epochs",
    "mixed_precision",
    "num_train_epochs",
    "lr_scheduler",
    "lr_warmup_steps",
    "num_validation_images",
    "dataloader_num_workers",
    "max_grad_norm",
]

FLAG_KEYS = [
    "gradient_checkpointing",
    "allow_tf32",
    "enable_xformers_memory_efficient_attention",
    "train_text_encoder",
    "use_8bit_adam",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch SDXL LoRA training via Diffusers.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_sdxl_lora.yaml"))
    parser.add_argument("--dry-run", action="store_true")
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
        raise ValueError("Training config must be a mapping/object.")
    return data


def checkpoint_step(path: Path) -> int:
    match = re.match(r"checkpoint-(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def find_latest_checkpoint(output_dir: Path) -> str | None:
    if not output_dir.exists():
        return None
    checkpoints = [p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    if not checkpoints:
        return None
    latest = max(checkpoints, key=checkpoint_step)
    return latest.name


def build_command(config: dict[str, Any]) -> list[str]:
    train_script = str(config.get("train_script", "train_text_to_image_lora_sdxl.py"))
    command = ["accelerate", "launch", train_script]

    missing = [k for k in ["pretrained_model_name_or_path", "train_data_dir", "output_dir"] if not config.get(k)]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    for key in ARG_KEYS:
        value = config.get(key)
        if value is None or value == "":
            continue
        command.extend([f"--{key}", str(value)])

    for key in FLAG_KEYS:
        if bool(config.get(key, False)):
            command.append(f"--{key}")

    resume = str(config.get("resume_from_checkpoint", "")).strip()
    output_dir = Path(str(config["output_dir"]))
    if resume.lower() == "latest":
        latest = find_latest_checkpoint(output_dir)
        if latest:
            command.extend(["--resume_from_checkpoint", latest])
    elif resume:
        command.extend(["--resume_from_checkpoint", resume])

    return command


def write_manifest(output_dir: Path, config: dict[str, Any], command: list[str]) -> Path:
    ensure_dir(output_dir)
    manifest_path = output_dir / "run_manifest.json"
    manifest = {
        "created_at": timestamp_utc(),
        "command": command,
        "config": config,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return manifest_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    command = build_command(config)
    output_dir = Path(str(config["output_dir"]))
    print("Training command:")
    print(" ".join(command))

    manifest_path: Path | None = None
    try:
        manifest_path = write_manifest(output_dir, config, command)
    except OSError as exc:
        if args.dry_run:
            print(f"Warning: could not write manifest to {output_dir}: {exc}")
        else:
            raise
    if manifest_path is not None:
        print(f"Manifest: {manifest_path}")
    if args.dry_run:
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
