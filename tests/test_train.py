from pathlib import Path

from pixelart.train import build_command, find_latest_checkpoint


def test_find_latest_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-500").mkdir()
    (tmp_path / "checkpoint-2000").mkdir()
    (tmp_path / "checkpoint-1500").mkdir()
    assert find_latest_checkpoint(tmp_path) == "checkpoint-2000"


def test_build_command_uses_latest_checkpoint(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    (output_dir / "checkpoint-1000").mkdir(parents=True)
    config = {
        "train_script": "train_text_to_image_lora_sdxl.py",
        "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
        "train_data_dir": "data/train",
        "output_dir": output_dir.as_posix(),
        "resolution": 1024,
        "train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "max_train_steps": 100,
        "resume_from_checkpoint": "latest",
    }
    command = build_command(config)
    assert "--resume_from_checkpoint" in command
    assert "checkpoint-1000" in command

