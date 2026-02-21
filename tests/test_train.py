from pathlib import Path

import pytest

from pixelart.train import (
    build_command,
    find_latest_checkpoint,
    prepare_resume_checkpoint,
    validate_resume_vs_max_steps,
)


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


def test_prepare_resume_checkpoint_stages_external_checkpoint(tmp_path: Path) -> None:
    source_ckpt = tmp_path / "previous_run" / "checkpoint-8000"
    source_ckpt.mkdir(parents=True)
    (source_ckpt / "state.txt").write_text("ok", encoding="utf-8")

    output_dir = tmp_path / "new_run"
    config = {
        "output_dir": output_dir.as_posix(),
        "resume_from_checkpoint": source_ckpt.as_posix(),
    }

    prepared = prepare_resume_checkpoint(config)

    assert prepared["resume_from_checkpoint"] == "checkpoint-8000"
    assert (output_dir / "checkpoint-8000" / "state.txt").exists()


def test_prepare_resume_checkpoint_raises_for_missing_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing" / "checkpoint-8000"
    config = {
        "output_dir": (tmp_path / "new_run").as_posix(),
        "resume_from_checkpoint": missing.as_posix(),
    }

    with pytest.raises(FileNotFoundError):
        prepare_resume_checkpoint(config)


def test_validate_resume_vs_max_steps_raises_when_target_not_above_resume() -> None:
    config = {
        "resume_from_checkpoint": "checkpoint-8000",
        "max_train_steps": 1500,
    }

    with pytest.raises(ValueError, match="max_train_steps"):
        validate_resume_vs_max_steps(config)
