from pathlib import Path

from pixelart.eval import build_output_paths, build_report_row


def test_build_report_row_contains_category_and_run_params(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    lora_dir = tmp_path / "lora"
    downscaled_dir = tmp_path / "downscaled"
    grids_dir = tmp_path / "grids"
    paths = build_output_paths(
        prompt_id="character_test",
        seed=777,
        baseline_dir=baseline_dir,
        lora_dir=lora_dir,
        downscaled_dir=downscaled_dir,
        grids_dir=grids_dir,
    )
    row = build_report_row(
        prompt_row={
            "id": "character_test",
            "category": "detail_readability",
            "prompt": "pixel art, test",
            "expected_traits": ["clear silhouette"],
            "failure_checks": ["detail collapse"],
        },
        seed=777,
        paths=paths,
        checkpoint_used="checkpoint-8000",
        eval_mode="memory_safe",
        steps=30,
        guidance_scale=7.0,
        width=1024,
        height=1024,
        lora_scale=0.65,
    )

    assert row["id"] == "character_test"
    assert row["category"] == "detail_readability"
    assert row["checkpoint_used"] == "checkpoint-8000"
    assert row["run_params"]["eval_mode"] == "memory_safe"
    assert row["run_params"]["lora_scale"] == 0.65
    assert row["expected_traits"] == ["clear silhouette"]
    assert row["failure_checks"] == ["detail collapse"]
    assert row["baseline"].endswith("character_test_seed777.png")
    assert row["grid"].endswith("character_test_seed777_grid.png")
