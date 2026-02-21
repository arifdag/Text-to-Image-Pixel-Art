from pathlib import Path

import yaml

from pixelart.eval import load_prompts
from pixelart.licenses import is_open_license


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_data_sources_corrective_contract() -> None:
    config = load_yaml(Path("configs/data_sources_corrective.yaml"))
    sources = config["sources"]
    assert isinstance(sources, list)
    assert len(sources) >= 8

    total_max_files = 0
    has_non_kenney_source = False
    for source in sources:
        assert source.get("url")
        assert is_open_license(str(source.get("license", "")))
        assert source.get("domain_tag") in {
            "prompt_following",
            "pixel_integrity",
            "detail_readability",
            "composition_clarity",
        }
        total_max_files += int(source.get("max_files", 0))
        attribution = str(source.get("attribution", "")).lower()
        if "kenney" not in attribution:
            has_non_kenney_source = True

    assert 2000 <= total_max_files <= 3000
    assert has_non_kenney_source


def test_train_corrective_config_constraints() -> None:
    corrective = load_yaml(Path("configs/train_sdxl_lora_corrective.yaml"))
    baseline = load_yaml(Path("configs/train_sdxl_lora.yaml"))

    assert corrective["train_data_dir"].endswith("data/train_corrective")
    assert "checkpoint-8000" in str(corrective["resume_from_checkpoint"])
    assert int(corrective["max_train_steps"]) > 8000
    assert int(corrective["max_train_steps"]) <= 12000
    assert float(corrective["learning_rate"]) < float(baseline["learning_rate"])
    assert corrective["output_dir"] != baseline["output_dir"]


def test_prompts_corrective_eval_schema() -> None:
    prompts = load_prompts(Path("configs/prompts_corrective_eval.json"))
    assert len(prompts) >= 8
    categories = {row["category"] for row in prompts}
    assert "pixel_integrity" in categories
    assert "prompt_following" in categories
    assert "composition_clarity" in categories


def test_eval_corrective_config_contract() -> None:
    config = load_yaml(Path("configs/eval_corrective.yaml"))
    assert config["prompts_file"] == "configs/prompts_corrective_eval.json"
    assert config["eval_mode"] == "memory_safe"
    assert "pixelart-lora-output-corrective-ckpt8000" in str(config["lora_path"])
    assert str(config["checkpoint_subdir"]).startswith("checkpoint-")
