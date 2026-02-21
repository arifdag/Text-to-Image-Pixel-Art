import json
from pathlib import Path

import pytest

from pixelart.eval import load_prompts


def write_payload(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_prompt(prompt_id: str) -> dict[str, object]:
    return {
        "id": prompt_id,
        "category": "prompt_following",
        "prompt": "pixel art, test prompt",
        "negative_prompt": "blurry, realistic",
        "seeds": [101, 102],
        "expected_traits": ["trait a"],
        "failure_checks": ["failure a"],
    }


def test_load_prompts_rejects_non_list(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts.json"
    write_payload(prompt_file, {"not": "a list"})
    with pytest.raises(ValueError, match="JSON list"):
        load_prompts(prompt_file)


def test_load_prompts_rejects_missing_required_field(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts.json"
    row = make_prompt("p1")
    row.pop("prompt")
    write_payload(prompt_file, [row])
    with pytest.raises(ValueError, match="non-empty 'prompt'"):
        load_prompts(prompt_file)


def test_load_prompts_rejects_empty_seeds(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts.json"
    row = make_prompt("p1")
    row["seeds"] = []
    write_payload(prompt_file, [row])
    with pytest.raises(ValueError, match="non-empty 'seeds'"):
        load_prompts(prompt_file)


def test_load_prompts_rejects_duplicate_ids(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts.json"
    write_payload(prompt_file, [make_prompt("dup"), make_prompt("dup")])
    with pytest.raises(ValueError, match="Duplicate prompt id"):
        load_prompts(prompt_file)


def test_load_prompts_rejects_invalid_category(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts.json"
    row = make_prompt("p1")
    row["category"] = "random_category"
    write_payload(prompt_file, [row])
    with pytest.raises(ValueError, match="unsupported category"):
        load_prompts(prompt_file)


def test_load_prompts_accepts_legacy_seed_field(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompts.json"
    row = make_prompt("legacy")
    row.pop("seeds")
    row["seed"] = 303
    write_payload(prompt_file, [row])
    rows = load_prompts(prompt_file)
    assert rows[0]["seeds"] == [303]
