from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from pixelart.data_clean import clean_dataset, edge_softness_score
from pixelart.io_utils import read_jsonl


def test_edge_softness_detects_blur() -> None:
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :32, :] = 255
    sharp = Image.fromarray(arr)
    blurred = sharp.filter(ImageFilter.GaussianBlur(radius=2))

    sharp_score = edge_softness_score(sharp)
    blurred_score = edge_softness_score(blurred)
    assert blurred_score > sharp_score


def test_clean_dataset_rejects_duplicate(tmp_path: Path) -> None:
    img = Image.new("RGB", (64, 64), (255, 0, 0))
    source_a = tmp_path / "a.png"
    source_b = tmp_path / "b.png"
    img.save(source_a)
    img.save(source_b)

    output_dir = tmp_path / "clean_images"
    index_out = tmp_path / "index.jsonl"
    rejects_out = tmp_path / "rejects.jsonl"
    records = [
        {"file_path": source_a.as_posix(), "source": "x", "license": "CC0-1.0"},
        {"file_path": source_b.as_posix(), "source": "x", "license": "CC0-1.0"},
    ]

    result = clean_dataset(
        records=records,
        output_dir=output_dir,
        index_out=index_out,
        rejects_out=rejects_out,
        resolution=128,
        phash_threshold=0,
        edge_softness_threshold=1.0,
    )
    assert result.kept == 1
    assert result.rejected == 1

    kept_rows = read_jsonl(index_out)
    reject_rows = read_jsonl(rejects_out)
    assert len(kept_rows) == 1
    assert reject_rows[0]["reason"] == "duplicate"


def test_clean_dataset_rejects_by_name_pattern(tmp_path: Path) -> None:
    sheet = Image.new("RGB", (128, 128), (0, 255, 0))
    sheet_path = tmp_path / "enemy_sheet.png"
    sheet.save(sheet_path)

    result = clean_dataset(
        records=[{"file_path": sheet_path.as_posix(), "source": "x", "license": "CC0-1.0"}],
        output_dir=tmp_path / "clean_images",
        index_out=tmp_path / "index.jsonl",
        rejects_out=tmp_path / "rejects.jsonl",
        resolution=128,
        phash_threshold=0,
        edge_softness_threshold=1.0,
        reject_name_patterns=["sheet"],
    )
    reject_rows = read_jsonl(tmp_path / "rejects.jsonl")
    assert result.kept == 0
    assert reject_rows[0]["reason"] == "likely_sprite_sheet_name"


def test_clean_dataset_rejects_large_source_dimensions(tmp_path: Path) -> None:
    img = Image.new("RGB", (512, 512), (200, 100, 50))
    source_path = tmp_path / "large.png"
    img.save(source_path)

    result = clean_dataset(
        records=[{"file_path": source_path.as_posix(), "source": "x", "license": "CC0-1.0"}],
        output_dir=tmp_path / "clean_images2",
        index_out=tmp_path / "index2.jsonl",
        rejects_out=tmp_path / "rejects2.jsonl",
        resolution=128,
        phash_threshold=0,
        edge_softness_threshold=1.0,
        max_source_dimension=384,
    )
    reject_rows = read_jsonl(tmp_path / "rejects2.jsonl")
    assert result.kept == 0
    assert reject_rows[0]["reason"] == "source_too_large"
