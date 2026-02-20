from pathlib import Path

from PIL import Image

from pixelart.caption import create_metadata_rows


def test_create_metadata_rows_with_template_caption(tmp_path: Path) -> None:
    img_path = tmp_path / "hero_knight.png"
    Image.new("RGB", (32, 32), (0, 128, 255)).save(img_path)

    rows = create_metadata_rows(
        records=[
            {
                "file_path": img_path.as_posix(),
                "source": "unit",
                "license": "CC0-1.0",
                "attribution": "tester",
            }
        ],
        output_dir=tmp_path / "train",
        style_tags=["pixel art", "sprite"],
        use_blip=False,
        blip_model="unused",
        prefer_source_prompt=False,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["file_name"].startswith("images/")
    assert "pixel art" in row["text"].lower()
    assert row["license"] == "CC0-1.0"

