from pathlib import Path

from PIL import Image

from pixelart.data_ingest import ingest_sources


def test_ingest_sources_copies_images(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    Image.new("RGB", (16, 16), (1, 2, 3)).save(src_dir / "one.png")

    records = ingest_sources(
        sources=[
            {
                "id": "demo",
                "local_path": src_dir.as_posix(),
                "license": "CC0-1.0",
                "attribution": "test",
            }
        ],
        output_dir=tmp_path / "raw",
        allow_non_open=False,
    )

    assert len(records) == 1
    assert Path(records[0]["file_path"]).exists()
    assert records[0]["license"] == "CC0-1.0"


def test_ingest_sources_uses_caption_sidecar(tmp_path: Path) -> None:
    src_dir = tmp_path / "src_with_caps"
    src_dir.mkdir()
    image_name = "hero_01.png"
    Image.new("RGB", (16, 16), (10, 20, 30)).save(src_dir / image_name)

    captions = tmp_path / "captions.jsonl"
    captions.write_text(
        '{"file_name":"hero_01.png","text":"pixel art hero knight sprite"}\n',
        encoding="utf-8",
    )

    records = ingest_sources(
        sources=[
            {
                "id": "demo_caps",
                "local_path": src_dir.as_posix(),
                "license": "CC0-1.0",
                "attribution": "test",
                "captions_path": captions.as_posix(),
            }
        ],
        output_dir=tmp_path / "raw_caps",
        allow_non_open=False,
    )

    assert len(records) == 1
    assert records[0]["prompt"] == "pixel art hero knight sprite"
