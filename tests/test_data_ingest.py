import io
import urllib.error
import zipfile
from pathlib import Path

from PIL import Image

from pixelart.data_ingest import extract_download_urls_from_html, ingest_sources, resolve_url_source


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
                "domain_tag": "icon_focus",
            }
        ],
        output_dir=tmp_path / "raw",
        allow_non_open=False,
    )

    assert len(records) == 1
    assert Path(records[0]["file_path"]).exists()
    assert records[0]["license"] == "CC0-1.0"
    assert records[0]["domain_tag"] == "icon_focus"


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


def test_extract_download_urls_from_html_prioritizes_asset_paths() -> None:
    html = """
    <html>
      <body>
        <a href="/content/some-page">page</a>
        <a href="/sites/default/files/pack.zip">zip</a>
        <a href="https://cdn.example.com/icon.png">icon</a>
        <a href="/sites/default/files/pack.zip">dup</a>
      </body>
    </html>
    """
    urls = extract_download_urls_from_html(html, "https://opengameart.org/content/sample")
    assert urls[0] == "https://opengameart.org/sites/default/files/pack.zip"
    assert urls.count("https://opengameart.org/sites/default/files/pack.zip") == 1
    assert "https://cdn.example.com/icon.png" in urls


def test_resolve_url_source_retries_fallback_urls(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []

    def png_bytes() -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), (255, 0, 0)).save(buffer, format="PNG")
        return buffer.getvalue()

    def fake_download_to(url: str, destination: Path, timeout: int = 60) -> tuple[Path, str]:
        del timeout
        calls.append(url)
        if "primary.example" in url:
            raise urllib.error.HTTPError(url, 404, "not found", None, None)
        if "backup.example" in url:
            with zipfile.ZipFile(destination, "w") as archive:
                archive.writestr("sprite.png", png_bytes())
            return destination, "application/zip"
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("pixelart.data_ingest.download_to", fake_download_to)
    images = resolve_url_source(
        "https://primary.example/missing.zip",
        tmp_path,
        fallback_urls=["https://backup.example/ok.zip"],
    )

    assert calls[0] == "https://primary.example/missing.zip"
    assert "https://backup.example/ok.zip" in calls
    assert len(images) == 1
    assert images[0].name == "sprite.png"


def test_resolve_url_source_extracts_asset_link_from_html_page(tmp_path: Path, monkeypatch) -> None:
    page_url = "https://example.org/content/corrective-pack"
    asset_url = "https://example.org/sites/default/files/corrective-pack.zip"
    calls: list[str] = []

    def png_bytes() -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), (0, 255, 0)).save(buffer, format="PNG")
        return buffer.getvalue()

    def fake_download_to(url: str, destination: Path, timeout: int = 60) -> tuple[Path, str]:
        del timeout
        calls.append(url)
        if url == page_url:
            destination.write_text(f'<a href="{asset_url}">download</a>', encoding="utf-8")
            return destination, "text/html"
        if url == asset_url:
            with zipfile.ZipFile(destination, "w") as archive:
                archive.writestr("sheet.png", png_bytes())
            return destination, "application/zip"
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("pixelart.data_ingest.download_to", fake_download_to)
    images = resolve_url_source(page_url, tmp_path)

    assert calls == [page_url, asset_url]
    assert len(images) == 1
    assert images[0].name == "sheet.png"


def test_ingest_sources_can_skip_failed_sources(tmp_path: Path) -> None:
    src_dir = tmp_path / "good_src"
    src_dir.mkdir()
    Image.new("RGB", (16, 16), (20, 30, 40)).save(src_dir / "ok.png")

    records = ingest_sources(
        sources=[
            {
                "id": "missing_source",
                "local_path": (tmp_path / "does_not_exist").as_posix(),
                "license": "CC0-1.0",
                "attribution": "test",
            },
            {
                "id": "good_source",
                "local_path": src_dir.as_posix(),
                "license": "CC0-1.0",
                "attribution": "test",
            },
        ],
        output_dir=tmp_path / "raw_mixed",
        allow_non_open=False,
        skip_failed_sources=True,
    )

    assert len(records) == 1
    assert records[0]["source"] == "good-source"
