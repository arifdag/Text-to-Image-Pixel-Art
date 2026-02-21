from __future__ import annotations

import argparse
import csv
import json
import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .io_utils import ensure_dir, list_image_files, slugify, timestamp_utc, write_jsonl
from .licenses import is_open_license

try:
    import yaml
except ImportError:  # pragma: no cover - handled at runtime if YAML is used
    yaml = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest image sources and build a raw index.")
    parser.add_argument("--config", type=Path, default=Path("configs/data_sources.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--index-out", type=Path, default=Path("data/raw/index.jsonl"))
    parser.add_argument("--registry-path", type=Path, default=Path("DATA_SOURCES.md"))
    parser.add_argument("--allow-non-open", action="store_true")
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
        raise ValueError("Config root must be a mapping/object.")
    return data


def extract_archive(archive_path: Path, target_dir: Path) -> list[Path]:
    ensure_dir(target_dir)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(target_dir)
    return list_image_files(target_dir)


def download_to(url: str, destination: Path) -> Path:
    ensure_dir(destination.parent)
    urllib.request.urlretrieve(url, destination)
    return destination


def resolve_source_images(source: dict[str, Any], workspace_tmp: Path) -> list[Path]:
    local_path = source.get("local_path")
    url = source.get("url")
    if local_path:
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"local_path does not exist: {local_path}")
        if path.is_dir():
            return list_image_files(path)
        if zipfile.is_zipfile(path):
            return extract_archive(path, workspace_tmp / "extracted")
        return [path] if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"} else []
    if url:
        filename = Path(urllib.parse.urlparse(url).path).name or "downloaded_source"
        downloaded_path = download_to(url, workspace_tmp / filename)
        if zipfile.is_zipfile(downloaded_path):
            return extract_archive(downloaded_path, workspace_tmp / "extracted")
        return (
            [downloaded_path]
            if downloaded_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
            else []
        )
    raise ValueError("Each source entry needs either 'local_path' or 'url'.")


def load_caption_map(source: dict[str, Any]) -> dict[str, str]:
    captions_path_raw = source.get("captions_path")
    if not captions_path_raw:
        return {}
    captions_path = Path(str(captions_path_raw))
    if not captions_path.exists():
        raise FileNotFoundError(f"captions_path does not exist: {captions_path}")

    key_field = str(source.get("captions_key_field", "file_name"))
    text_field = str(source.get("captions_text_field", "text"))
    mapping: dict[str, str] = {}

    if captions_path.suffix.lower() == ".csv":
        with captions_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                key = str(row.get(key_field, "")).strip()
                text = str(row.get(text_field, "")).strip()
                if key and text:
                    mapping[key] = text
                    mapping[Path(key).name] = text
    else:
        with captions_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = str(row.get(key_field, "")).strip()
                text = str(row.get(text_field, "")).strip()
                if key and text:
                    mapping[key] = text
                    mapping[Path(key).name] = text
    return mapping


def append_registry_rows(registry_path: Path, sources: list[dict[str, Any]]) -> None:
    header = [
        "# Data Sources",
        "",
        "| source_id | license | attribution | notes |",
        "|---|---|---|---|",
    ]
    if not registry_path.exists():
        registry_path.write_text("\n".join(header) + "\n", encoding="utf-8")
    existing = registry_path.read_text(encoding="utf-8")
    existing_ids = {line.split("|")[1].strip() for line in existing.splitlines() if line.startswith("|")}

    lines_to_add: list[str] = []
    for source in sources:
        source_id = str(source["id"])
        if source_id in existing_ids:
            continue
        license_name = str(source["license"]).replace("|", "/")
        attribution = str(source.get("attribution", "n/a")).replace("|", "/")
        notes = str(source.get("notes", "")).replace("|", "/")
        lines_to_add.append(f"| {source_id} | {license_name} | {attribution} | {notes} |")
    if lines_to_add:
        with registry_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines_to_add) + "\n")


def ingest_sources(
    sources: list[dict[str, Any]],
    output_dir: Path,
    allow_non_open: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in sources:
        source_id = slugify(str(source.get("id", "source")))
        license_name = str(source.get("license", "")).strip()
        attribution = str(source.get("attribution", "n/a")).strip()
        domain_tag = str(source.get("domain_tag", "")).strip()
        max_files = int(source.get("max_files", 0))

        if not allow_non_open and not is_open_license(license_name):
            raise ValueError(
                f"Source '{source_id}' uses non-open/unknown license '{license_name}'. "
                "Set --allow-non-open to bypass."
            )

        source_output_dir = ensure_dir(output_dir / source_id)
        caption_map = load_caption_map(source)
        with TemporaryDirectory(prefix=f"ingest_{source_id}_") as tmp:
            resolved_images = resolve_source_images(source, Path(tmp))
            if max_files > 0:
                resolved_images = resolved_images[:max_files]

            for idx, src_path in enumerate(resolved_images):
                extension = src_path.suffix.lower() or ".png"
                filename = f"{source_id}_{idx:06d}{extension}"
                destination = source_output_dir / filename
                shutil.copy2(src_path, destination)
                records.append(
                    {
                        "file_path": destination.as_posix(),
                        "source": source_id,
                        "license": license_name,
                        "attribution": attribution,
                        "domain_tag": domain_tag,
                        "source_ref": source.get("url") or source.get("local_path", ""),
                        "ingested_at": timestamp_utc(),
                        "prompt": caption_map.get(src_path.name, ""),
                    }
                )
    return records


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    sources = config.get("sources", [])
    if not isinstance(sources, list) or not sources:
        raise ValueError("Config must include a non-empty 'sources' list.")

    ensure_dir(args.output_dir)
    records = ingest_sources(
        sources=sources,
        output_dir=args.output_dir,
        allow_non_open=args.allow_non_open,
    )
    write_jsonl(args.index_out, records)
    append_registry_rows(args.registry_path, sources)
    print(f"Ingested {len(records)} files to {args.output_dir}")
    print(f"Index: {args.index_out}")
    print(f"Registry: {args.registry_path}")


if __name__ == "__main__":
    main()
