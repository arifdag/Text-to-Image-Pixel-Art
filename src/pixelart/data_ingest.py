from __future__ import annotations

import argparse
import csv
import json
import re
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

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
HTML_LIKE_SUFFIXES = {"", ".html", ".htm", ".php", ".asp", ".aspx"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest image sources and build a raw index.")
    parser.add_argument("--config", type=Path, default=Path("configs/data_sources.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--index-out", type=Path, default=Path("data/raw/index.jsonl"))
    parser.add_argument("--registry-path", type=Path, default=Path("DATA_SOURCES.md"))
    parser.add_argument("--allow-non-open", action="store_true")
    parser.add_argument(
        "--skip-failed-sources",
        action="store_true",
        help="Continue ingest even if one or more sources fail to download.",
    )
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


def download_to(url: str, destination: Path, timeout: int = 60) -> tuple[Path, str]:
    ensure_dir(destination.parent)
    with urllib.request.urlopen(url, timeout=timeout) as response:
        destination.write_bytes(response.read())
        content_type = str(response.headers.get("Content-Type", ""))
    return destination, content_type


def extract_download_urls_from_html(html: str, page_url: str) -> list[str]:
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    candidates: list[str] = []
    for href in hrefs:
        absolute = urllib.parse.urljoin(page_url, href.strip())
        path = urllib.parse.urlparse(absolute).path.lower()
        if path.endswith(".zip") or any(path.endswith(ext) for ext in IMAGE_EXTENSIONS):
            candidates.append(absolute)

    preferred: list[str] = []
    others: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        path = urllib.parse.urlparse(candidate).path.lower()
        if "/sites/default/files/" in path:
            preferred.append(candidate)
        else:
            others.append(candidate)
    return preferred + others


def dedupe_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for url in urls:
        normalized = str(url).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def derive_mirror_urls(url: str) -> list[str]:
    parsed = urllib.parse.urlparse(url)
    if parsed.netloc == "lpc.opengameart.org":
        return [parsed._replace(netloc="opengameart.org").geturl()]
    if parsed.netloc == "opengameart.org":
        return [parsed._replace(netloc="lpc.opengameart.org").geturl()]
    return []


def resolve_url_source(url: str, workspace_tmp: Path, fallback_urls: list[str] | None = None) -> list[Path]:
    seed_urls = [url]
    if fallback_urls:
        seed_urls.extend(fallback_urls)
    mirror_urls: list[str] = []
    for candidate in seed_urls:
        mirror_urls.extend(derive_mirror_urls(candidate))
    candidate_urls = dedupe_urls(seed_urls + mirror_urls)

    errors: list[str] = []
    for candidate_url in candidate_urls:
        parsed = urllib.parse.urlparse(candidate_url)
        filename = Path(parsed.path).name or "downloaded_source"

        try:
            downloaded_path, content_type = download_to(candidate_url, workspace_tmp / filename)
        except Exception as exc:
            errors.append(f"{candidate_url} -> {type(exc).__name__}: {exc}")
            continue

        if zipfile.is_zipfile(downloaded_path):
            return extract_archive(downloaded_path, workspace_tmp / "extracted")
        if downloaded_path.suffix.lower() in IMAGE_EXTENSIONS:
            return [downloaded_path]

        page_suffix = downloaded_path.suffix.lower()
        is_html_like = "html" in content_type.lower() or page_suffix in HTML_LIKE_SUFFIXES
        if not is_html_like:
            errors.append(f"{candidate_url} -> unsupported content type '{content_type or page_suffix}'")
            continue

        html = downloaded_path.read_text(encoding="utf-8", errors="ignore")
        asset_links = extract_download_urls_from_html(html, candidate_url)
        if not asset_links:
            errors.append(f"{candidate_url} -> no downloadable asset links found")
            continue

        for asset_url in asset_links:
            asset_name = Path(urllib.parse.urlparse(asset_url).path).name or "asset_download"
            try:
                asset_path, _ = download_to(asset_url, workspace_tmp / asset_name)
            except Exception:
                continue
            if zipfile.is_zipfile(asset_path):
                return extract_archive(asset_path, workspace_tmp / "extracted")
            if asset_path.suffix.lower() in IMAGE_EXTENSIONS:
                return [asset_path]
        errors.append(f"{candidate_url} -> asset links resolved but none downloaded successfully")

    details = "; ".join(errors[:3])
    if len(errors) > 3:
        details = f"{details}; ... {len(errors) - 3} more attempts"
    raise ValueError(
        f"Could not resolve downloadable image/archive from source URL: {url}"
        + (f" ({details})" if details else "")
    )


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
        return [path] if path.suffix.lower() in IMAGE_EXTENSIONS else []
    if url:
        fallback_urls_raw = source.get("fallback_urls", [])
        if isinstance(fallback_urls_raw, list):
            fallback_urls = [str(item).strip() for item in fallback_urls_raw if str(item).strip()]
        elif fallback_urls_raw:
            fallback_urls = [str(fallback_urls_raw).strip()]
        else:
            fallback_urls = []
        return resolve_url_source(str(url), workspace_tmp, fallback_urls=fallback_urls)
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
    skip_failed_sources: bool = False,
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
            try:
                resolved_images = resolve_source_images(source, Path(tmp))
            except Exception as exc:
                if skip_failed_sources:
                    print(f"[WARN] Skipping source '{source_id}' due to ingest error: {exc}")
                    continue
                raise RuntimeError(f"Failed to ingest source '{source_id}': {exc}") from exc
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
        skip_failed_sources=args.skip_failed_sources,
    )
    write_jsonl(args.index_out, records)
    append_registry_rows(args.registry_path, sources)
    print(f"Ingested {len(records)} files to {args.output_dir}")
    print(f"Index: {args.index_out}")
    print(f"Registry: {args.registry_path}")


if __name__ == "__main__":
    main()
