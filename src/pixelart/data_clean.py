from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .io_utils import ensure_dir, list_image_files, read_jsonl, timestamp_utc, write_jsonl

try:
    import imagehash
except ImportError:  # pragma: no cover - optional fallback path
    imagehash = None


try:
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
    RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # pragma: no cover
    RESAMPLE_NEAREST = Image.NEAREST
    RESAMPLE_BILINEAR = Image.BILINEAR


@dataclass
class CleanResult:
    kept: int
    rejected: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean raw pixel-art dataset for SDXL training.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--index-in", type=Path, default=Path("data/raw/index.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/clean/images"))
    parser.add_argument("--index-out", type=Path, default=Path("data/clean/index.jsonl"))
    parser.add_argument("--rejects-out", type=Path, default=Path("data/clean/rejects.jsonl"))
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--phash-threshold", type=int, default=6)
    parser.add_argument("--edge-softness-threshold", type=float, default=0.55)
    return parser.parse_args()


def center_crop_resize_nearest(image: Image.Image, resolution: int) -> Image.Image:
    rgb = image.convert("RGB")
    width, height = rgb.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    cropped = rgb.crop((left, top, left + side, top + side))
    return cropped.resize((resolution, resolution), RESAMPLE_NEAREST)


def edge_softness_score(image: Image.Image) -> float:
    # Higher score means softer/blurrier edges.
    gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    gx = np.abs(np.diff(gray, axis=1, append=gray[:, -1:]))
    gy = np.abs(np.diff(gray, axis=0, append=gray[-1:, :]))
    grad = gx + gy

    edge_pixels = grad > 0.03
    edge_count = float(edge_pixels.sum())
    if edge_count <= 1.0:
        return 1.0

    soft_edges = np.logical_and(grad > 0.03, grad < 0.12)
    return float(soft_edges.sum() / edge_count)


def perceptual_hash(image: Image.Image) -> Any:
    if imagehash is not None:
        return imagehash.phash(image.convert("RGB"), hash_size=16)
    reduced = image.convert("RGB").resize((64, 64), RESAMPLE_BILINEAR)
    return hashlib.md5(reduced.tobytes()).hexdigest()


def hash_distance(hash_a: Any, hash_b: Any) -> int:
    if imagehash is not None:
        return int(hash_a - hash_b)
    return 0 if hash_a == hash_b else math.inf


def build_input_records(input_dir: Path, index_in: Path) -> list[dict[str, Any]]:
    indexed = read_jsonl(index_in)
    if indexed:
        return indexed
    return [{"file_path": path.as_posix()} for path in list_image_files(input_dir)]


def clean_dataset(
    records: list[dict[str, Any]],
    output_dir: Path,
    index_out: Path,
    rejects_out: Path,
    resolution: int,
    phash_threshold: int,
    edge_softness_threshold: float,
) -> CleanResult:
    ensure_dir(output_dir)
    seen_hashes: list[Any] = []
    kept_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    for i, record in enumerate(records):
        image_path = Path(str(record["file_path"]))
        if not image_path.exists():
            rejected_rows.append(
                {
                    "file_path": image_path.as_posix(),
                    "reason": "missing_file",
                    "timestamp": timestamp_utc(),
                }
            )
            continue

        try:
            with Image.open(image_path) as img:
                softness = edge_softness_score(img)
                if softness > edge_softness_threshold:
                    rejected_rows.append(
                        {
                            **record,
                            "reason": "soft_edges",
                            "edge_softness": round(softness, 6),
                            "timestamp": timestamp_utc(),
                        }
                    )
                    continue

                cleaned = center_crop_resize_nearest(img, resolution)
                p_hash = perceptual_hash(cleaned)
        except Exception as exc:  # pragma: no cover - image decoding edge cases
            rejected_rows.append(
                {
                    **record,
                    "reason": "decode_error",
                    "error": str(exc),
                    "timestamp": timestamp_utc(),
                }
            )
            continue

        is_duplicate = any(hash_distance(p_hash, h) <= phash_threshold for h in seen_hashes)
        if is_duplicate:
            rejected_rows.append(
                {
                    **record,
                    "reason": "duplicate",
                    "timestamp": timestamp_utc(),
                }
            )
            continue

        seen_hashes.append(p_hash)
        filename = f"clean_{i:06d}.png"
        output_path = output_dir / filename
        cleaned.save(output_path, format="PNG")

        kept_rows.append(
            {
                **record,
                "file_path": output_path.as_posix(),
                "edge_softness": round(softness, 6),
                "processed_at": timestamp_utc(),
            }
        )

    write_jsonl(index_out, kept_rows)
    write_jsonl(rejects_out, rejected_rows)
    return CleanResult(kept=len(kept_rows), rejected=len(rejected_rows))


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    ensure_dir(args.index_out.parent)
    ensure_dir(args.rejects_out.parent)

    records = build_input_records(args.input_dir, args.index_in)
    result = clean_dataset(
        records=records,
        output_dir=args.output_dir,
        index_out=args.index_out,
        rejects_out=args.rejects_out,
        resolution=args.resolution,
        phash_threshold=args.phash_threshold,
        edge_softness_threshold=args.edge_softness_threshold,
    )
    print(f"Kept {result.kept} images")
    print(f"Rejected {result.rejected} images")
    print(f"Clean index: {args.index_out}")


if __name__ == "__main__":
    main()
