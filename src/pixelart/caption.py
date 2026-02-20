from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from .io_utils import ensure_dir, read_jsonl, timestamp_utc, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create training captions and metadata.jsonl.")
    parser.add_argument("--index-in", type=Path, default=Path("data/clean/index.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--metadata-out", type=Path, default=Path("data/train/metadata.jsonl"))
    parser.add_argument(
        "--style-tags",
        type=str,
        default="pixel art,sprite,limited palette,crisp edges,retro game style",
    )
    parser.add_argument("--use-blip", action="store_true")
    parser.add_argument("--blip-model", type=str, default="Salesforce/blip2-opt-2.7b")
    parser.add_argument("--prefer-source-prompt", action="store_true")
    return parser.parse_args()


def tags_from_string(tags: str) -> list[str]:
    cleaned = [tag.strip() for tag in tags.split(",")]
    return [tag for tag in cleaned if tag]


def caption_from_filename(path: Path) -> str:
    name = path.stem.replace("_", " ").replace("-", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name or "character sprite"


def unique_join(parts: list[str]) -> str:
    seen: set[str] = set()
    ordered: list[str] = []
    for part in parts:
        norm = " ".join(part.strip().lower().split())
        if not norm or norm in seen:
            continue
        seen.add(norm)
        ordered.append(part.strip())
    return ", ".join(ordered)


class BlipCaptioner:
    def __init__(self, model_name: str) -> None:
        import torch
        from transformers import AutoProcessor, Blip2ForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_name)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        if not torch.cuda.is_available():
            self.model = self.model.to("cpu")

    def generate(self, image: Image.Image) -> str:
        import torch

        inputs = self.processor(images=image, return_tensors="pt")
        device = self.model.device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=40)
        return self.processor.batch_decode(output, skip_special_tokens=True)[0].strip()


def create_metadata_rows(
    records: list[dict[str, Any]],
    output_dir: Path,
    style_tags: list[str],
    use_blip: bool,
    blip_model: str,
    prefer_source_prompt: bool,
) -> list[dict[str, Any]]:
    images_out = ensure_dir(output_dir / "images")
    captioner = BlipCaptioner(blip_model) if use_blip else None
    rows: list[dict[str, Any]] = []

    for idx, record in enumerate(records):
        source_path = Path(str(record["file_path"]))
        if not source_path.exists():
            continue

        image_name = f"train_{idx:06d}.png"
        destination = images_out / image_name
        shutil.copy2(source_path, destination)

        caption_parts: list[str] = []
        prompt = str(record.get("prompt", "")).strip()
        if prefer_source_prompt and prompt:
            caption_parts.append(prompt)
        elif captioner is not None:
            with Image.open(destination) as img:
                caption_parts.append(captioner.generate(img.convert("RGB")))
        elif prompt:
            caption_parts.append(prompt)
        else:
            caption_parts.append(caption_from_filename(source_path))

        if record.get("domain_tag"):
            caption_parts.append(str(record["domain_tag"]))
        caption_parts.extend(style_tags)

        rows.append(
            {
                "file_name": f"images/{image_name}",
                "text": unique_join(caption_parts),
                "license": record.get("license", "unknown"),
                "source": record.get("source", "unknown"),
                "attribution": record.get("attribution", "n/a"),
                "created_at": timestamp_utc(),
            }
        )

    return rows


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.index_in)
    if not records:
        raise ValueError(f"No input records found in {args.index_in}")

    style_tags = tags_from_string(args.style_tags)
    ensure_dir(args.output_dir)

    rows = create_metadata_rows(
        records=records,
        output_dir=args.output_dir,
        style_tags=style_tags,
        use_blip=args.use_blip,
        blip_model=args.blip_model,
        prefer_source_prompt=args.prefer_source_prompt,
    )
    write_jsonl(args.metadata_out, rows)
    print(f"Wrote {len(rows)} caption rows to {args.metadata_out}")


if __name__ == "__main__":
    main()

