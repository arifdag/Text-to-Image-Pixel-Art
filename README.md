# Text-to-Image Pixel Art Generator

Colab-first SDXL LoRA pipeline for generating crisp pixel-art sprites from text prompts under a single A100 time budget (<20 hours).

## What this repo contains
- `src/pixelart/`: ingestion, cleaning, captioning, training launcher, and eval CLIs.
- `configs/`: default config for data sources, SDXL LoRA training, and eval prompt suite.
- `notebooks/colab_train_pixel_art.ipynb`: end-to-end Colab workflow.
- `notebooks/colab_corrective_ckpt8000.ipynb`: targeted corrective continuation workflow.
- `tests/`: unit tests for data cleaning + reproducibility helpers.
- `DATA_SOURCES.md`: source/attribution register.

## Quick start
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

Run the full local pipeline shape (small sample data):
```powershell
python -m pixelart.data_ingest --config configs/data_sources.yaml
python -m pixelart.data_clean --input-dir data/raw --index-in data/raw/index.jsonl
python -m pixelart.caption --index-in data/clean/index.jsonl --prefer-source-prompt
python -m pixelart.train --config configs/train_sdxl_lora.yaml --dry-run
python -m pixelart.eval --config configs/eval.yaml
```

Corrective continuation pipeline (checkpoint-8000 focused):
```powershell
python -m pixelart.data_ingest --config configs/data_sources_corrective.yaml --output-dir data/raw_corrective --index-out data/raw_corrective/index.jsonl
python -m pixelart.data_clean --input-dir data/raw_corrective --index-in data/raw_corrective/index.jsonl --output-dir data/clean_corrective/images --index-out data/clean_corrective/index.jsonl --rejects-out data/clean_corrective/rejects.jsonl --max-source-dimension 384 --max-aspect-ratio 1.8 --reject-name-patterns "sheet,spritesheet,sprite_sheet,atlas,tilemap,tilesheet"
python -m pixelart.caption --index-in data/clean_corrective/index.jsonl --output-dir data/train_corrective --metadata-out data/train_corrective/metadata.jsonl --prefer-source-prompt
python -m pixelart.train --config configs/train_sdxl_lora_corrective.yaml --dry-run
python -m pixelart.eval --config configs/eval_corrective.yaml
```

## Core SDXL pixel-art settings
- Train at `1024x1024`.
- Use `--image_interpolation_mode=nearest`.
- Generate at 1024, then downscale by `8x` (nearest-neighbor) for 128px output.
- Keep SDXL refiner off for pixel integrity.

## Reproducibility and acceptance
- Prompt benchmark suite + fixed seeds are in `configs/prompts_v2.json`.
- Eval outputs save baseline vs LoRA side-by-side grids in `artifacts/eval/grids/`.
- Eval supports `eval_mode: speed|memory_safe` in `configs/eval.yaml`.
- Training launcher writes `run_manifest.json` to output dir.
