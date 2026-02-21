# Corrective Run Plan (Checkpoint-8000)

## Goal
Run a targeted continuation from `checkpoint-8000` to fix:
1. Pixel integrity failures (icons collapsing into repeated patterns, soft edges).
2. Prompt-following failures (wrong counts/roles/attributes).
3. Single-subject failures (sprite-sheet style outputs).

## Scope
1. CC0-only corrective dataset (`configs/data_sources_corrective.yaml`).
2. Strict cleaning gates for likely sprite sheets and oversized source files.
3. New corrective training config (`configs/train_sdxl_lora_corrective.yaml`).
4. Focused corrective eval prompts (`configs/prompts_corrective_eval.json`).
5. Corrective eval config (`configs/eval_corrective.yaml`).
6. Colab notebook to run the full pipeline (`notebooks/colab_corrective_ckpt8000.ipynb`).

## Dataset Targets
1. Total ingest cap: ~2,500 images (via `max_files` per source).
2. Mix emphasis:
- `pixel_integrity`
- `prompt_following`
- `composition_clarity`
- `detail_readability`

## Training Defaults
1. Resume point: `/content/drive/MyDrive/pixelart-lora-output/checkpoint-8000`
2. Output dir: `/content/drive/MyDrive/pixelart-lora-output-corrective-ckpt8000`
3. `max_train_steps: 1500`
4. `learning_rate: 3e-5`
5. Keep SDXL pixel-art settings unchanged (`resolution=1024`, `nearest` interpolation).

## Acceptance Criteria
1. Corrective run produces checkpoints and `run_manifest.json`.
2. Corrective eval runs for all seeds/prompts.
3. Failures seen in prior eval are reduced, especially:
- coin/icon repetition collapse
- count/position misses (e.g., exact 3 lineup)
- multi-sprite-sheet style outputs when one subject is requested

## Notes
1. Dataset URLs are OpenGameArt mirror links and may occasionally move.
2. If a source URL fails, replace it with the equivalent archive URL and rerun ingest.
