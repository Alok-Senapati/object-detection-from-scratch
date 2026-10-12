# Project 03 — Object Detection from Scratch, Transfer Learning & Fine-Tuning

Build and understand a minimal detector before studying representation reuse. Track A implements geometry, assignment, decoding, losses, NMS and AP. Track B retains an understandable head and compares scratch, frozen, partial and full backbone training. Detection-pretrained transfer is a separate reference, with its structural and data advantages disclosed.

## Setup — Python 3.12 / uv

Use `main` for setup and practice. For an existing checkout, update it first:

```bash
git switch main
git pull --ff-only origin main
```

```bash
git clone --branch main https://github.com/Alok-Senapati/object-detection-from-scratch.git
cd object-detection-from-scratch
uv sync --locked --extra cuda   # Windows/WSL2 NVIDIA laptop
# CPU-only machine or CI: uv sync --locked --extra cpu
uv run --no-sync od-diagnose
uv run --no-sync od-smoke
uv run --no-sync python scripts/check.py
uv run --no-sync tensorboard --logdir artifacts
```

Select exactly one of `cpu` or `cuda`. Use `--no-sync` on subsequent commands so a default sync does not replace the selected wheel variant. CPU sync on macOS uses the official CPU wheel and can select MPS when available; untested here. Optional notebooks: add `--extra notebooks` to your chosen sync command.

PyTorch 2.10.0 / torchvision 0.25.0 are deliberate compatible baseline pins, not a claim of newest versions. CUDA wheels use cu128; a matching local CUDA toolkit is not needed for wheel execution, but a compatible NVIDIA driver is. Windows and WSL installations are separate. Verify the actual device before experiments. Installation source: https://pytorch.org/get-started/previous-versions/

## Verified local setup

User-provided Windows validation on 2026-10-12: Python 3.12.13, PyTorch 2.10.0+cu128, CUDA runtime 12.8, RTX 3060 Laptop GPU (6143.5 MiB). CUDA diagnosis and smoke passed; Ruff, formatting, strict Mypy and all 9 tests passed. TensorBoard served at http://localhost:6006/. See [validation evidence](docs/SETUP_VALIDATION.md) for scope and remaining checks.

If uv reports that hardlinking is unavailable, installation still succeeds using copies. In PowerShell, use `uv sync --locked --extra cuda --link-mode=copy` to select copy mode explicitly. This affects package installation/storage, not model CUDA availability. TensorBoard can display our PyTorch scalar logs without installing TensorFlow.

## Layout and ownership

| Path | Responsibility |
|---|---|
| `src/od_scratch/infra/` | Fully implemented generic experiment utilities |
| `src/od_scratch/data/` | Future VOC ingestion and detection-aware transforms |
| `src/od_scratch/geometry/` | Your box conversions and overlap implementations |
| `src/od_scratch/models/` | Your backbone/head/decoding, later official backbones |
| `src/od_scratch/losses/` | Your assignment and loss implementations |
| `src/od_scratch/evaluation/` | Your NMS/AP plus independent evaluator checks |
| `src/od_scratch/training/` | Future detection training orchestration |
| `configs/`, `scripts/`, `tests/` | Validated settings, checks and contract tests |
| `notebooks/`, `docs/` | Exercises, taskbook, protocols and results |
| `data/`, `artifacts/` | Local ignored datasets and outputs |

No detector architecture, training loop, CIFAR loader, portal or trained weights are included. See [reuse audit](docs/REUSE_AUDIT.md), [taskbook](docs/MENTORING_TASKBOOK.md), [Stage 1](docs/STAGE_01.md), [experiment log](docs/EXPERIMENT_LOG.md), [transfer protocol](docs/TRANSFER_LEARNING_EXPERIMENTS.md) and [validation](docs/SETUP_VALIDATION.md).

## YOLO fine-tuning

Stage 26 includes an official YOLO26n fine-tuning reference, progressing through head-only, partial and full updates; YOLO11n provides a mature comparison screen. Keep A-D on the understandable detector and treat YOLO as a separate detection-pretrained E benchmark. See [YOLO protocol](docs/YOLO_FINE_TUNING.md). No YOLO dependency or weights are installed until this stage.

## Infrastructure contracts

- `seed_everything` returns a generator; use it with `seed_worker` in DataLoader. Call before model/loader/CUDA setup. Set `PYTHONHASHSEED=42` **before launching Python** if hash randomization matters. Strict deterministic kernels may fail rather than silently fall back. Reproducibility across hardware/versions is not guaranteed.
- `Experiment` records config, git SHA/dirty state, versions, JSONL and TensorBoard scalars in a unique directory. Save split manifests, class mappings and weight provenance there before any real training.
- `Timer(device)` synchronizes CUDA/MPS. Warm up before benchmarks. VRAM statistics are PyTorch CUDA allocator values; CPU/MPS returns unavailable, not zero.
- Checkpoint helpers store state dictionaries atomically and deserialize with `weights_only=True`. Supply an architecture explicitly. Caller owns optimizer/scheduler/scaler/RNG/loader restoration; exact resume is not implemented. `strict=False` does not permit shape mismatch.
- Optimizer factories accept parameter groups. Cosine uses the actual epoch budget. Step schedulers after optimizer updates; plateau consumes validation loss. Warmup and unfreezing policy are taught in Track B.

## Local experiment budget

Begin at 320-pixel input, batch 2, workers 0; profile before increasing. Diagnostics first: forward/backward, then tiny-set overfit, then short train/val runs. AMP/accumulation come later with numerical checks; do not silently change effective batch between comparisons. No cloud requirement.

No detection results exist yet. Setup passing is not evidence that a detector trains successfully.
