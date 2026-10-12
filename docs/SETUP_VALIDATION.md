# Setup validation — 2026-10-12

## Agent execution environment — executed successfully

- `uv lock --check`: resolved 153 packages; lock and project metadata agree.
- `uv sync --locked --extra cpu`: clean Python 3.12 environment installed, editable package built.
- `uv run --no-sync python scripts/check.py`: Ruff lint passed, 29 Python files formatted, strict Mypy passed for 20 source files, Pytest **9 passed**.
- `uv run --no-sync od-diagnose`: Python 3.12.14, torch 2.10.0+cpu, CUDA unavailable, MPS unavailable, selected CPU.
- `uv run --no-sync od-smoke`: finite tensor loss/backward, JSONL metrics and TensorBoard event creation passed on CPU. Synthetic tensor smoke; no detection metrics.
- Unit checks cover RNG repeatability, config typo rejection, device contract, atomic checkpoint roundtrip including optimizer state, mismatched head rejection, TensorBoard event readback, frozen gradients, optimizer groups/epoch-budget scheduling and weighted statistics/timer.
- Stage 1 annotation figure rendered and visually inspected. It is synthetic educational geometry, not a dataset or prediction artifact.

## User-provided Windows CUDA evidence — 2026-10-12

The user supplied command output from native Windows PowerShell 7.6.6, not WSL2. This evidence was reviewed; it was not independently executed by the agent.

- `uv sync --locked --extra cuda`: succeeded; resolved 153 packages and installed torch 2.10.0+cu128 / torchvision 0.25.0+cu128.
- `od-diagnose`: Python 3.12.13, CUDA runtime 12.8, selected `cuda`, NVIDIA GeForce RTX 3060 Laptop GPU, total memory 6143.5 MiB. Allocated/reserved/peak values were zero at the idle diagnosis, not a measured training memory requirement.
- `od-smoke`: passed on CUDA; local run `artifacts/20261012T175127-0e2ebe33`. The supplied output establishes smoke completion, but no GPU timing or peak-memory values were supplied.
- `python scripts/check.py`: Ruff passed, 30 files already formatted, strict Mypy passed for 20 source files, 9 tests passed in 6.34 seconds. The file count is recorded as reported; no source snapshot/hash was supplied for this local run.
- TensorBoard 2.21.0 served at http://localhost:6006/. Its TensorFlow-absent message does not block this project's PyTorch scalar logging.
- uv fell back from hardlinks to copies and completed installation. Explicit PowerShell-compatible copy mode: `uv sync --locked --extra cuda --link-mode=copy`.

## Remaining checks / limits

- Native Windows CUDA diagnosis/smoke is now evidenced above. WSL2 CUDA and macOS MPS remain untested; the agent environment exposes no GPU.
- Workload peak GPU VRAM, throughput, inference latency and mixed precision remain unmeasured. These need later detector implementation and local profiling.
- Notebook optional environment installation.
- VOC download, annotations/split validation, any detector training/evaluation, and any pretrained weights. These belong to later mentorship stages.
- CI configuration is supplied; remote CI result is reported separately in the PR/status response and is not inferred from local checks.

## Stage 0 execution gate

Setup/check/device/smoke execution requirements are satisfied by the agent CPU checks and user-provided native Windows CUDA evidence. The brief reproducibility quiz remains open. Next task: Stage 1 target representation and diagnostic answers. Commit/checkpoint/split provenance and deterministic settings are required for actual experiments; setup does not implement exact mid-training resume.
