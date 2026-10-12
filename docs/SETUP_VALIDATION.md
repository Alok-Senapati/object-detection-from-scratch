# Setup validation — 2026-10-12

## Executed successfully

- `uv lock --check`: resolved 153 packages; lock and project metadata agree.
- `uv sync --locked --extra cpu`: clean Python 3.12 environment installed, editable package built.
- `uv run --no-sync python scripts/check.py`: Ruff lint passed, 29 Python files formatted, strict Mypy passed for 20 source files, Pytest **9 passed**.
- `uv run --no-sync od-diagnose`: Python 3.12.14, torch 2.10.0+cpu, CUDA unavailable, MPS unavailable, selected CPU.
- `uv run --no-sync od-smoke`: finite tensor loss/backward, JSONL metrics and TensorBoard event creation passed on CPU. Synthetic tensor smoke; no detection metrics.
- Unit checks cover RNG repeatability, config typo rejection, device contract, atomic checkpoint roundtrip including optimizer state, mismatched head rejection, TensorBoard event readback, frozen gradients, optimizer groups/epoch-budget scheduling and weighted statistics/timer.
- Stage 1 annotation figure rendered and visually inspected. It is synthetic educational geometry, not a dataset or prediction artifact.

## Not executed / limits

- Windows/WSL2 CUDA and macOS MPS execution; no GPU exposed in this execution environment. CUDA wheel sources resolve in the lock, but CUDA installation/execution was not tested.
- Laptop GPU VRAM, speed, inference latency and mixed precision. These require the user's local hardware and later detector implementation.
- Notebook optional environment installation.
- VOC download, annotations/split validation, any detector training/evaluation, and any pretrained weights. These belong to later mentorship stages.
- CI configuration is supplied; remote CI result is reported separately in the PR/status response and is not inferred from local checks.

## Local Stage 0 completion

Use README's CUDA sync command. Run `od-diagnose` and `od-smoke` with `--no-sync`; submit the diagnosis plus smoke result. A CPU fallback on an intended CUDA laptop must be investigated before longer experiments. Commit/checkpoint/split provenance and deterministic settings are required for actual experiments; setup does not implement exact mid-training resume.
