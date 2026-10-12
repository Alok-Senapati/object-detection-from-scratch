# Reference utility audit

Inspected latest `main` on 2026-10-12: `Alok-Senapati/cifar100-cnn` at `731b56a06e3f2af2dced1364023a2725cf056f7b`. No AGENTS.md found in the reference checkout. New repository was empty (size 0 and contents API reported empty).

| Reference | Decision and target |
|---|---|
| `utils/timer.py` | Adapt context-manager/perf_counter pattern to typed synchronized `infra/profiling.py` |
| `utils/printer.py` | Replace decoration with standard idempotent console logging in `infra/experiment.py` |
| `utils/diagnose.py` | Adapt per-parameter/total gradient norms; remove `.data` access |
| `model/trainer.py::get_optimizer` | Extract factory only; allow discriminative parameter groups |
| `model/trainer.py::get_scheduler` | Extract factory only; replace fixed T_max=100 with epoch budget and reject typo names |
| `model/trainer.py` checkpoint/TensorBoard patterns | Reimplement task-independent state persistence and scalar logging; do not copy loop or evaluation |
| `model/loader.py` | Retain restricted state loading idea; caller supplies model rather than dynamic class import |
| `utils/visualizer.py` | Adapt only generic scalar curve saving; exclude confusion matrices, classification accuracy and digit/CIFAR assumptions |
| `scripts/train_baseline.py` | Inspect seed/device pattern only; expand seeding and isolate device selection |
| Reference tests | New temporary-directory/roundtrip/shape contract tests; no model or dataset tests copied |

New fully implemented utilities: validated TOML config, worker seeding, metadata/JSONL, device/VRAM probe, weighted statistics, parameter counts and smoke CLI. No verbatim utility files copied; all adaptations explicitly listed above. Excluded: all models, classification trainer/evaluator, data loaders/transforms, activations, error analysis, UI, datasets, checkpoints and experiment artifacts.
