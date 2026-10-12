# Transfer-learning experiment protocol

All experiments are PLANNED, not completed. Track B opens after the from-scratch detector and evaluator pass Track A acceptance gates.

| ID | Initialization | Update scope | Comparison limits |
|---|---|---|---|
| A | Random backbone and head | All | Mechanism baseline; architecture disclosed |
| B | ImageNet backbone, random same head | Head only | Backbone buffers need an explicit BN policy |
| C | Same ImageNet checkpoint and head init | Head + last stage, then selected stages | Unfreezing schedule changes optimization |
| D | Same ImageNet checkpoint and head init | All | Discriminative LRs and BN policy recorded |
| E | COCO-pretrained YOLO26n; YOLO11n comparison screen | Adapted class output; head-only, partial and full fine-tuning | Separate architecture/neck/head/assignment/loss and detection-pretraining advantages |

Use one official backbone architecture for a random-init A transfer control and B/C/D, with compatible feature adapters and identical head. Preserve the original minimal-detector A as a separate baseline if its architecture differs. The comparison matrix must name both A variants rather than conceal structural differences.

## Controls

Fix image split manifests, class set, resolution/letterbox, evaluator, augmentation distribution and selection criterion. Distinguish fixed-budget comparisons (same update budget) from convergence-oriented comparisons (separately declared budgets). Fix initial head and subset seeds where possible. Record total/trainable counts; frozen models legitimately have different throughput/memory. No claim of one-variable difference if architecture, normalization, augmentation, pretraining or optimization also changes.

For 10%, 25%, 50%, 100%, use **nested train-image subsets** from a fixed seeded ordering; include every annotation for a selected image. Report actual class/box counts and missing-class coverage. Validation/test never shrink. Repeat subset selection with additional seeds for final low-data claims if affordable. Do not remove images merely because they include non-target objects without defining their treatment.

## Gradient and checkpoint topics to demonstrate

`h=f_theta(x)`, prediction `g_phi(h)`: frozen theta receives no parameter gradient/update; trainable phi does. A frozen layer still permits gradient with respect to its input when earlier layers require gradients. `no_grad()` around the backbone is valid only when upstream gradients are unnecessary. `requires_grad=False` does not change train/eval mode or BN running mean/variance. Optimizer groups must exclude frozen parameters and be rebuilt/updated on unfreezing.

Demonstrate early transferable features vs task-specific features, source/target domain shift, feature extraction vs partial/full tuning, catastrophic forgetting, negative transfer and representation drift with equations, shapes, visual and numerical examples at the corresponding stages. Use discriminative groups (illustrative: backbone 1e-4, head 1e-3), then justify from validation evidence rather than assume those rates are optimal.

Inspect checkpoint keys and shapes before loading. `strict=False` still rejects shape mismatch: explicitly filter/replace mismatched head tensors and log every loaded/skipped key. Preserve class mapping. Input-channel adaptation may use a learned input projection or documented kernel adaptation; neither guarantees probe/raster transfer. Fit target normalization on train only. RGB appearance is not equivalent to probe semantics.

Adapters/LoRA are an optional measured intervention only if appropriate to the selected vision architecture. AMP, accumulation, staged freezing and VRAM profiling must preserve documented effective batch; AMP numerical behavior is tested before benchmark use.

## Metrics contract

Report VOC2007 11-point AP50 under its own name. Separately report all-point interpolated AP50/AP75 and mean AP over .50:.05:.95 with precise integration/ignore conventions; do not call this full COCO evaluation unless COCO area/max-detection conventions are implemented. Cross-check your AP with an independent trusted evaluator before accepting results. Preserve difficult annotations for reference ignore handling. Precision/recall require explicit score and IoU thresholds and one-to-one matching.

Training times and throughput use declared windows; latency excludes initialization and reports warmup, batch, synchronization and included preprocessing/postprocessing. Unmeasured metrics remain N/A with reason.

## Final report gate

Complete A–E matrix, per-class AP, PR curves, low-data curves, localization/duplicate/occlusion analysis, resource measurements, provenance and confounders. Select a model for a declared deployment objective, not mAP alone. Test is run after experiment choices are locked.

## YOLO reference extension

Stage 26 now explicitly includes official YOLO fine-tuning with class-map/head audits, E1 head-only, E2 partial and E3 full training, followed by low-data and deployment checks. See [YOLO_FINE_TUNING.md](YOLO_FINE_TUNING.md) for substeps, controls, model shortlist and resource budget. No YOLO training has run.
