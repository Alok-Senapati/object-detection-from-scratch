# Mentoring taskbook

Teaching order at every stage: concept → why → intuition → mathematics → numerical walkthrough → visualization → guided implementation → experiment → analysis. You implement all project code. The mentor teaches concepts, gives focused examples and hints, and reviews submissions; repository edits require an explicit request. Documentation updates may be performed when requested. PR #3 remains unmerged reference material and is not learner completion evidence. Never mark complete without evidence. No beginner CNN repeat and no unrelated LLM content.

## Current state — 24 November 2026

- **Milestone 1 — Reliable detection data pipeline: complete.** Learner implementation, synthetic integration tests, successful CI and a supplied real six-image training-batch visualization have been reviewed.
- **Next: Milestone 2 — Bounding-box geometry.** Teach corner vs center-size coordinates, work a numerical example, then let the learner implement vectorized conversions before normalization/clipping and IoU.
- Code evidence on main: `2ce3764` (parser/preview), `3e3f041` (dataset/collation), `2975e25` (remove duplicate loading/mapping and expand tests), `624f398` (batch visualization and inspection CLIs).
- [CI for 624f398](https://github.com/Alok-Senapati/object-detection-from-scratch/actions/runs/45457385619) passed project checks, CPU diagnostics and smoke execution. Real-data visualization evidence is separate from synthetic CI.
- Image evidence: learner attachment `visualize_train_batch_1795544817334067600.png` ([saved image](https://chatgpt.com/library?file_id=libfile_872d9d0dd2e08191a09884acfbec131d&account_id=personal)), reviewed in conversation. Six variable-size images show car, horse/person, cyclists and aeroplanes with plausibly aligned boxes, decoded class names and T/D flags. This is an annotation overlay, not detector output; it does not verify every annotation in the dataset.
- Stage 0 execution gates remain passed; reproducibility quiz remains open. Original Stage 1 diagnostic questions not explicitly answered remain open, but do not require restarting the completed data milestone.
- Original Stage 4 is partial: ingestion is complete; split hashes/leakage audit, invertible letterbox and joint transforms remain pending.
- No detector training, trained checkpoint or detection AP results have been recorded.

## Working agreement

Teach one concept at a time: what/why/where, intuition, numerical example and shapes, then implementation contract, learner attempt, review and evidence. Skip beginner Python/CNN repetition. The learner writes all project code; do not implement ahead or create PRs unless explicitly requested. Keep future solutions unrevealed until an attempt or request for help.

## Project milestones

These 13 working milestones group the original detailed stages below; milestone numbers and stage numbers are different. Preserve the detailed stages as a concept checklist.

| Milestone | Scope / original stages | State and acceptance evidence |
|---|---|---|
| 1 | Reliable detection data pipeline — formulation plus ingestion portion of Stages 1 and 4 | Complete: parser, tensor dataset, variable-size collate, tests and real batch visualization reviewed |
| 2 | Bounding-box geometry — Stages 2–3 | Next: conversion roundtrips, normalization/clipping, hand-checked IoU and related overlap functions |
| 3 | Training-ready images and splits — remaining Stage 4 | Planned: reversible letterbox, joint image/box augmentation and split leakage audit |
| 4 | Minimal detector architecture — Stage 5 | Planned: declared backbone/head shapes and validated forward pass |
| 5 | Decode predictions and assign targets — Stages 6–7 | Planned: numerical decoding, assignment and collision/boundary cases |
| 6 | Detection losses and gradient flow — Stage 8 | Planned: separate loss diagnostics, finite gradients and empty-target behavior |
| 7 | Make the detector learn — Stage 9 | Planned: tiny-set overfit with plausible decoded boxes |
| 8 | Evaluate detections correctly — Stages 10–11 | Planned: NMS, one-to-one matching and independently checked AP |
| 9 | Scratch baseline and justified improvement — Stages 12–14 | Planned: reproducible validation results, resource measurements and error analysis |
| 10 | Controlled transfer learning — Stages 15–23 and 25 | Planned: matched random/frozen/partial/full backbone experiments with BN and optimizer audits |
| 11 | Official YOLO fine-tuning — Stage 26 | Planned: dataset export, provenance, fine-tuning and common evaluation; separate external reference |
| 12 | Low-data and domain adaptation — Stages 24 and 27 | Planned: nested subsets and domain-shift experiment; suitable target data required |
| 13 | Inference application and final benchmark — Stages 28–30 plus delivery | Planned: applicable efficient/adaptor tuning, inference CLI, lightweight demo/API, supported export, latency and final report |

Stages 28–29 may be brought forward when experiments need them. Adapter methods require architectural justification; no unsupported technique is mandatory merely to fill the checklist.

## Milestone 1 review record

- Implemented and reviewed: validated VOC parsing; RGB float32 CHW images in [0,1]; fixed zero-based VOC class IDs; float32 boxes [N,4]; int64 labels [N]; boolean difficult/truncated flags; empty boxes [0,4]; image-size checks; lazy loading; variable-size batching; annotated batch visualization.
- Refactor outcome: duplicate standalone loading and class definitions removed; VOCDataset is the single loading implementation. This satisfies the single-source requirement without requiring a separate load_sample function.
- Learner explanation: recognized different object counts and the need to preserve image/target association. Mentor clarified that unequal shapes prevent stacking; equal-sized targets could stack while retaining image identity via the batch axis.
- Collate concept initially needed explanation; implementation was reviewed after guidance. Do not label all understanding as independently demonstrated.
- Visual review: sampled boxes and labels align with visible objects; overlapping instances and T/D flags are retained. Small-label crowding in the cyclist panel is cosmetic.
- Limits: six-image visual inspection is not a dataset-wide audit; empty samples are evidenced by synthetic tests. Split overlap/content hashes and training/evaluation difficult-object policies are still pending.
- Next checkpoint: teach xyxy vs cxcywh with a worked example, then review learner-written vectorized conversion functions with roundtrip, empty and degenerate cases.

## Original stage record

| Stage | State | Evidence | Takeaway |
|---|---|---|---|
| 0 | CPU/native Windows CUDA execution passed; quiz pending | SETUP_VALIDATION.md and successful CI | Setup execution does not establish detector performance |
| 1 | Practical target contract evidenced; remaining diagnostic questions open | Dataset/collate code, tests and conversation | Targets are variable-size instance sets; default stacking cannot handle unequal shapes |
| 4 | Ingestion complete; transforms and split audit pending | Commits through 624f398 and supplied six-image overlay | Image/target alignment checked on a real sample batch |

## Stage 0 — Environment, repository setup and diagnostic

- Status: execution gates complete; reproducibility quiz pending
- Acceptance: Clean sync; tests/lint/types pass; local device smoke evidenced.
- Brief quiz: Why can seeding alone fail to reproduce CUDA runs?
- Coding exercise: completed from supplied Windows diagnosis, CUDA smoke and all 9 tests; versions and idle VRAM recorded in SETUP_VALIDATION.md.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: native Windows CUDA execution is verified by supplied logs; idle allocator statistics are not training VRAM. Reproducibility quiz answer remains pending.

## Stage 1 — Detection formulation

- Status: practical target contract evidenced through Milestone 1; unanswered diagnostic questions remain open
- Acceptance: Represent variable-size object sets, empty images and explicit shapes.
- Brief quiz: Why does image classification lose instance count?
- Coding evidence: learner implemented real VOC targets, empty-target handling and collate; reviewed in commits through 624f398. The original synthetic exercise was not separately submitted.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: variable object counts require an explicit batching policy; preserve image/target ordering. Remaining original diagnostic answers are not inferred from code.

## Stage 2 — Bounding-box representations

- Status: next lesson (Milestone 2); implementation not yet reviewed
- Acceptance: Roundtrip xyxy/cxcywh, clip/normalize with edge-case tests.
- Brief quiz: Why must x coordinates be normalized by width?
- Coding exercise: Implement conversions and draw boxes; cover empty/degenerate boxes.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 3 — IoU, GIoU, DIoU, CIoU

- Status: planned
- Acceptance: Derive each term and validate numeric/disjoint/degenerate cases.
- Brief quiz: Why can ordinary IoU give no useful signal for separated boxes?
- Coding exercise: Implement overlap functions and numerical examples yourself.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 4 — VOC ingestion and transforms

- Status: partial — parsing, dataset, batching and visualization complete; transforms and split audit pending
- Acceptance: Inspect annotations, hashed leak-free splits and invertible letterbox.
- Brief quiz: What changes in the boxes when padding is added?
- Coding exercise: Parse XML, preserve difficult flags and transform image and boxes together.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 5 — Minimal one-stage detector design

- Status: planned
- Acceptance: Specify backbone/head interface and prediction shape before coding.
- Brief quiz: Why should a head retain spatial information?
- Coding exercise: Draw your proposed architecture and estimate output/parameter sizes.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 6 — Grid prediction and decoding

- Status: planned
- Acceptance: Explain raw logits vs decoded pixel boxes and probabilities.
- Brief quiz: Is objectness the same quantity as class probability?
- Coding exercise: Implement decoder and numerical grid examples.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 7 — Assignment and target encoding

- Status: planned
- Acceptance: Explicit positive/negative/ignored policy and collision tests.
- Brief quiz: What if two object centers occupy one cell?
- Coding exercise: Implement target assignment with collision and boundary tests.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 8 — Detection losses and imbalance

- Status: planned
- Acceptance: Define reductions, masks, weights and separate loss diagnostics.
- Brief quiz: Why can easy background dominate the loss?
- Coding exercise: Implement loss terms; verify gradients and empty-target behavior.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 9 — Train and debug minimal detector

- Status: planned
- Acceptance: Tiny-set overfit, finite gradients, decoded visual checks.
- Brief quiz: How would you distinguish assignment failure from learning-rate failure?
- Coding exercise: Write detection training orchestration and evidence-backed debug report.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 10 — Thresholding and NMS

- Status: planned
- Acceptance: Hand-calculated class-aware NMS cases match implementation.
- Brief quiz: Why can class-agnostic NMS suppress valid overlapping objects?
- Coding exercise: Implement greedy NMS; compare output at multiple score thresholds.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 11 — Precision, recall, AP and mAP

- Status: planned
- Acceptance: One-to-one matching and independent evaluator agreement.
- Brief quiz: Why is duplicate detection a false positive?
- Coding exercise: Implement PR/AP and hand-check a ranked list.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 12 — Small YOLO-style extension

- Status: planned
- Acceptance: Justify grid capacity/multi-scale changes from baseline errors.
- Brief quiz: Which failure evidence supports adding a finer feature map?
- Coding exercise: Design and implement a small extension with explicit tensor shapes.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 13 — Controlled scratch experiments

- Status: planned
- Acceptance: Predeclared interventions, fixed evaluation and confounder log.
- Brief quiz: Why is changing loss and augmentation together hard to interpret?
- Coding exercise: Run small controlled architecture/loss/assignment/augmentation ablations.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 14 — Scratch baseline report

- Status: planned
- Acceptance: Measured AP, errors/resources and complete provenance.
- Brief quiz: What prevents a fair baseline comparison?
- Coding exercise: Write baseline report and lock transfer comparison protocol.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 15 — Transfer fundamentals

- Status: planned
- Acceptance: Explain learned features, source/target tasks and domain shift.
- Brief quiz: Why might an edge filter transfer better than a class template?
- Coding exercise: Inspect early/late representations and explain reuse mathematically.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 16 — Feature extraction vs fine-tuning

- Status: planned
- Acceptance: Predict gradient/update/buffer behavior for each mode.
- Brief quiz: Can a frozen backbone still propagate an input gradient?
- Coding exercise: Run a small gradient-flow experiment and distinguish transfer strategies.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 17 — Official backbone anatomy

- Status: planned
- Acceptance: Name feature stages, strides/channels and replaced classifier.
- Brief quiz: Why is the classification vector often unsuitable for spatial detection?
- Coding exercise: Extract stage feature maps without a ready-made detector.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 18 — Pretrained backbone integration

- Status: planned
- Acceptance: Same head contract, checkpoint provenance and shape verification.
- Brief quiz: What makes a changed backbone a structural confounder?
- Coding exercise: Add an interface adapter and a matched random-init backbone control.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 19 — Frozen-backbone experiment B

- Status: planned
- Acceptance: Head learns; frozen weights and intended BN buffers verified.
- Brief quiz: Does requires_grad=False stop BN statistics updates?
- Coding exercise: Run frozen head training with explicit BN policy.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 20 — Progressive unfreezing C

- Status: planned
- Acceptance: Layer scope/optimizer groups and gradients audited.
- Brief quiz: What must change in the optimizer when a stage is unfrozen?
- Coding exercise: Unfreeze last stage(s), recording trainable counts and learning rates.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 21 — Full fine-tuning D

- Status: planned
- Acceptance: All intended layers update; stable validation learning curve.
- Brief quiz: Why can pretrained parameters need smaller updates?
- Coding exercise: Run full fine-tuning under the declared budget.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 22 — Fine-tuning optimization

- Status: planned
- Acceptance: Discriminative groups/warmup/decay/BN policies justified.
- Brief quiz: Why should warmup duration be specified in updates?
- Coding exercise: Compare a focused optimization intervention and log per-group LR.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 23 — Forgetting, negative transfer and drift

- Status: planned
- Acceptance: Distinguish mechanisms using controlled validation evidence.
- Brief quiz: Does worse target performance alone prove catastrophic forgetting?
- Coding exercise: Measure representation drift and report limits of forgetting evidence.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 24 — Low-data experiments

- Status: planned
- Acceptance: Nested subsets, fixed val/test, actual class/box counts.
- Brief quiz: Why must validation remain fixed across label fractions?
- Coding exercise: Compare 10/25/50/100% with fixed membership and uncertainty.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 25 — Matched scratch vs pretrained

- Status: planned
- Acceptance: Same architecture control and all pretraining advantages disclosed.
- Brief quiz: Does same head imply identical detector capacity?
- Coding exercise: Build the matched architecture/init comparison report.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 26 — Detection checkpoint transfer E and YOLO fine-tuning

- Status: planned
- Acceptance: New class map, replacement head and loaded/skipped key audit.
- Brief quiz: Why does strict=False not solve wrong classifier tensor size?
- Coding exercise: Adapt an official detector as a separate reference experiment.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

For Stage 26 substeps, quizzes and acceptance gates, see [YOLO_FINE_TUNING.md](YOLO_FINE_TUNING.md). Mandatory primary reference: YOLO26n E1/E2/E3; YOLO11n is a small comparison screen. Optional alternatives do not expand the mandatory track. All YOLO runs remain planned.

## Stage 27 — Domain adaptation

- Status: planned
- Acceptance: Input-channel and normalization policies tied to target semantics.
- Brief quiz: Why might copying RGB kernels fail for probe raster channels?
- Coding exercise: Design a small target-domain adaptation experiment with train-only statistics.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 28 — Lightweight vision fine-tuning

- Status: planned
- Acceptance: Architecture-fit justification and measured comparison or reasoned deferral.
- Brief quiz: Where could LoRA/adapters fit this specific architecture?
- Coding exercise: Test an appropriate adapter method or document why it is deferred.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 29 — Efficient fine-tuning

- Status: planned
- Acceptance: Numerics, effective batch and measured VRAM/speed reported.
- Brief quiz: Is accumulation identical to a larger BN batch?
- Coding exercise: Profile AMP/accumulation/freezing under a memory budget.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 30 — Final benchmark and selection

- Status: planned
- Acceptance: A–E evidence matrix, sealed-test report and deployment tradeoff.
- Brief quiz: Which objective could justify a lower-mAP model?
- Coding exercise: Publish final reproducible report and select for a stated use case.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

