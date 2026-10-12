# Mentoring taskbook

Teaching order at every stage: concept → why → intuition → mathematics → numerical walkthrough → visualization → guided implementation → experiment → analysis. You implement detection algorithms; mentor implements generic infrastructure. Never mark complete without evidence. No beginner CNN repeat and no unrelated LLM content.

## Current state

- Stage 0 infrastructure: implemented; validation evidence in SETUP_VALIDATION.md. Your Windows/WSL GPU smoke remains pending.
- Stage 1: introduced; your diagnostic and coding submission pending.
- Stages 2–30: planned.
- Next concrete task: run local setup, then submit Stage 1 target shapes and diagnostic answers. Stage 1 completion does not require Stage 2 conversion algorithms.

## Milestone record

| Stage | State | Evidence | Takeaway |
|---|---|---|---|
| 0 | Infrastructure verified in execution environment; laptop pending | SETUP_VALIDATION.md | CPU checks do not establish laptop GPU readiness |
| 1 | Awaiting learner submission | STAGE_01.md | Pending your explanation |

## Stage 0 — Environment, repository setup and diagnostic

- Status: partially complete
- Acceptance: Clean sync; tests/lint/types pass; local device smoke evidenced.
- Brief quiz: Why can seeding alone fail to reproduce CUDA runs?
- Coding exercise: Run diagnose and smoke on your laptop; report versions/device/VRAM.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 1 — Detection formulation

- Status: in progress
- Acceptance: Represent variable-size object sets, empty images and explicit shapes.
- Brief quiz: Why does image classification lose instance count?
- Coding exercise: Build the three-object and empty targets from STAGE_01.md.
- Experiment/analysis: submit a small numerical or execution check appropriate to this stage, inspect its evidence, and explain one failure case before longer runs.
- Documented takeaway: pending learner submission and review; record what changed your understanding, evidence and next action here.

## Stage 2 — Bounding-box representations

- Status: planned
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

- Status: planned
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

