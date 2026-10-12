# Experiment log

Status: no detection experiments executed. Setup checks are recorded separately in SETUP_VALIDATION.md. Never replace missing values with fabricated numbers.

## Required run entry

- ID / date / git SHA and dirty diff / environment / seed(s)
- Question and falsifiable hypothesis / stage / smoke, exploratory or benchmark status
- Baseline and single intervention; all additional differences disclosed
- Dataset/source, class mapping, exact train/val/test manifest hashes, subset membership
- Input geometry, normalization, augmentation, difficult-object policy
- Architecture, initialization, pretraining source and checkpoint hash; head/assignment/decoder
- Loss reductions/weights, optimizer groups, BN parameter and running-stat policy
- Batch and effective batch, AMP, epochs/updates, LR/warmup/scheduler, checkpoint rule
- Validation selection rule and evaluator version: AP integration method, IoU thresholds, NMS and confidence settings
- Measured AP50, AP75, AP@[.50:.05:.95], per-class AP, precision/recall at stated score and IoU thresholds
- Total/trainable parameters, wall-clock training time, images/s, peak allocated/reserved VRAM
- Inference batch, device, resolution, warmup count, timed samples, synchronized p50/p95 latency; state whether preprocessing and NMS included
- Failure examples: missed objects, duplicates, misclassification, inaccurate boxes, small/occluded objects
- Evidence paths / interpretation / confounders / next action / documented takeaway

## Reviewer rules

A tiny-set overfit check precedes expensive runs. Log localization/objectness/classification separately and inspect decoded predictions. Never tune on test. Predeclare checkpoint criterion. Comparable A–D must retain the same head where feasible; different backbone shapes/capacity are confounders. E is an external reference. Screen cheaply, then repeat meaningful comparisons with at least three seeds where affordable and report spread. Good mAP does not waive planned mechanism experiments.
