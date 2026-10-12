# YOLO fine-tuning — Track B / Stage 26

Added 2026-10-12. All runs are planned. Start only after the scratch detector/evaluator and ImageNet-backbone B/C/D experiments pass their acceptance gates. Use the official architecture and pretrained weights here; this does not replace the learner's scratch implementation.

## Model selection snapshot

Verified official sources on 2026-10-12:

| Family | Status and use in this project | Official source |
|---|---|---|
| Ultralytics YOLO26 | Latest released Ultralytics family; primary candidate YOLO26n, then s if profiling permits. Supports an optional NMS-free inference head; explicitly select/report inference path. | https://docs.ultralytics.com/models/yolo26/ |
| Ultralytics YOLO11 | Mature comparison; YOLO11n first, small optional. | https://docs.ultralytics.com/models/yolo11/ |
| YOLO12 | Attention-centric community research alternative; optional, not required. | https://github.com/sunsmarterjie/yolov12 |
| YOLOv13 | Separate research lineage using hypergraph-enhanced perception; optional architecture study, not an Ultralytics YOLO26 successor. | https://github.com/iMoonLab/yolov13 |
| YOLO27 | Preview only; official docs say weights/configurations/implementation unavailable. Excluded from executable benchmarks until release is verified. | https://docs.ultralytics.com/models/yolo27/ |
| RF-DETR | Current real-time transformer family designed for fine-tuning; optional later architectural comparison. Nano is not a YOLO-sized nano: upstream deployment parameter counts are about 30.5M for RF-DETR-N. Profile independently. | https://github.com/roboflow/rf-detr |
| RT-DETRv4 | Released research reference using vision-foundation-model supervision; optional. | https://github.com/RT-DETRs/RT-DETRv4 |
| DEIMv2 / EdgeCrafter | Current research alternatives for transferred/distilled vision features; separate optional study, no extra mandatory implementation track. | https://github.com/Intellindust-AI-Lab/DEIMv2 and https://intellindust-ai-lab.github.io/projects/EdgeCrafter/ |

This is a practical shortlist, not a universal leaderboard. YOLO names come from multiple teams; version numbers do not define one linear release sequence. Recheck releases/package compatibility at execution time. Store package/version, upstream code SHA, weight URL/hash and dataset provenance. No framework dependency or weight download is added at this planning stage.

## Learning steps and gates

| Step | Acceptance criteria | Brief quiz | Coding exercise / evidence | Takeaway to record |
|---|---|---|---|---|
| 26.1 Data adaptation | VOC-to-YOLO labels invert correctly; class map and split hashes preserved; overlay audit passes | Why is YOLO cxcywh normalization different from VOC XML coordinates? | Implement exporter from the already validated parser; include empty images and explicit difficult/non-target object policies | Pending learner review |
| 26.2 Architecture/checkpoint audit | Explain backbone, neck, head and all prediction tensor shapes; log loaded/reinitialized class tensors | Does loading COCO weights mean the new head starts entirely random? | Inspect checkpoint/class remapping, compare tensor values and shapes before/after adaptation | Pending learner review |
| 26.3 Staged fine-tuning | Verify weights, gradients, trainable counts, BN buffers and optimizer scope for E1/E2/E3 | Does a framework freeze flag also freeze BN running statistics? | Run short staged diagnostics, then declared-budget comparisons | Pending learner review |
| 26.4 Benchmark/deployment | Common evaluator agrees; errors, AP and resources reported; exported predictions numerically checked | Can model-only latency establish end-to-end application FPS? | Benchmark selected model and validate ONNX export on held-out images | Pending learner review |

Teach architecture and loss differences explicitly. Modern YOLO versions need not have a separate objectness output or match the minimal detector's head/loss decomposition. YOLO26's optional one-to-one/NMS-free path is a later comparison against the NMS concepts learned in Track A, not a reason to skip NMS.

## E experiment matrix

| ID | Initialization | Trainable scope | Purpose |
|---|---|---|---|
| E0 | Random initialization of the same selected YOLO architecture | All | Optional short fixed-budget control for the advantage of detection pretraining |
| E1 | COCO-pretrained, adapted class output | Detection head only; freeze backbone AND neck | Feature-reuse diagnostic |
| E2 | Same starting checkpoint and class-head policy | Last backbone stage + neck + head | Partial unfreezing diagnostic |
| E3 | Same starting checkpoint and class-head policy | Full model | Full fine-tuning benchmark |
| E4 | Selected strategy | Selected scope | 10/25/50/100% nested train-image subsets, affordable screening then confirm meaningful findings |
| E5 | Selected best validation configuration | Inference only | Export parity, latency/throughput/VRAM and failure analysis |

E1/E2/E3 are independent scope comparisons from the SAME initial adapted checkpoint. A separate staged warm-start run can follow, but label it as cumulative training and record its total updates; do not conflate it with the independent comparison. Do not train every model family/size/fraction exhaustively. YOLO26n is primary; YOLO11n gets a small matched screen. Scale up only for a falsifiable question and within measured VRAM limits.

Freeze by audited module names/stages, not an assumed hardcoded layer count. Verify intended freezing after trainer setup and epoch train-mode changes; inspect optimizer membership and both parameter/buffer changes. Rebuild/update optimizer when unfreezing. Use a custom trainer only where the official interface cannot express the intended controlled groups/BN policy.

## Comparison integrity

- Retain the existing A-D understandable-head matrix. YOLO E differs in backbone, neck, head, assignment, loss, augmentation recipe and COCO pretraining; it is a system comparison, not a one-variable replacement.
- Export the exact image manifests instead of a built-in dataset YAML that might download/combine extra data. Preserve official test isolation and the shared evaluator's difficult-object handling.
- Audit available source-pretraining/target image overlap. COCO-pretrained source categories may overlap VOC classes; report this and do not describe the experiment as unseen-class generalization. Unknown source overlap must be stated as unknown.
- Class-output adaptation can reuse matching class rows depending on library version/settings. Explicitly record that policy and compare against reinitialized class outputs only as a separately declared intervention.
- Begin with matched input resolution/geometry and disabled recipe differences where practical. Framework-native letterboxing/augmentation/evaluation differences must be documented; optionally run a separate native-recipe benchmark.
- Fix optimizer rather than silently relying on `optimizer=auto`; current docs say auto can choose both optimizer and LR. Record warmup, weight decay, all groups, effective batch/accumulation, EMA and checkpoint-selection behavior.
- Evaluate predictions with the common AP implementation/reference, not only each framework's summary. Match AP integration, IoU thresholds, max detections, ignore flags, score filtering and inference path.
- For YOLO26, log whether NMS or NMS-free prediction path was used for validation/checkpoint choice AND deployment. Report actual training and fused-deployment parameter counts separately when they differ.

## Local budget and reporting

Proposed diagnostic start on RTX 3060 Laptop 6 GB: nano scale, 320 or 416 pixels, batch 2, workers 0, AMP after numerical validation. These are starting hypotheses, not a measured fit guarantee. Profile peak allocated/reserved VRAM before increasing resolution/batch or moving to small. Any OOM automatic retry/batch change must be recorded and effective batch controls restored before comparisons.

Measure AP50/AP75/AP@[.50:.05:.95], per-class AP, declared-threshold precision/recall, total/trainable parameters, training time, images/s, GPU allocator peaks and batch-1 synchronized latency. Export checks distinguish model-only from preprocessing+model+postprocessing latency. No expected mAP or training time is invented.

Official training configuration reference: https://docs.ultralytics.com/modes/train/
