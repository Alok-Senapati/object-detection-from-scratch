# Local data only

Stage 4 will use VOC2007 official `train`, `val`, `test` image manifests. No downloads yet. Image-level isolation; never split boxes from the same image across splits. All training augmentations stay within train. Hash manifests and check IDs and duplicate images before training. Keep official test sealed until benchmark decisions are frozen.

A manageable diagnostic subset may use a fixed train-only manifest; disclose excluded classes/images and treat its metrics as subset metrics. Begin full experiments on VOC2007 alone; adding VOC2012 is a separate data intervention.

VOC XML uses 1-based inclusive coordinates. Our future geometry convention is zero-based continuous pixel edges, `xyxy`, area `(x2-x1)*(y2-y1)` without `+1`. Explicit conversion from VOC: `(xmin-1, ymin-1, xmax, ymax)`. Keep `difficult` flags; training and evaluation policies must be explicit and reference-checked.

Official devkit: https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/htmldoc/
