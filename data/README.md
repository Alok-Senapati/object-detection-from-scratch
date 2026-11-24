# Local data only

The loader supports VOC2007 official `train`, `val`, `trainval`, and `test` image manifests. It never downloads data. Supply `JPEGImages/`, `Annotations/`, and `ImageSets/Main/<split>.txt` under the chosen root, with one image ID per line. Blank lines are ignored; duplicate IDs and missing files are rejected. Image-level isolation is required; never split boxes from the same image across splits. All training augmentations stay within train. Hash manifests and check IDs and duplicate images before training. Keep official test sealed until benchmark decisions are frozen.

A manageable diagnostic subset may use a fixed train-only manifest; disclose excluded classes/images and treat its metrics as subset metrics. Begin full experiments on VOC2007 alone; adding VOC2012 is a separate data intervention.

VOC XML uses 1-based inclusive coordinates. The parser converts to zero-based continuous pixel edges, `xyxy`, area `(x2-x1)*(y2-y1)` without `+1`. Explicit conversion from VOC: `(xmin-1, ymin-1, xmax, ymax)`. The loader preserves `difficult` and `truncated` flags without filtering; training and evaluation policies must be explicit and reference-checked. Class IDs follow `VOC_CLASSES` in `od_scratch.data.dataset`, from aeroplane (0) to tvmonitor (19), without a background class.

Official devkit: https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2007/htmldoc/
