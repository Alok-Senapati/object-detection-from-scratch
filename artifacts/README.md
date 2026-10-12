# Ignored local run outputs

Each run stores metadata.json, metrics.jsonl and tensorboard/. Real experiments must also store split hashes/manifests, classes, resolved preprocessing, seeds, checkpoint source/hash, timings and the final report. Never commit datasets or checkpoints. Keep metrics unavailable until measured.
