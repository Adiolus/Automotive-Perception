# Semantic Segmentation Report

## Current implementation

- Architecture: SegFormer MiT-B0, initialized from `nvidia/mit-b0`.
- Local checkpoint: `models/segmentation/epoch=9-step=34970.ckpt`.
- Taxonomy: 41 IDD class IDs, defined in
  [`class_mapping.py`](../../../src/fusion/class_mapping.py). Semantic IDs are
  distinct from YOLO detection class labels.
- Inference: source video frames are processed asynchronously; the integer
  mask is resized to the source frame dimensions before object association.
- Fusion entry point: [`run_fusion.py`](../../../src/fusion/run_fusion.py)
  with `--semantic-fusion`; object ROI association is implemented in
  [`semantic_object_fusion.py`](../../../src/fusion/semantic_object_fusion.py).
- Output quality: semantic context is advisory. The reported confidence is a
  dominant-pixel fraction adjusted for mask age, not calibrated model
  confidence. No ground-truth pixel annotations for the evaluated road video
  were available, and sampled class disagreements mean semantic labels must
  not replace YOLO labels.

## Current measured run

On the 1,801-frame `indian_road.mp4` staged fusion run, SegFormer completed
360 of 361 submitted requests, with one dropped request and no worker errors.
The raw inference rate was 16.67 FPS and the effective update rate was 6.00 Hz.
Of 19,794 tracker records, 19,689 received an accepted mask context, 94 had no
result, and 11 masks were expired by the configured 10-frame age limit. All
accepted contexts came from an earlier frame; they were age-bounded and
freshness-decayed.

These are asynchronous segmentation-stage figures, not end-to-end perception
FPS. The tracked-object and depth inputs were saved artifacts, and the
segmentation results were not ground-truth validated. Details, commands,
depth/TTC limitations, and lane audit are in
[`SEMANTIC_FUSION_INTEGRATION.md`](./SEMANTIC_FUSION_INTEGRATION.md).
