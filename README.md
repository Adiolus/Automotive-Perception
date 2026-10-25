# Automotive Perception & ADAS

Real-time camera-based automotive perception system using computer vision and AI.

## Objective

Develop a vision-based system capable of:

- Object detection and classification
- Object tracking
- Moving/stationary classification
- Lane detection and tracking
- Road boundary detection
- Distance estimation
- Motion estimation
- Vision-based speed estimation

## Team

- Aditya
- Balaji

## Current Implementation Status

This project is an experimental automotive-perception pipeline for road-scene
video. Components are under active development; reported throughput and
component-level results should not be interpreted as validated safety performance.

### Pipeline components

- **Object detection:** Custom YOLO-based detection for road users.
- **Multi-object tracking:** Persistent track IDs with duplicate suppression.
  Identity continuity after complete scene exit and re-entry remains an open issue.
- **Depth estimation:** Asynchronous monocular depth estimation. The output is
  relative depth, not calibrated metric distance.
- **Semantic segmentation:** Asynchronous SegFormer integration for adding
  semantic context to tracked objects. Semantic context does not replace the
  detector's class labels.
- **Fusion:** Associates asynchronous perception outputs using frame and
  timestamp information, subject to configured freshness limits.
- **Risk estimation:** Experimental heuristic risk states. Metric time-to-collision
  must not be inferred from relative depth alone.
- **Lane perception:** Classical computer-vision lane detection, geometry,
  stitching, and tracking are available as a baseline. Lane drift, false
  hypotheses, and lane identity stability still require validation.

### Validation and limitations

- Focused semantic-fusion tests are available under `tests/`.
- Existing evaluation artifacts are stored under `runs/` when generated locally.
- Replay throughput is not equivalent to full live end-to-end throughput.
- Detection counts, risk-state counts, and track statistics are not accuracy
  metrics without suitable ground truth.
- CARLA-based controlled validation is planned but is not yet part of the
  verified pipeline.

### Reproducibility

Use the project's configured Python environment and the relevant scripts under
`src/` to reproduce individual component runs. Check each script's `--help`
output for its current arguments. Dataset files, model weights, generated videos,
and virtual environments should remain outside Git unless explicitly required.
