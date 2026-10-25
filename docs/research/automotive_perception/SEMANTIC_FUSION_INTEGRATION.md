# Semantic Fusion Integration and Validation

## Scope

This change connects SegFormer semantic context to the existing file-driven
tracker/depth fusion stage. It does not create a continuous live-camera
YOLO-to-risk pipeline: detections and tracks are still read from the saved
tracker CSV, while video frames are read to obtain frame-aligned segmentation
masks. The final visualization also replays frozen risk records.

No model weights were changed or trained, and risk-score thresholds were not
changed. TTC eligibility was tightened because the available depth values are
relative estimates rather than calibrated metric distances.

## Data flow and safeguards

```text
saved tracker CSV + saved depth CSV + source video
             -> bounded asynchronous SegFormer worker
             -> frame-age-aware object/semantic fusion
             -> fused CSV
             -> stateful risk evaluation
             -> bounded visualization replay
```

- SegFormer runs asynchronously with a bounded latest-frame-wins request
  queue. Frame identity and timestamp travel with each result.
- A mask is used only when its frame is not in the future, has an integer
  class-map shape matching the source frame, and is no more than the configured
  10 frames old. Confidence is reduced for older accepted masks; missing,
  invalid, and expired results remain explicit.
- The IDD semantic IDs come from
  [`class_mapping.py`](../../../src/fusion/class_mapping.py). They are kept
  separate from YOLO labels: semantic classes never overwrite tracker classes
  or risk scores. Road IDs are 0 and 2; sidewalk is 3.
- The semantic confidence field is a dominant-pixel fraction adjusted for
  freshness, not a calibrated probability.
- Runtime validation checks that the local SegFormer checkpoint loads
  completely into the installed model. The local checkpoint was verified to
  supply 208 matching tensors, including all 16 decoder-head tensors.
- Depth Anything V2 output and the saved depth CSV are relative estimates.
  Their change is not calibrated in meters per second. Fusion and risk
  therefore reject TTC unless depth is explicitly metric and its derivative
  is explicitly per-second. The existing relative trend-based approach
  heuristic remains available, but it is not a metric TTC. The current saved
  data produces no valid TTC.
- Semantic context is passed through risk output for inspection; it does not
  change risk classes, thresholds, confidence, or scores.

## Measurements

The full staged run used `videos/indian_road.mp4` (1,801 frames), the saved
production tracker CSV, and the saved schedule-5 depth CSV. Values below are
from the final full run unless marked as a short sample.

| Stage / measure | Result | Interpretation |
|---|---:|---|
| Tracker records fused | 19,794 | Every saved tracker record was processed |
| Depth joins | 19,609 / 19,794 (99.07%) | 185 missing; 4,159 joined observations were stale |
| Fusion-stage throughput | 329.36 records/s | Offline record processing, not detector throughput |
| Video frame handling | 29.97 frames/s | Source-rate-paced fusion loop, not live perception FPS |
| Segmentation submitted / completed | 361 / 360 | One request was dropped; zero worker errors |
| Raw SegFormer inference rate | 16.67 inferences/s | Full run on the available local device |
| Effective segmentation update rate | 6.00 Hz | One completed result per roughly five 30-FPS source frames |
| Accepted semantic contexts | 19,689 / 19,794 | 94 records had no result; 11 results were expired |
| Accepted contexts from an earlier frame | 19,689 | Every accepted context was aged; all remained within the configured limit |
| Object-context fusion latency | p50 0.066 ms; p95 0.584 ms | Per tracker record; excludes SegFormer inference |
| Risk records evaluated | 19,794 | 19,689 semantic contexts passed through; valid TTC: 0 |
| Risk distribution | SAFE 13,235; LOW 4,471; MEDIUM 2,088; HIGH 0; CRITICAL 0 | Existing risk scoring; semantic fields do not alter it |
| Visualization replay | 1,801 frames and 19,794 records processed; 0 perception records dropped | Unpaced bounded encoder wrote 234 frames and dropped 1,567 visualization frames |

The accepted semantic masks were not from the exact tracker frame in this run;
they were older masks that passed the age limit. This is distinct from the 11
expired masks, which were rejected. Semantic categories have not been
validated against ground truth for this video. A sampled output showed
YOLO/SegFormer disagreement, so semantic categories must remain advisory.

### Depth-only short sample

The bounded depth sample used 60 source frames at 30 FPS on MPS. It submitted
13 requests, dropped 7, completed 6 inferences, and observed 4 results in the
frame loop. Raw inference was 2.47 FPS, the effective update rate was 3.00 Hz,
and the sample loop processed 24.02 source frames/s. Inference latency was
p50 345 ms and p95 621 ms. It produced 351 track rows and no worker errors.
The source was intentionally incomplete (`--max-frames 60`), so these figures
are not a full-video benchmark. An earlier 60-frame attempt completed only two
inferences; these short asynchronous samples varied and should not be
extrapolated as stable throughput.

### Synthetic microbenchmark

The separate 300-frame fusion benchmark uses mocked inputs. It measured p50
0.293 ms and p95 0.319 ms; these synthetic timings are not model-backed and
are not comparable to end-to-end perception.

## Performance conclusion

The requested end-to-end perception target cannot be claimed from these
measurements. Detection and tracking were replayed from a saved CSV, while
depth and segmentation ran asynchronously at much lower update rates. The
approximately 30-FPS fusion replay, 24-FPS short depth sample, risk throughput,
and visualization encoder rate each measure different stages; none is a
measurement of a live YOLO-to-risk camera pipeline. Memory usage was not
measured.

## Lane-path audit

The current [`run_lane.py`](../../../src/lane/run_lane.py) path uses the
classical color/gradient/Hough detector and `LaneStitcher`; it was not modified
or presented as part of this semantic-fusion benchmark. Its ego-lane relation
remains heuristic. The repository's UFLD-v2 CULane evaluation is a separate
learned-model path, not the detector used by `run_lane.py`; its existing
category F1 values are:

| CULane category | F1 |
|---|---:|
| Normal | 0.920 |
| Crowd | 0.738 |
| Hlight | 0.629 |
| Shadow | 0.726 |
| Noline | 0.479 |
| Arrow | 0.876 |
| Curve | 0.689 |
| Cross | 0.000 |
| Night | 0.697 |

These are pre-existing evaluation results, not rerun for this change. The
cross category has no ground-truth positives (`TP=0`, `FN=0`), 1,996 false
positives, and evaluator recall `-1`; this is a no-positive-ground-truth case,
not a negative recall. Before relying on lane/object association, investigate
those false positives and add conservative intersection/no-lane handling.

## Validation and reproduction

Focused tests: `18 passed`. The changed Python modules passed `py_compile`,
workspace diagnostics reported no errors in the changed files, and
`git diff --check` passed.

Full semantic-fusion command:

```bash
venv/bin/python src/fusion/run_fusion.py \
  --tracking-csv runs/tracking_eval/FINAL_TRACKER_PRODUCTION.csv \
  --depth-csv runs/depth_eval/schedule_5/per_track_depth.csv \
  --semantic-fusion \
  --video videos/indian_road.mp4 \
  --segmentation-checkpoint models/segmentation/epoch=9-step=34970.ckpt \
  --segmentation-interval 5 \
  --max-segmentation-age-frames 10 \
  --realtime \
  --output-csv runs/fusion_eval/semantic_fused_tracks.csv \
  --summary-json runs/fusion_eval/semantic_fusion_summary.json
```

Depth-only bounded sample:

```bash
venv/bin/python src/depth/run_depth.py --schedule 5 --realtime --max-frames 60
```

Downstream evaluation can use the fused CSV with
[`run_risk.py`](../../../src/risk/run_risk.py), then replay the resulting risk
CSV with [`realtime_pipeline.py`](../../../src/pipeline/realtime_pipeline.py).
