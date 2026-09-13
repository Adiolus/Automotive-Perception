# Automotive Perception Project Status

## Frozen / accepted components

- Object detection: YOLO baseline, visually validated; checkpoint preserved.
- Tracking: `src/tracking/final_tracker_production.py` - Ownership Lock50 Production Tracker. Measured 24.76 FPS end-to-end for its production artifact run; 227 IDs created, 47-frame median duration, 0 duplicate-track pairs at IoU >= 0.5, 41 high-overlap handoffs, 23 candidate genuine switches. Red-shirt complete-exit/re-entry remains unresolved; striped-shirt continuity passed.
- Depth: Depth Anything V2 Small, asynchronous every 5 frames. Raw depth inference about 2.33 FPS, effective update rate about 2.28 Hz; depth worker did not block the main path.
- Fusion: asynchronous tracker-depth interface with timestamp alignment and stale-data handling; segmentation/lane inputs optional.
- Risk: stateful per-track risk layer with EMA depth/closing speed, filtered TTC, persistence/hysteresis and optional lane/semantic interfaces.
- Real-time pipeline: asynchronous visualization/encoding separated from perception. Strict source-rate benchmark reached 30 FPS with 0 dropped perception records and 29.98 FPS recorded visualization in the tested setup.

## Lane status

- Current lane implementation is a lightweight classical baseline using HSV white/yellow masks, Canny/Hough geometry, polynomial curves, provisional BEV and temporal lane matching.
- Full-video baseline: 1,801 frames, 601 detector updates, every 3 frames, raw lane inference 6.09 FPS, effective update rate 10.01 Hz, lane loop 18.24 FPS, 527 persistent lane IDs created.
- Important limitation: visual inspection found substantial false lane hypotheses under traffic, glare and occlusion.
- Next lane step: benchmark a learned detector (CLRerNet first), then Indian-domain adaptation, BEV normalization, temporal lane stitching, topology and ego-lane relation.

## Integration direction

```text
YOLO + Tracker (every frame)
          +
Depth Anything V2 Small (async/cache)
          +
Segmentation (async/cache)
          +
Lane detector + stitching (async/cache)
          -> common timestamped fusion state
          -> TTC + trajectory + lane conflict + semantic context
          -> risk state
```
