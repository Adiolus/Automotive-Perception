"""Timestamp-align tracker/depth records and optionally run asynchronous IDD fusion."""

import argparse
import csv
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fusion import FusionEngine, parse_config, parse_depth, parse_tracking
from semantic_fusion_interface import SceneFusionInterface, SegmentationWorker


DEFAULT_TRACKING = ROOT / "runs/tracking_eval/FINAL_TRACKER_PRODUCTION.csv"
DEFAULT_DEPTH = ROOT / "runs/depth_eval/schedule_5/per_track_depth.csv"
DEFAULT_VIDEO = ROOT / "videos/indian_road.mp4"
DEFAULT_SEGMENTATION_CHECKPOINT = ROOT / "models/segmentation/epoch=9-step=34970.ckpt"
DEFAULT_CONFIG = HERE / "fusion_config.yaml"
DEFAULT_OUTPUT = ROOT / "runs/fusion_eval/fused_tracks.csv"
OUTPUT_FIELDS = [
    "frame", "timestamp", "track_id", "class", "bbox", "velocity", "trajectory",
    "depth", "depth_type", "depth_quality", "depth_change", "depth_change_per_second",
    "depth_history", "depth_trend", "closing_speed", "closing_speed_type",
    "ttc", "ttc_valid", "ttc_reason",
    "lane_relation", "lane_center_offset", "lane_crossing", "lane_conflict",
    "lane_confidence", "semantic_context", "semantic_class_id", "semantic_distribution",
    "semantic_confidence", "road_contact", "sidewalk_contact", "segmentation_age_frames",
    "confidence", "risk_inputs", "data_age", "staleness",
]


def _json(value):
    return json.dumps(value, separators=(",", ":"))


def _write_record(writer, record):
    row = dict(record)
    row["class"] = row.pop("class_name")
    for key in (
        "bbox", "velocity", "trajectory", "depth_history", "semantic_distribution",
        "risk_inputs", "data_age", "staleness",
    ):
        row[key] = _json(row[key])
    writer.writerow(row)


def _next_row(rows):
    try:
        return next(rows)
    except StopIteration:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-csv", type=Path, default=DEFAULT_TRACKING)
    parser.add_argument("--depth-csv", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--semantic-fusion", action="store_true")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--segmentation-checkpoint", type=Path, default=DEFAULT_SEGMENTATION_CHECKPOINT)
    parser.add_argument("--segmentation-device")
    parser.add_argument("--segmentation-interval", type=int, default=5)
    parser.add_argument("--max-segmentation-age-frames", type=int, default=5)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()

    if args.max_records is not None and args.max_records < 1:
        parser.error("--max-records must be positive")
    if args.segmentation_interval < 1:
        parser.error("--segmentation-interval must be positive")
    if args.semantic_fusion and not args.video.is_file():
        raise FileNotFoundError(f"video not found: {args.video}")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    engine = FusionEngine(parse_config(args.config))
    scene_fusion = (
        SceneFusionInterface(args.max_segmentation_age_frames)
        if args.semantic_fusion else None
    )
    capture = None
    video_fps = None
    video_frame = 0
    current_frame = None
    if args.semantic_fusion:
        capture = cv2.VideoCapture(str(args.video))
        if not capture.isOpened():
            raise RuntimeError(f"could not open {args.video}")
        video_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        segmentation_worker = (
            SegmentationWorker(args.segmentation_checkpoint, device=args.segmentation_device)
            if args.semantic_fusion else None
        )
    except Exception:
        if capture is not None:
            capture.release()
        raise

    started = time.perf_counter()
    realtime_started = started
    depth_rows = 0
    records = 0
    last_tracking_timestamp = -math.inf
    last_depth_timestamp = -math.inf
    next_depth = None
    depth_handle = None
    latencies_ms = deque(maxlen=4096)
    source_complete = True
    semantic_context_records = 0
    semantic_stale_records = 0

    try:
        with args.tracking_csv.open(newline="") as tracking_handle, args.output_csv.open(
            "w", newline=""
        ) as output_handle:
            tracking_rows = csv.DictReader(tracking_handle)
            writer = csv.DictWriter(output_handle, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()

            if args.depth_csv.is_file():
                depth_handle = args.depth_csv.open(newline="")
                depth_rows_iter = iter(csv.DictReader(depth_handle))
                next_depth = _next_row(depth_rows_iter)
            else:
                depth_rows_iter = iter(())

            for row in tracking_rows:
                tracking = parse_tracking(row)
                if tracking.timestamp < last_tracking_timestamp:
                    raise ValueError("tracking CSV timestamps must be nondecreasing")
                last_tracking_timestamp = tracking.timestamp

                while next_depth is not None:
                    depth_timestamp = float(next_depth["timestamp"])
                    if depth_timestamp < last_depth_timestamp:
                        raise ValueError("depth CSV timestamps must be nondecreasing")
                    if depth_timestamp > tracking.timestamp + 1e-6:
                        break
                    engine.add_depth(parse_depth(next_depth))
                    depth_rows += 1
                    last_depth_timestamp = depth_timestamp
                    next_depth = _next_row(depth_rows_iter)

                if segmentation_worker is not None:
                    if tracking.frame < 1:
                        raise ValueError("semantic fusion requires one-based video frame IDs")
                    while video_frame < tracking.frame:
                        ok, current_frame = capture.read()
                        if not ok:
                            raise RuntimeError(
                                f"video ended before tracker frame {tracking.frame}"
                            )
                        video_frame += 1
                        if video_frame == 1 or video_frame % args.segmentation_interval == 0:
                            segmentation_worker.submit(
                                video_frame,
                                video_frame / video_fps,
                                current_frame,
                            )
                        if args.realtime:
                            delay = (
                                realtime_started + video_frame / video_fps
                                - time.perf_counter()
                            )
                            if delay > 0:
                                time.sleep(delay)
                    seg_result = segmentation_worker.get_latest()
                    context_result = scene_fusion.process(
                        tracking.frame,
                        tracking.timestamp,
                        current_frame,
                        [{
                            "track_id": tracking.track_id,
                            "yolo_class": tracking.class_name,
                            "bbox": tracking.bbox,
                        }],
                        seg_result,
                        None,
                    )
                    latencies_ms.append(context_result["processing_time_ms"])
                    context = context_result["object_contexts"][0]
                else:
                    context = None

                record = engine.fuse(tracking).to_dict()
                if context is not None:
                    if context["context_valid"]:
                        semantic_context_records += 1
                        engine.metrics["segmentation_available"] += 1
                    if (
                        context["context_valid"]
                        and context["segmentation_age_frames"] not in (None, 0)
                    ):
                        semantic_stale_records += 1
                    record.update({
                        "semantic_context": context["semantic_class_name"],
                        "semantic_class_id": context["semantic_class_id"],
                        "semantic_distribution": context["semantic_distribution"],
                        "semantic_confidence": context["semantic_confidence"],
                        "road_contact": context["road_contact"],
                        "sidewalk_contact": context["sidewalk_contact"],
                        "segmentation_age_frames": context["segmentation_age_frames"],
                    })
                    record["data_age"]["segmentation_seconds"] = (
                        max(0.0, tracking.timestamp - seg_result["timestamp"])
                        if context_result["segmentation_valid"] and seg_result else None
                    )
                    record["staleness"]["segmentation"] = (
                        "fresh" if context_result["segmentation_age"] == 0
                        else "stale" if context_result["segmentation_valid"]
                        else "unavailable"
                    )
                    record["risk_inputs"]["semantic_available"] = context["context_valid"]
                    record["risk_inputs"]["semantic_class_id"] = context["semantic_class_id"]
                _write_record(writer, record)
                records += 1
                if args.max_records is not None and records >= args.max_records:
                    source_complete = False
                    break
    finally:
        if capture is not None:
            capture.release()
        if depth_handle is not None:
            depth_handle.close()
        worker_errors = segmentation_worker.close() if segmentation_worker else []

    elapsed = time.perf_counter() - started
    summary = engine.summary()
    summary.update({
        "tracking_csv": str(args.tracking_csv),
        "depth_csv": str(args.depth_csv),
        "depth_input_rows_consumed": depth_rows,
        "fused_records": records,
        "source_complete": source_complete,
        "fusion_records_per_second": records / max(elapsed, 1e-9),
        "fusion_stage_elapsed_seconds": elapsed,
        "video_frames_per_second": (
            video_frame / max(elapsed, 1e-9) if scene_fusion else None
        ),
        "semantic_context_records": semantic_context_records,
        "semantic_stale_records": semantic_stale_records,
        "semantic_fusion_enabled": scene_fusion is not None,
        "semantic_fusion_latency_p50_ms": (
            float(np.percentile(latencies_ms, 50)) if latencies_ms else None
        ),
        "semantic_fusion_latency_p95_ms": (
            float(np.percentile(latencies_ms, 95)) if latencies_ms else None
        ),
        "segmentation": {
            "video": str(args.video) if scene_fusion else None,
            "checkpoint": str(args.segmentation_checkpoint) if scene_fusion else None,
            "source_frames_read": video_frame,
            "submitted": segmentation_worker.submitted if segmentation_worker else 0,
            "inferences_completed": segmentation_worker.inference_count if segmentation_worker else 0,
            "requests_dropped": segmentation_worker.dropped if segmentation_worker else 0,
            "raw_inference_fps": (
                segmentation_worker.inference_count
                / max(segmentation_worker.inference_seconds, 1e-9)
                if segmentation_worker else None
            ),
            "effective_update_rate_hz": (
                segmentation_worker.inference_count
                / max(video_frame / video_fps, 1e-9)
                if segmentation_worker and video_fps else None
            ),
            "worker_errors": worker_errors,
            "context_metrics": scene_fusion.metrics if scene_fusion else None,
        },
        "lane_input": "unavailable",
        "output_csv": str(args.output_csv),
        "contract_version": "fusion-v2",
        "measurement_scope": "offline tracker/depth fusion stage; not end-to-end camera perception",
    })
    summary_path = args.summary_json or args.output_csv.with_name("fusion_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if worker_errors:
        raise RuntimeError(
            f"Segmentation worker reported {len(worker_errors)} error(s); see {summary_path}"
        )


if __name__ == "__main__":
    main()
