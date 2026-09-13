"""Run contract-based fusion with real tracker/depth inputs and optional mocks."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fusion import FusionEngine, parse_config, parse_depth, parse_tracking


DEFAULT_TRACKING = Path("runs/tracking_eval/FINAL_TRACKER_PRODUCTION.csv")
DEFAULT_DEPTH = Path("runs/depth_eval/schedule_5/per_track_depth.csv")
DEFAULT_CONFIG = HERE / "fusion_config.yaml"
DEFAULT_OUTPUT = Path("runs/fusion_eval/fused_tracks.csv")


def load_depth(engine, path):
    if not path or not path.exists():
        return 0
    count = 0
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            engine.add_depth(parse_depth(row))
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-csv", type=Path, default=DEFAULT_TRACKING)
    parser.add_argument("--depth-csv", type=Path, default=DEFAULT_DEPTH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args()

    config = parse_config(args.config)
    engine = FusionEngine(config)
    depth_rows = load_depth(engine, args.depth_csv)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fused_count = 0
    with args.output_csv.open("w", newline="") as output_handle:
        writer = csv.DictWriter(
            output_handle,
            fieldnames=[
                "frame", "timestamp", "track_id", "class", "bbox", "velocity",
                "trajectory", "depth", "depth_quality", "depth_change", "depth_history",
                "depth_trend", "closing_speed", "ttc", "ttc_valid", "ttc_reason",
                "lane_relation", "lane_center_offset", "lane_crossing", "lane_conflict",
                "lane_confidence", "semantic_context", "semantic_confidence", "confidence",
                "risk_inputs", "data_age", "staleness",
            ],
        )
        writer.writeheader()
        with args.tracking_csv.open(newline="") as tracking_handle:
            for row in csv.DictReader(tracking_handle):
                record = engine.fuse(parse_tracking(row)).to_dict()
                writer.writerow({
                    "frame": record["frame"],
                    "timestamp": record["timestamp"],
                    "track_id": record["track_id"],
                    "class": record["class_name"],
                    "bbox": json.dumps(record["bbox"], separators=(",", ":")),
                    "velocity": json.dumps(record["velocity"], separators=(",", ":")),
                    "trajectory": json.dumps(record["trajectory"], separators=(",", ":")),
                    "depth": record["depth"],
                    "depth_quality": record["depth_quality"],
                    "depth_change": record["depth_change"],
                    "depth_history": json.dumps(record["depth_history"], separators=(",", ":")),
                    "depth_trend": record["depth_trend"],
                    "closing_speed": record["closing_speed"],
                    "ttc": record["ttc"],
                    "ttc_valid": record["ttc_valid"],
                    "ttc_reason": record["ttc_reason"],
                    "lane_relation": record["lane_relation"],
                    "lane_center_offset": record["lane_center_offset"],
                    "lane_crossing": record["lane_crossing"],
                    "lane_conflict": record["lane_conflict"],
                    "lane_confidence": record["lane_confidence"],
                    "semantic_context": record["semantic_context"],
                    "semantic_confidence": record["semantic_confidence"],
                    "confidence": record["confidence"],
                    "risk_inputs": json.dumps(record["risk_inputs"], separators=(",", ":")),
                    "data_age": json.dumps(record["data_age"], separators=(",", ":")),
                    "staleness": json.dumps(record["staleness"], separators=(",", ":")),
                })
                fused_count += 1
    elapsed = time.perf_counter() - started
    summary = engine.summary()
    summary.update({
        "tracking_csv": str(args.tracking_csv),
        "depth_csv": str(args.depth_csv),
        "depth_input_rows": depth_rows,
        "fused_records": fused_count,
        "processing_fps": fused_count / max(elapsed, 1e-9),
        "segmentation_input": "unavailable",
        "lane_input": "unavailable",
        "output_csv": str(args.output_csv),
        "contract_version": "fusion-v1",
    })
    summary_path = args.summary_json or args.output_csv.with_name("fusion_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
