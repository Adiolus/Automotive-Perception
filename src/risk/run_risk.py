"""Run stateful risk evaluation and render a risk overlay."""

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from risk_engine import RiskEngine, parse_config


FUSED_INPUT = Path("runs/fusion_eval/fused_tracks.csv")
VIDEO_INPUT = Path("videos/indian_road.mp4")
OUTPUT_DIR = Path("runs/risk_eval")


def dump_record(record):
    result = dict(record)
    for key in ("bbox", "velocity", "trajectory", "depth_history", "risk_inputs", "data_age", "staleness"):
        try:
            result[key] = json.loads(result.get(key, "{}" if key != "trajectory" and key != "depth_history" else "[]"))
        except json.JSONDecodeError:
            result[key] = {} if key not in ("trajectory", "depth_history") else []
    return result


def draw_risk(frame, rows):
    colors = {
        "SAFE": (80, 190, 80),
        "LOW": (180, 200, 80),
        "MEDIUM": (0, 190, 255),
        "HIGH": (0, 100, 255),
        "CRITICAL": (0, 0, 255),
    }
    for row in rows:
        x1, y1, x2, y2 = [int(value) for value in row["bbox"]]
        level = row["risk_level"]
        color = colors[level]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        ttc = row.get("filtered_ttc")
        ttc_text = f"TTC {ttc:.2f}" if ttc is not None else "TTC --"
        label = (
            f"ID {row['track_id']} {row['class_name']} {level} | "
            f"D {row.get('depth') if row.get('depth') is not None else '--'} | "
            f"{ttc_text} | C {row.get('smoothed_closing_speed') if row.get('smoothed_closing_speed') is not None else '--'}"
        )
        cv2.putText(frame, label, (x1, max(18, y1 - 6)), 0, 0.43, color, 1, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused-csv", type=Path, default=FUSED_INPUT)
    parser.add_argument("--video", type=Path, default=VIDEO_INPUT)
    parser.add_argument("--config", type=Path, default=Path("src/risk/risk_config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    engine = RiskEngine(parse_config(args.config))
    output_csv = args.output_dir / "risk_tracks.csv"
    summary_path = args.output_dir / "risk_summary.json"
    rows_by_frame = defaultdict(list)
    level_counts = Counter()
    approach_count = 0
    valid_ttc_count = 0
    records = 0
    persistence_samples = []
    previous_levels = {}
    track_level_runs = defaultdict(int)
    started = time.perf_counter()
    with output_csv.open("w", newline="") as output_handle, args.fused_csv.open(newline="") as input_handle:
        input_rows = csv.DictReader(input_handle)
        fieldnames = [
            "frame", "timestamp", "track_id", "class", "bbox", "velocity", "trajectory",
            "depth", "smoothed_depth", "depth_change", "closing_speed", "smoothed_closing_speed",
            "raw_ttc", "filtered_ttc", "ttc", "ttc_valid", "ttc_reason", "approaching",
            "approach_persistence_frames", "trajectory_stability", "risk_score", "risk_level",
            "risk_confidence", "dominant_risk_reason", "secondary_risk_reasons", "lane_conflict",
            "semantic_context", "confidence", "risk_inputs", "data_age", "staleness",
        ]
        writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
        writer.writeheader()
        for input_row in input_rows:
            fused = dump_record(input_row)
            risk = engine.evaluate({
                **input_row,
                **fused,
                "track_id": input_row["track_id"],
            })
            level = risk["risk_level"]
            level_counts[level] += 1
            records += 1
            if risk["approaching"]:
                approach_count += 1
            if str(input_row.get("ttc_valid", "")).lower() == "true":
                valid_ttc_count += 1
            track_level_runs[risk["track_id"]] += 1
            if previous_levels.get(risk["track_id"]) != level:
                if risk["track_id"] in previous_levels:
                    persistence_samples.append(track_level_runs[risk["track_id"]])
                track_level_runs[risk["track_id"]] = 1
            previous_levels[risk["track_id"]] = level
            rows_by_frame[int(risk["frame"])].append(risk)
            writer.writerow({
                "frame": risk["frame"], "timestamp": risk["timestamp"], "track_id": risk["track_id"],
                "class": risk["class_name"], "bbox": json.dumps(risk["bbox"], separators=(",", ":")),
                "velocity": json.dumps(risk["velocity"], separators=(",", ":")), "trajectory": json.dumps(risk["trajectory"], separators=(",", ":")),
                "depth": risk["depth"], "smoothed_depth": risk["smoothed_depth"], "depth_change": risk["depth_change"],
                "closing_speed": risk["closing_speed"], "smoothed_closing_speed": risk["smoothed_closing_speed"],
                "raw_ttc": risk["raw_ttc"], "filtered_ttc": risk["filtered_ttc"], "ttc": risk["ttc"],
                "ttc_valid": risk["ttc_valid"], "ttc_reason": risk["ttc_reason"], "approaching": risk["approaching"],
                "approach_persistence_frames": risk["approach_persistence_frames"], "trajectory_stability": risk["trajectory_stability"],
                "risk_score": risk["risk_score"], "risk_level": level, "risk_confidence": risk["risk_confidence"],
                "dominant_risk_reason": risk["dominant_risk_reason"], "secondary_risk_reasons": json.dumps(risk["secondary_risk_reasons"]),
                "lane_conflict": risk["lane_conflict"], "semantic_context": risk["semantic_context"], "confidence": risk["confidence"],
                "risk_inputs": json.dumps(risk["risk_inputs"], separators=(",", ":")), "data_age": json.dumps(risk["data_age"], separators=(",", ":")),
                "staleness": json.dumps(risk["staleness"], separators=(",", ":")),
            })
    video = cv2.VideoCapture(str(args.video))
    if not video.isOpened():
        raise RuntimeError(f"could not open {args.video}")
    fps = video.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = args.output_dir / "risk_visualization.mp4"
    writer_video = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame = 0
    while True:
        ok, image = video.read()
        if not ok:
            break
        frame += 1
        image = draw_risk(image, rows_by_frame.get(frame, []))
        cv2.putText(image, f"Risk frame {frame}/{int(video.get(cv2.CAP_PROP_FRAME_COUNT))}", (20, 32), 0, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        writer_video.write(image)
    video.release()
    writer_video.release()
    elapsed = time.perf_counter() - started
    summary = {
        "total_tracks_evaluated": len(engine.states),
        "total_track_time_records": records,
        "valid_ttc_percentage": 100.0 * valid_ttc_count / max(records, 1),
        "approaching_object_percentage": 100.0 * approach_count / max(records, 1),
        "risk_distribution": {
            level: level_counts.get(level, 0)
            for level in ("SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL")
        },
        "high_events": engine.high_events,
        "critical_events": engine.critical_events,
        "average_risk_persistence_records": sum(persistence_samples) / max(len(persistence_samples), 1),
        "risk_state_transitions": engine.transition_count,
        "processing_fps": records / max(elapsed, 1e-9),
        "risk_config": str(args.config),
        "fused_input": str(args.fused_csv),
        "segmentation": "unavailable",
        "lane": "unavailable",
        "output_csv": str(output_csv),
        "output_video": str(output_video),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    report = args.output_dir / "risk_report.md"
    report.write_text(
        "# Risk Engine Report\n\n"
        + "Stateful class-aware risk over frozen fusion output. TTC is a relative-depth proxy, not metric seconds.\n\n"
        + "```json\n" + json.dumps(summary, indent=2) + "\n```\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
