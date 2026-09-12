"""Run the isolated lane detector/stitcher benchmark."""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2

from lane_detector import ClassicalLaneDetector
from lane_stitcher import LaneStitcher


VIDEO = Path("videos/indian_road.mp4")
OUTPUT = Path("runs/lane_eval")


def config(path):
    values = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        values[key] = float(value)
    return values


def draw(frame, state, detector):
    inverse = cv2.invert(detector.homography)[1]
    for lane in state["lanes"]:
        source = __import__("numpy").float32(lane["polyline"]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(source, inverse).reshape(-1, 2)
        points = [(int(x), int(y)) for x, y in projected if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]]
        for first, second in zip(points, points[1:]):
            cv2.line(frame, first, second, (0, 255, 255), 3)
        if points:
            cv2.putText(frame, f"Lane {lane['lane_id']} {lane['ego_relation']} {lane['confidence']:.2f}", points[-1], 0, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"lanes={len(state['lanes'])} ego={state['ego_lane_id']}", (20, 32), 0, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=VIDEO)
    parser.add_argument("--config", type=Path, default=Path("src/lane/lane_config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = config(args.config)
    detector = ClassicalLaneDetector(cfg)
    stitcher = LaneStitcher(cfg)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(str(args.output_dir / "lane_visualization.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    csv_file = (args.output_dir / "lane_tracks.csv").open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "timestamp", "lane_id", "polyline", "confidence", "lane_category", "lane_style", "lane_color", "ego_relation", "lane_offset", "boundary_distance", "visibility", "parent_lane_id", "child_lane_ids"])
    frame = 0; processed = 0; detections = 0; observations = 0; started = time.perf_counter(); counts = []
    while True:
        ok, image = cap.read()
        if not ok: break
        frame += 1
        if frame == 1 or frame % int(cfg["process_every_n_frames"]) == 0:
            observation = detector.detect(image, frame)
            processed += 1; detections += len(observation["lanes"])
            state = stitcher.update(observation, frame / fps)
        else:
            state = stitcher.last_state or {"frame": frame, "timestamp": frame / fps, "lanes": [], "ego_lane_id": None}
        observations += len(state["lanes"]); counts.append(len(state["lanes"]))
        for lane in state["lanes"]:
            csv_writer.writerow([frame, f"{frame / fps:.4f}", lane["lane_id"], json.dumps(lane["polyline"], separators=(",", ":")), f"{lane['confidence']:.4f}", "lane_marking", lane["style"], lane["color"], lane["ego_relation"], "", "", lane["visibility"], "", "[]"])
        writer.write(draw(image, state, detector))
    cap.release(); writer.release(); csv_file.close(); elapsed = time.perf_counter() - started
    summary = {"model": "classical_color_gradient_hough", "frames": frame, "detector_updates": processed, "raw_lane_fps": processed / max(elapsed, 1e-9), "effective_lane_update_rate": processed / max(frame / fps, 1e-9), "complete_lane_loop_fps": frame / max(elapsed, 1e-9), "mean_lanes_per_frame": sum(counts) / max(len(counts), 1), "lane_observations": detections, "lane_track_observations": observations, "persistent_lane_ids_created": stitcher.created, "lane_id_switches": stitcher.switches, "calibration": "unavailable; provisional projective BEV only", "ground_truth_metrics": "unavailable; no lane labels in repository", "scenario_evaluation": "see scenario_matrix.md"}
    (args.output_dir / "lane_benchmark.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "lane_report.md").write_text("# Lane Benchmark\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
