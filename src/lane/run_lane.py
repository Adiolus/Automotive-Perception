"""Run the isolated lane detector/stitcher benchmark."""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from lane_bev import warp
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
    height, width = frame.shape[:2]
    for lane in state["lanes"]:
        source = np.float32(lane["polyline"]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(source, inverse).reshape(-1, 2)
        color = (0, 175, 255) if lane["visibility"] == "predicted" else (0, 255, 255)
        for index, (first, second) in enumerate(zip(projected, projected[1:])):
            if lane["visibility"] == "predicted" and index % 2:
                continue
            clipped, start, end = cv2.clipLine(
                (0, 0, width, height),
                tuple(np.rint(first).astype(int)),
                tuple(np.rint(second).astype(int)),
            )
            if clipped:
                cv2.line(frame, start, end, color, 2 if lane["visibility"] == "predicted" else 3)
        visible_points = projected[
            (projected[:, 0] >= 0) & (projected[:, 0] < width)
            & (projected[:, 1] >= 0) & (projected[:, 1] < height)
        ]
        if len(visible_points):
            x, y = np.rint(visible_points[-1]).astype(int)
            label = (
                f"Lane {lane['lane_id']} {lane['ego_relation']} "
                f"{lane['visibility']} {lane['confidence']:.2f}"
            )
            cv2.putText(frame, label, (x, y), 0, 0.45, color, 1, cv2.LINE_AA)
    boundaries = state.get("ego_lane_boundary_ids")
    ego_text = f"ego_bounds={boundaries}" if boundaries else "ego=uncertain"
    cv2.putText(frame, f"lanes={len(state['lanes'])} {ego_text}", (20, 32), 0, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_raw_detections(frame, observation, detector):
    if observation is None:
        cv2.putText(frame, "Raw lane detector: not sampled", (20, 32), 0, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    inverse = cv2.invert(detector.homography)[1]
    height, width = frame.shape[:2]
    for index, lane in enumerate(observation["lanes"], start=1):
        source = np.float32(lane["polyline"]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(source, inverse).reshape(-1, 2)
        for first, second in zip(projected, projected[1:]):
            clipped, start, end = cv2.clipLine(
                (0, 0, width, height),
                tuple(np.rint(first).astype(int)),
                tuple(np.rint(second).astype(int)),
            )
            if clipped:
                cv2.line(frame, start, end, (255, 180, 0), 2)
        visible = projected[
            (projected[:, 0] >= 0) & (projected[:, 0] < width)
            & (projected[:, 1] >= 0) & (projected[:, 1] < height)
        ]
        if len(visible):
            x, y = np.rint(visible[-1]).astype(int)
            cv2.putText(
                frame,
                f"raw {index} {lane['detection_confidence']:.2f}",
                (x, y),
                0,
                0.45,
                (255, 180, 0),
                1,
                cv2.LINE_AA,
            )
    cv2.putText(frame, f"Raw lane detections: {len(observation['lanes'])}", (20, 32), 0, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def draw_bev(observation, bev, sampled):
    image = bev.copy()
    if not sampled:
        cv2.putText(image, "Detector update skipped", (16, 28), 0, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return image
    for index, lane in enumerate(observation["lanes"], start=1):
        points = np.rint(np.asarray(lane["polyline"])).astype(int)
        for first, second in zip(points, points[1:]):
            clipped, start, end = cv2.clipLine(
                (0, 0, image.shape[1], image.shape[0]),
                tuple(first),
                tuple(second),
            )
            if clipped:
                cv2.line(image, start, end, (0, 255, 255), 2)
        if len(points):
            x, y = points[-1]
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                cv2.putText(image, f"raw {index}", (x, y), 0, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, f"BEV/IPM | raw detections={len(observation['lanes'])}", (16, 28), 0, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def create_writer(path, fps, size):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        size,
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"could not open video writer: {path}")
    return writer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=VIDEO)
    parser.add_argument("--config", type=Path, default=Path("src/lane/lane_config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    output_names = (
        "lane_visualization.mp4",
        "lane_tracking.mp4",
        "raw_lane_detections.mp4",
        "lane_bev_ipm.mp4",
        "lane_tracks.csv",
        "lane_benchmark.json",
        "lane_metrics.json",
        "lane_report.md",
    )
    existing_outputs = [name for name in output_names if (args.output_dir / name).exists()]
    if existing_outputs:
        parser.error(
            "refusing to overwrite existing output(s): "
            + ", ".join(existing_outputs)
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = config(args.config)
    detector = ClassicalLaneDetector(cfg)
    stitcher = LaneStitcher(cfg)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracking_writer = create_writer(args.output_dir / "lane_tracking.mp4", fps, (width, height))
    compatibility_writer = create_writer(args.output_dir / "lane_visualization.mp4", fps, (width, height))
    raw_writer = create_writer(args.output_dir / "raw_lane_detections.mp4", fps, (width, height))
    bev_writer = create_writer(
        args.output_dir / "lane_bev_ipm.mp4",
        fps,
        (int(cfg["bev_width"]), int(cfg["bev_height"])),
    )
    csv_file = (args.output_dir / "lane_tracks.csv").open("w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "timestamp", "lane_id", "polyline", "confidence", "lane_category", "lane_style", "lane_color", "ego_relation", "lane_offset", "boundary_distance", "visibility", "parent_lane_id", "child_lane_ids"])
    frame = 0; processed = 0; detections = 0; observations = 0; predicted = 0; started = time.perf_counter(); counts = []
    while True:
        if args.max_frames is not None and frame >= args.max_frames:
            break
        ok, image = cap.read()
        if not ok: break
        frame += 1
        observation = None
        if frame == 1 or frame % int(cfg["process_every_n_frames"]) == 0:
            observation = detector.detect(image, frame)
            processed += 1; detections += len(observation["lanes"])
            state = stitcher.update(observation, frame / fps)
        else:
            state = stitcher.hold(frame, frame / fps)
        observations += len(state["lanes"]); counts.append(len(state["lanes"]))
        predicted += sum(lane["visibility"] == "predicted" for lane in state["lanes"])
        for lane in state["lanes"]:
            csv_writer.writerow([frame, f"{frame / fps:.4f}", lane["lane_id"], json.dumps(lane["polyline"], separators=(",", ":")), f"{lane['confidence']:.4f}", "lane_marking", lane["style"], lane["color"], lane["ego_relation"], "", "", lane["visibility"], "", "[]"])
        rendered = draw(image.copy(), state, detector)
        tracking_writer.write(rendered)
        compatibility_writer.write(rendered)
        raw_writer.write(draw_raw_detections(image.copy(), observation, detector))
        bev_image = warp(image, detector.homography, cfg)
        bev_writer.write(draw_bev(observation or {"lanes": []}, bev_image, observation is not None))
    cap.release()
    tracking_writer.release()
    compatibility_writer.release()
    raw_writer.release()
    bev_writer.release()
    csv_file.close()
    elapsed = time.perf_counter() - started
    summary = {
        "model": "classical_color_gradient_hough",
        "frames": frame,
        "detector_updates": processed,
        "raw_lane_fps": processed / max(elapsed, 1e-9),
        "effective_lane_update_rate": processed / max(frame / fps, 1e-9),
        "complete_lane_loop_fps": frame / max(elapsed, 1e-9),
        "mean_lanes_per_frame": sum(counts) / max(len(counts), 1),
        "lane_observations": detections,
        "lane_track_observations": observations,
        "predicted_lane_observations": predicted,
        "persistent_lane_ids_created": stitcher.created,
        "lane_id_switches": None,
        "lane_id_switch_measurement": "unavailable without lane identity ground truth",
        "calibration": "unavailable; provisional projective BEV only",
        "ground_truth_metrics": "unavailable; no lane labels in repository",
        "scenario_evaluation": "see scenario_matrix.md",
    }
    (args.output_dir / "lane_benchmark.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "lane_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "lane_report.md").write_text("# Lane Benchmark\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
