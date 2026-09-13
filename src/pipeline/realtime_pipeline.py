"""Nonblocking execution shell for frozen perception records and visualization."""

import argparse
import csv
import json
import queue
import threading
import time
from collections import defaultdict
from pathlib import Path

import cv2


VIDEO = Path("videos/indian_road.mp4")
RISK_CSV = Path("runs/risk_eval/risk_tracks.csv")
OUTPUT_DIR = Path("runs/pipeline_eval")


class VisualizationWorker:
    def __init__(self, output_path, fps, size, queue_size):
        self.output_path = output_path
        self.fps = fps
        self.size = size
        self.frames = queue.Queue(maxsize=queue_size)
        self.stop_requested = False
        self.dropped_frames = 0
        self.written_frames = 0
        self.started = None
        self.finished = None
        self.writer = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, frame_index, timestamp, frame):
        packet = (frame_index, timestamp, frame)
        try:
            self.frames.put_nowait(packet)
            return True
        except queue.Full:
            self.dropped_frames += 1
            return False

    def close(self):
        self.stop_requested = True
        self.thread.join(timeout=120.0)

    def _run(self):
        self.started = time.perf_counter()
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            self.size,
        )
        while not self.stop_requested or not self.frames.empty():
            try:
                _, _, frame = self.frames.get(timeout=0.05)
            except queue.Empty:
                continue
            self.writer.write(frame)
            self.written_frames += 1
            self.frames.task_done()
        self.writer.release()
        self.finished = time.perf_counter()


def load_states(path):
    by_frame = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            row["frame"] = int(row["frame"])
            row["track_id"] = str(row["track_id"])
            row["bbox"] = [float(value) for value in json.loads(row["bbox"])]
            by_frame[row["frame"]].append(row)
    return by_frame


def overlay(frame, states, frame_index):
    colors = {
        "SAFE": (80, 190, 80),
        "LOW": (180, 200, 80),
        "MEDIUM": (0, 190, 255),
        "HIGH": (0, 100, 255),
        "CRITICAL": (0, 0, 255),
    }
    for state in states:
        x1, y1, x2, y2 = [int(value) for value in state["bbox"]]
        level = state["risk_level"]
        color = colors.get(level, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        ttc = state.get("filtered_ttc") or "--"
        closing = state.get("smoothed_closing_speed") or "--"
        label = f"ID {state['track_id']} {state['class']} {level} TTC:{ttc} C:{closing}"
        cv2.putText(frame, label, (x1, max(18, y1 - 6)), 0, 0.43, color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Perception frame {frame_index}", (20, 32), 0, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=VIDEO)
    parser.add_argument("--risk-csv", type=Path, default=RISK_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    states = load_states(args.risk_csv)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_video = args.output_dir / "final_visualization.mp4"
    visualization = VisualizationWorker(output_video, fps, (width, height), args.queue_size)
    started = time.perf_counter()
    realtime_started = started
    frames = 0
    perception_records = 0
    visualization_submitted = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        frame_states = states.get(frames, [])
        perception_records += len(frame_states)
        rendered = overlay(frame, frame_states, frames)
        if visualization.submit(frames, frames / fps, rendered):
            visualization_submitted += 1
        if args.realtime:
            target = realtime_started + frames / fps
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
    cap.release()
    perception_finished = time.perf_counter()
    visualization.close()
    finished = time.perf_counter()
    metrics = {
        "frames": frames,
        "source_total_frames": total_frames,
        "perception_records_processed": perception_records,
        "perception_records_dropped": 0,
        "perception_fps": frames / max(perception_finished - started, 1e-9),
        "visualization_frames_submitted": visualization_submitted,
        "visualization_frames_written": visualization.written_frames,
        "dropped_visualization_frames": visualization.dropped_frames,
        "visualization_fps": visualization.written_frames / max((visualization.finished or finished) - (visualization.started or started), 1e-9),
        "recorded_video_fps": visualization.written_frames / max(finished - started, 1e-9),
        "total_processing_seconds": finished - started,
        "queue_size": args.queue_size,
        "visualization_output": str(output_video),
        "perception_input": str(args.risk_csv),
        "architecture": "frozen YOLO/tracker/depth/fusion/risk state -> nonblocking visualization",
        "perception_state_dropped": False,
        "realtime_pacing": args.realtime,
    }
    (args.output_dir / "realtime_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    report = (
        "# Real-Time Execution Report\n\n"
        "Perception records are processed synchronously and never dropped. Visualization uses a bounded producer-consumer queue; only visualization frames may be dropped.\n\n"
        "```json\n" + json.dumps(metrics, indent=2) + "\n```\n"
    )
    (args.output_dir / "realtime_report.md").write_text(report)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
