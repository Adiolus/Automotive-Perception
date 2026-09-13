"""Asynchronous Depth Anything V2 Small pipeline for the project road video."""

import argparse
import csv
import json
import math
import os
import queue
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


MODEL_NAME = "depth-anything/Depth-Anything-V2-Small-hf"
VIDEO_PATH = Path("videos/indian_road.mp4")
TRACKER_CSV = Path("runs/tracking_eval/FINAL_TRACKER_PRODUCTION.csv")
OUTPUT_ROOT = Path("runs/depth_eval")


class DepthWorker:
    """Single-slot asynchronous depth worker; stale requests are discarded."""

    def __init__(self, model_name, device):
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self.requests = queue.Queue(maxsize=1)
        self.latest = None
        self.latest_lock = threading.Lock()
        self.inference_count = 0
        self.inference_seconds = 0.0
        self.errors = []
        self.stop_requested = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, frame_index, frame_bgr):
        request = (frame_index, frame_bgr.copy())
        try:
            self.requests.put_nowait(request)
            return True
        except queue.Full:
            return False

    def get_latest(self):
        with self.latest_lock:
            return self.latest

    def close(self):
        self.stop_requested = True
        self.thread.join(timeout=30.0)

    def _run(self):
        while not self.stop_requested or not self.requests.empty():
            try:
                frame_index, frame_bgr = self.requests.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                started = time.perf_counter()
                height, width = frame_bgr.shape[:2]
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                inputs = self.processor(
                    images=Image.fromarray(rgb),
                    return_tensors="pt",
                )
                inputs = {
                    key: value.to(self.device)
                    for key, value in inputs.items()
                }
                with torch.inference_mode():
                    outputs = self.model(**inputs)
                depth = outputs.predicted_depth
                depth = torch.nn.functional.interpolate(
                    depth.unsqueeze(1),
                    size=(height, width),
                    mode="bicubic",
                    align_corners=False,
                )[0, 0]
                depth = depth.detach().float().cpu().numpy()
                if not np.isfinite(depth).any():
                    raise RuntimeError("depth output contained no finite values")
                finite = np.isfinite(depth)
                low = float(np.percentile(depth[finite], 1.0))
                high = float(np.percentile(depth[finite], 99.0))
                normalized = np.clip((depth - low) / max(high - low, 1e-6), 0.0, 1.0)
                elapsed = time.perf_counter() - started
                with self.latest_lock:
                    self.latest = {
                        "frame": frame_index,
                        "depth": depth.astype(np.float32),
                        "normalized": normalized.astype(np.float32),
                        "inference_seconds": elapsed,
                        "statistics": {
                            "finite_pixels": int(finite.sum()),
                            "min": float(np.min(depth[finite])),
                            "median": float(np.median(depth[finite])),
                            "mean": float(np.mean(depth[finite])),
                            "p90": float(np.percentile(depth[finite], 90.0)),
                            "max": float(np.max(depth[finite])),
                        },
                    }
                self.inference_count += 1
                self.inference_seconds += elapsed
            except Exception as error:  # keep video/tracker loop alive
                self.errors.append(f"frame {frame_index}: {error}")
            finally:
                self.requests.task_done()


def load_tracks(path):
    tracks = defaultdict(list)
    if not path or not path.exists():
        return tracks
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            tracks[frame].append(row)
    return tracks


def robust_depth(depth_map, bbox):
    """Estimate relative depth from an interior region, excluding box edges."""
    height, width = depth_map.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in bbox]
    x1, x2 = sorted((max(0.0, x1), min(float(width), x2)))
    y1, y2 = sorted((max(0.0, y1), min(float(height), y2)))
    box_width = max(2.0, x2 - x1)
    box_height = max(2.0, y2 - y1)
    # Favor the object interior and especially the torso/vehicle body region.
    ix1 = int(x1 + 0.20 * box_width)
    ix2 = int(x2 - 0.20 * box_width)
    iy1 = int(y1 + 0.15 * box_height)
    iy2 = int(y2 - 0.15 * box_height)
    crop = depth_map[max(0, iy1):min(height, iy2), max(0, ix1):min(width, ix2)]
    values = crop[np.isfinite(crop)]
    if values.size < 4:
        return math.nan, 0.0, math.nan
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 1e-6:
        values = values[np.abs(values - median) <= 3.0 * mad]
    if values.size == 0:
        return median, 0.0, mad
    robust = float(np.median(values))
    quality = float(min(1.0, values.size / max(16.0, crop.size)) * (1.0 / (1.0 + 4.0 * mad)))
    return robust, quality, mad


def colorize(normalized):
    image = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(image, cv2.COLORMAP_INFERNO)


def run_schedule(schedule, args):
    output_dir = OUTPUT_ROOT / f"schedule_{schedule}"
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks = load_tracks(args.tracker_csv)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    worker = DepthWorker(MODEL_NAME, device)
    cap = cv2.VideoCapture(str(args.video))
    loop_started = time.perf_counter()
    realtime_started = loop_started
    if not cap.isOpened():
        raise RuntimeError(f"could not open {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None
    if args.visualize:
        writer = cv2.VideoWriter(
            str(output_dir / "depth_preview.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            video_fps,
            (width, height),
        )
    track_output = output_dir / "per_track_depth.csv"
    stats_output = output_dir / "depth_frame_stats.csv"
    cache_dir = output_dir / "depth_cache"
    if args.save_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
    history = defaultdict(lambda: deque(maxlen=8))
    latest = None
    frame_index = 0
    submitted = 0
    completed_depth_frames = set()
    rows_written = 0
    with track_output.open("w", newline="") as handle, stats_output.open("w", newline="") as stats_handle:
        writer_csv = csv.writer(handle)
        stats_csv = csv.writer(stats_handle)
        stats_csv.writerow(["frame", "finite_pixels", "min", "median", "mean", "p90", "max", "inference_seconds"])
        writer_csv.writerow([
            "frame", "timestamp", "track_id", "class", "x1", "y1", "x2", "y2",
            "depth", "depth_quality", "depth_mad", "depth_frame", "depth_age_frames",
            "depth_previous", "depth_change", "trend",
        ])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if frame_index == 1 or frame_index % schedule == 0:
                if worker.submit(frame_index, frame):
                    submitted += 1
            candidate = worker.get_latest()
            if candidate is not None and (
                latest is None or candidate["frame"] > latest["frame"]
            ):
                latest = candidate
                completed_depth_frames.add(candidate["frame"])
                stats = candidate["statistics"]
                stats_csv.writerow([
                    candidate["frame"], stats["finite_pixels"], stats["min"],
                    stats["median"], stats["mean"], stats["p90"], stats["max"],
                    candidate["inference_seconds"],
                ])
                if args.save_cache:
                    np.savez_compressed(
                        cache_dir / f"depth_{candidate['frame']:06d}.npz",
                        depth=candidate["depth"].astype(np.float16),
                        normalized=candidate["normalized"].astype(np.float16),
                        frame=np.int32(candidate["frame"]),
                    )
            if latest is not None:
                for row in tracks.get(frame_index, []):
                    bbox = [row[key] for key in ("x1", "y1", "x2", "y2")]
                    value, quality, mad = robust_depth(latest["depth"], bbox)
                    track_id = row["track_id"]
                    previous = history[track_id][-1] if history[track_id] else math.nan
                    change = value - previous if np.isfinite(previous) and np.isfinite(value) else math.nan
                    history[track_id].append(value)
                    recent = [item for item in history[track_id] if np.isfinite(item)]
                    trend = "stable"
                    if len(recent) >= 3:
                        slope = np.polyfit(np.arange(len(recent)), recent, 1)[0]
                        if slope > 0.01:
                            trend = "receding"
                        elif slope < -0.01:
                            trend = "approaching"
                    writer_csv.writerow([
                        frame_index, f"{frame_index / video_fps:.4f}", track_id, row["class"],
                        *[row[key] for key in ("x1", "y1", "x2", "y2")],
                        f"{value:.6f}" if np.isfinite(value) else "",
                        f"{quality:.6f}", f"{mad:.6f}" if np.isfinite(mad) else "",
                        latest["frame"], frame_index - latest["frame"],
                        f"{previous:.6f}" if np.isfinite(previous) else "",
                        f"{change:.6f}" if np.isfinite(change) else "", trend,
                    ])
                    rows_written += 1
                if writer is not None:
                    depth_color = colorize(latest["normalized"])
                    visual = cv2.addWeighted(frame, 0.58, depth_color, 0.42, 0.0)
                    cv2.putText(visual, f"Depth frame {latest['frame']} age {frame_index - latest['frame']}", (20, 32), 0, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    writer.write(visual)
            elif writer is not None:
                writer.write(frame)
            if args.realtime:
                target_time = realtime_started + frame_index / video_fps
                delay = target_time - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
    cap.release()
    if writer is not None:
        writer.release()
    worker.close()
    elapsed = time.perf_counter() - loop_started
    report = {
        "model": MODEL_NAME,
        "device": device,
        "input_resolution": [width, height],
        "video_fps": video_fps,
        "frames": frame_index,
        "schedule_interval": schedule,
        "submitted_depth_requests": submitted,
        "completed_depth_frames": len(completed_depth_frames),
        "raw_depth_inference_fps": worker.inference_count / max(worker.inference_seconds, 1e-6),
        "scheduled_depth_update_fps": worker.inference_count / max(frame_index / video_fps, 1e-6),
        "end_to_end_fps": frame_index / max(elapsed, 1e-6),
        "depth_inference_count": worker.inference_count,
        "depth_inference_seconds": worker.inference_seconds,
        "depth_frame_percentage": 100.0 * worker.inference_count / max(frame_index, 1),
        "track_rows": rows_written,
        "worker_errors": worker.errors[:10],
        "output_directory": str(output_dir),
    }
    (output_dir / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=VIDEO_PATH)
    parser.add_argument("--tracker-csv", type=Path, default=TRACKER_CSV)
    parser.add_argument("--schedule", type=int, choices=(3, 5, 8), default=5)
    parser.add_argument("--benchmark-all", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--save-cache", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    args = parser.parse_args()
    schedules = (3, 5, 8) if args.benchmark_all else (args.schedule,)
    reports = [run_schedule(schedule, args) for schedule in schedules]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "benchmark_report.json").write_text(json.dumps(reports, indent=2) + "\n")
    lines = [
        "# Depth Anything V2 Small Benchmark",
        "",
        f"Model: `{MODEL_NAME}`",
        f"Video: `{args.video}`",
        "",
        "| Schedule | Depth updates | Raw depth FPS | Scheduled rate | End-to-end FPS |",
        "|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        lines.append(
            f"| every {report['schedule_interval']} frames | {report['completed_depth_frames']} | "
            f"{report['raw_depth_inference_fps']:.2f} | {report['scheduled_depth_update_fps']:.2f} | "
            f"{report['end_to_end_fps']:.2f} |"
        )
    lines += ["", "Depth maps are relative/inverse-depth estimates; larger normalized values indicate nearer surfaces."]
    (OUTPUT_ROOT / "benchmark_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
