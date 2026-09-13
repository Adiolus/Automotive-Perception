"""Quantify tracker fragmentation and duplicate outputs from tracker CSV files."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def iou(first, second):
    intersection_width = max(
        0.0,
        min(first[2], second[2]) - max(first[0], second[0]),
    )
    intersection_height = max(
        0.0,
        min(first[3], second[3]) - max(first[1], second[1]),
    )
    intersection = intersection_width * intersection_height
    first_area = max(1.0, first[2] - first[0]) * max(1.0, first[3] - first[1])
    second_area = max(1.0, second[2] - second[0]) * max(1.0, second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def load_rows(path):
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "frame": int(row["frame"]),
                    "track_id": int(row["track_id"]),
                    "class": row["class"],
                    "box": [
                        float(row["x1"]),
                        float(row["y1"]),
                        float(row["x2"]),
                        float(row["y2"]),
                    ],
                }
            )
    return rows


def union(parent, first, second):
    first_root = first
    while parent[first_root] != first_root:
        parent[first_root] = parent[parent[first_root]]
        first_root = parent[first_root]
    second_root = second
    while parent[second_root] != second_root:
        parent[second_root] = parent[parent[second_root]]
        second_root = parent[second_root]
    if first_root != second_root:
        parent[second_root] = first_root


def analyze(path, overlap_threshold):
    rows = load_rows(path)
    by_frame = defaultdict(list)
    by_id = defaultdict(list)
    for index, row in enumerate(rows):
        row["index"] = index
        by_frame[row["frame"]].append(row)
        by_id[row["track_id"]].append(row)

    durations = [len(items) for items in by_id.values()]
    duplicate_pairs = 0
    duplicate_frames = 0
    handoffs = 0
    handoff_classes = defaultdict(int)

    for frame_rows in by_frame.values():
        frame_pairs = 0
        for first_index, first in enumerate(frame_rows):
            for second in frame_rows[first_index + 1 :]:
                if (
                    first["class"] == second["class"]
                    and iou(first["box"], second["box"]) >= overlap_threshold
                ):
                    frame_pairs += 1
        duplicate_pairs += frame_pairs
        duplicate_frames += int(frame_pairs > 0)

    parent = {row["index"]: row["index"] for row in rows}
    frames = sorted(by_frame)
    for frame in frames[:-1]:
        for first in by_frame[frame]:
            candidates = [
                second
                for second in by_frame[frame + 1]
                if second["class"] == first["class"]
                and iou(first["box"], second["box"]) >= overlap_threshold
            ]
            if not candidates:
                continue
            best = max(candidates, key=lambda item: iou(first["box"], item["box"]))
            same_id = any(
                item["track_id"] == first["track_id"] for item in candidates
            )
            if not same_id:
                handoffs += 1
                handoff_classes[first["class"]] += 1
            union(parent, first["index"], best["index"])

    components = defaultdict(list)
    for row in rows:
        root = row["index"]
        while parent[root] != root:
            root = parent[root]
        components[root].append(row)

    class_ids = defaultdict(set)
    for row in rows:
        class_ids[row["class"]].add(row["track_id"])

    physical_proxy_count = 0
    fragmented_proxy_count = 0
    ids_per_proxy = []
    for component in components.values():
        track_ids = {row["track_id"] for row in component}
        if len(component) < 2:
            continue
        physical_proxy_count += 1
        ids_per_proxy.append(len(track_ids))
        fragmented_proxy_count += int(len(track_ids) > 1)

    return {
        "file": str(path),
        "rows": len(rows),
        "frames": len(by_frame),
        "unique_ids": len(by_id),
        "classes": {
            name: len(track_ids)
            for name, track_ids in sorted(class_ids.items())
        },
        "duration": {
            "min": min(durations, default=0),
            "median": sorted(durations)[len(durations) // 2] if durations else 0,
            "max": max(durations, default=0),
            "under_5": sum(duration < 5 for duration in durations),
            "under_15": sum(duration < 15 for duration in durations),
            "under_30": sum(duration < 30 for duration in durations),
        },
        "duplicate_tracks": {
            "iou_threshold": overlap_threshold,
            "pairs": duplicate_pairs,
            "frames": duplicate_frames,
        },
        "association": {
            "high_overlap_handoffs": handoffs,
            "handoff_classes": dict(sorted(handoff_classes.items())),
        },
        "physical_trajectory_proxy": {
            "components": physical_proxy_count,
            "components_with_multiple_ids": fragmented_proxy_count,
            "ids_per_component_max": max(ids_per_proxy, default=0),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_files", nargs="+", type=Path)
    parser.add_argument("--overlap-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = [analyze(path, args.overlap_threshold) for path in args.csv_files]
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()