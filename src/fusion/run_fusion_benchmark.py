import cv2
import time
import json
import csv
import numpy as np
from semantic_fusion_interface import SceneFusionInterface

def draw_debug_overlay(frame, contexts):
    debug_frame = frame.copy()
    for ctx in contexts:
        x1, y1, x2, y2 = ctx['bbox']

        # Visual styling
        color = (0, 255, 0) if ctx['road_contact'] else (0, 0, 255)
        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)

        # Metadata Block
        text_lines = [
            f"Trk {ctx['track_id']} | YOLO: {ctx['yolo_class']}",
            f"Sem: {ctx['semantic_class']} (Conf: {ctx['semantic_confidence']:.2f})",
            f"Depth: {ctx['depth_type']} {ctx['depth_value']:.2f}",
            f"Road: {'YES' if ctx['road_contact'] else 'NO'}"
        ]

        for i, line in enumerate(text_lines):
            cv2.putText(debug_frame, line, (x1, y1 - 10 - (i*15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return debug_frame

def run_benchmark():
    # Mocking the pipeline inputs based on your 30 FPS / 5 FPS async setup
    num_frames = 300
    fusion = SceneFusionInterface()
    latencies = []

    csv_file = open("runs/fusion_eval/fusion_output.csv", "w", newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["frame_id", "timestamp", "track_id", "yolo_class", "bbox_x1", "bbox_y1",
                     "bbox_x2", "bbox_y2", "semantic_class", "semantic_confidence", "depth_value",
                     "depth_valid", "road_contact", "context_valid", "segmentation_age"])

    for frame_id in range(num_frames):
        # MOCK: Tracked objects from YOLO (main loop)
        tracked_objects = [{"track_id": 17, "yolo_class": "car", "bbox": [100, 200, 300, 400]}]

        # MOCK: Async SegFormer result (Updates every 6 frames based on your 5.0 effective FPS)
        seg_frame_id = frame_id - (frame_id % 6)
        seg_result = {
            "frame_id": seg_frame_id,
            "valid": True,
            "mask": np.zeros((720, 1280), dtype=np.uint8) # Mock mask
        }

        result = fusion.process(frame_id, frame_id/30.0, None, tracked_objects, seg_result, None)
        latencies.append(result['processing_time_ms'])

        for ctx in result['object_contexts']:
            writer.writerow([
                ctx['frame_id'], ctx['timestamp'], ctx['track_id'], ctx['yolo_class'],
                ctx['bbox'][0], ctx['bbox'][1], ctx['bbox'][2], ctx['bbox'][3],
                ctx['semantic_class'], f"{ctx['semantic_confidence']:.2f}",
                f"{ctx['depth_value']:.2f}", ctx['depth_valid'], ctx['road_contact'],
                ctx['context_valid'], ctx['segmentation_age']
            ])

    csv_file.close()

    metrics = {
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "mean_latency_ms": float(np.mean(latencies)),
        "total_frames_processed": num_frames,
        "stale_handling": "Supported. Age calculated strictly via frame_id delta."
    }

    with open("runs/fusion_eval/fusion_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    run_benchmark()