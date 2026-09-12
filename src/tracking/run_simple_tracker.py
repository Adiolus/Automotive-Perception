from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

MODEL_PATH = "runs/detect/runs/detect/idd_train20-3/weights/best.pt"
VIDEO_PATH = "videos/indian_road.mp4"
OUTPUT_PATH = "runs/tracking_eval/simple_iou_tracker_15s.mp4"

MAX_FRAMES = 450

# Matching thresholds
IOU_THRESHOLD = 0.30
MAX_MISSED = 60

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

Path("runs/tracking_eval").mkdir(parents=True, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

# Each track:
# {
#   id,
#   cls,
#   conf,
#   bbox,
#   velocity,
#   missed
# }

tracks = []
next_id = 1


def iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)

    intersection = iw * ih

    area_a = max(0.0, box_a[2] - box_a[0]) * max(
        0.0, box_a[3] - box_a[1]
    )

    area_b = max(0.0, box_b[2] - box_b[0]) * max(
        0.0, box_b[3] - box_b[1]
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def center(box):
    return np.array([
        (box[0] + box[2]) / 2.0,
        (box[1] + box[3]) / 2.0
    ], dtype=np.float32)


frame_idx = 0

while frame_idx < MAX_FRAMES:

    ret, frame = cap.read()

    if not ret:
        break

    frame_idx += 1

    result = model.predict(
        frame,
        imgsz=960,
        conf=0.35,
        device="mps",
        verbose=False
    )[0]

    detections = []

    if result.boxes is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confs):

            detections.append({
                "bbox": box.astype(np.float32),
                "cls": int(cls),
                "conf": float(conf)
            })

    # ---------------------------------------------------------
    # PREDICT CURRENT POSITION OF EXISTING TRACKS
    # ---------------------------------------------------------

    predicted = []

    for track in tracks:

        pbox = track["bbox"].copy()

        dx, dy = track["velocity"]

        pbox[[0, 2]] += dx
        pbox[[1, 3]] += dy

        predicted.append(pbox)

    # ---------------------------------------------------------
    # GREEDY CLASS-AWARE IOU MATCHING
    # ---------------------------------------------------------

    candidate_matches = []

    for ti, track in enumerate(tracks):

        for di, det in enumerate(detections):

            # NEVER match different classes
            if track["cls"] != det["cls"]:
                continue

            score = iou(predicted[ti], det["bbox"])

            if score >= IOU_THRESHOLD:
                candidate_matches.append(
                    (score, ti, di)
                )

    # Highest IoU matches first
    candidate_matches.sort(reverse=True)

    matched_tracks = set()
    matched_detections = set()

    for score, ti, di in candidate_matches:

        if ti in matched_tracks:
            continue

        if di in matched_detections:
            continue

        matched_tracks.add(ti)
        matched_detections.add(di)

        track = tracks[ti]
        det = detections[di]

        old_center = center(track["bbox"])
        new_center = center(det["bbox"])

        # Smooth velocity estimate
        new_velocity = (
            0.7 * track["velocity"]
            + 0.3 * (new_center - old_center)
        )

        track["velocity"] = new_velocity
        track["bbox"] = det["bbox"]
        track["conf"] = det["conf"]
        track["missed"] = 0

    # ---------------------------------------------------------
    # UNMATCHED TRACKS
    # ---------------------------------------------------------

    for ti, track in enumerate(tracks):

        if ti not in matched_tracks:

            track["bbox"] = predicted[ti]
            track["missed"] += 1

    # ---------------------------------------------------------
    # CREATE NEW TRACKS
    # ---------------------------------------------------------

    for di, det in enumerate(detections):

        if di in matched_detections:
            continue

        tracks.append({
            "id": next_id,
            "cls": det["cls"],
            "conf": det["conf"],
            "bbox": det["bbox"].copy(),
            "velocity": np.array([0.0, 0.0], dtype=np.float32),
            "missed": 0
        })

        next_id += 1

    # ---------------------------------------------------------
    # DELETE OLD TRACKS
    # ---------------------------------------------------------

    tracks = [
        t for t in tracks
        if t["missed"] <= MAX_MISSED
    ]

    # ---------------------------------------------------------
    # DRAW
    # ---------------------------------------------------------

    for track in tracks:

        if track["missed"] > 0:
            continue

        x1, y1, x2, y2 = track["bbox"].astype(int)

        cls_name = model.names.get(
            track["cls"],
            str(track["cls"])
        )

        label = (
            f"{cls_name} | ID {track['id']} | "
            f"{track['conf']:.2f}"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"Frame: {frame_idx}/{MAX_FRAMES}  Tracks: {len(tracks)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    out.write(frame)

    if frame_idx % 50 == 0:
        print(
            f"Processed {frame_idx}/{MAX_FRAMES} frames | "
            f"active tracks: {len(tracks)}"
        )

cap.release()
out.release()

print("\n========== DONE ==========")
print(f"Frames processed: {frame_idx}")
print(f"Next track ID: {next_id}")
print(f"Output: {OUTPUT_PATH}")
