from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from scipy.optimize import linear_sum_assignment

MODEL_PATH = "runs/detect/runs/detect/idd_train20-3/weights/best.pt"
VIDEO_PATH = "videos/indian_road.mp4"
OUTPUT_PATH = "runs/tracking_eval/hungarian_15s.mp4"

MAX_FRAMES = 450

# Association limits
MIN_IOU = 0.05
MAX_CENTER_DIST = 150.0
MAX_MISSED = 30

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

Path("runs/tracking_eval").mkdir(parents=True, exist_ok=True)

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

tracks = []
next_id = 1


def get_center(box):
    return np.array([
        (box[0] + box[2]) / 2,
        (box[1] + box[3]) / 2
    ], dtype=np.float32)


def get_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2-x1) * max(0, y2-y1)

    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])

    union = area_a + area_b - inter

    return inter / union if union > 0 else 0


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
                "conf": float(conf),
                "center": get_center(box)
            })

    # ---------------------------------------------------------
    # GLOBAL ONE-TO-ONE MATCHING
    # ---------------------------------------------------------

    if tracks and detections:

        cost = np.full(
            (len(tracks), len(detections)),
            1000.0,
            dtype=np.float32
        )

        for ti, track in enumerate(tracks):

            for di, det in enumerate(detections):

                # Never match different classes
                if track["cls"] != det["cls"]:
                    continue

                overlap = get_iou(
                    track["bbox"],
                    det["bbox"]
                )

                distance = np.linalg.norm(
                    track["center"] - det["center"]
                )

                # Hard spatial gate
                if (
                    overlap < MIN_IOU
                    and distance > MAX_CENTER_DIST
                ):
                    continue

                # Lower cost = better match
                #
                # IoU is dominant.
                # Center distance breaks ties between nearby
                # same-class objects.
                cost[ti, di] = (
                    0.75 * (1.0 - overlap)
                    + 0.25 * min(
                        distance / MAX_CENTER_DIST,
                        1.0
                    )
                )

        rows, cols = linear_sum_assignment(cost)

        matched_tracks = set()
        matched_detections = set()

        for ti, di in zip(rows, cols):

            if cost[ti, di] >= 1000:
                continue

            track = tracks[ti]
            det = detections[di]

            matched_tracks.add(ti)
            matched_detections.add(di)

            track["bbox"] = det["bbox"].copy()
            track["center"] = det["center"].copy()
            track["conf"] = det["conf"]
            track["missed"] = 0
            track["age"] += 1

        # Unmatched tracks
        for ti, track in enumerate(tracks):

            if ti not in matched_tracks:
                track["missed"] += 1

    else:

        matched_detections = set()

    # ---------------------------------------------------------
    # CREATE TRACKS FOR UNMATCHED DETECTIONS
    # ---------------------------------------------------------

    for di, det in enumerate(detections):

        if di in matched_detections:
            continue

        tracks.append({
            "id": next_id,
            "cls": det["cls"],
            "bbox": det["bbox"].copy(),
            "center": det["center"].copy(),
            "conf": det["conf"],
            "missed": 0,
            "age": 1
        })

        next_id += 1

    # ---------------------------------------------------------
    # REMOVE DEAD TRACKS
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
        f"Frame {frame_idx}/{MAX_FRAMES} | "
        f"Active tracks {len(tracks)}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    out.write(frame)

    if frame_idx % 50 == 0:
        print(
            f"Processed {frame_idx}/{MAX_FRAMES} | "
            f"detections={len(detections)} | "
            f"tracks={len(tracks)} | "
            f"next_id={next_id}"
        )

cap.release()
out.release()

print("\n========== DONE ==========")
print(f"Frames: {frame_idx}")
print(f"Total IDs created: {next_id - 1}")
print(f"Output: {OUTPUT_PATH}")
