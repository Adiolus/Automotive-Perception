import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Official OC-SORT
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "external" / "OC_SORT"))
from trackers.ocsort_tracker.ocsort import OCSort


# =========================
# CONFIG
# =========================
MODEL_PATH = "runs/detect/runs/detect/idd_train20-3/weights/best.pt"
VIDEO_PATH = "videos/indian_road.mp4"
OUTPUT_PATH = "runs/tracking_eval/yolo_ocsort_test.mp4"

IMG_SIZE = 960
CONF = 0.35
MAX_FRAMES = 450


# =========================
# SETUP
# =========================
Path("runs/tracking_eval").mkdir(parents=True, exist_ok=True)

print("Loading YOLO...")
model = YOLO(MODEL_PATH)

print("Initializing OC-SORT...")
tracker = OCSort(
    det_thresh=CONF,
    max_age=30,
    min_hits=1,
    iou_threshold=0.3,
    delta_t=3,
    inertia=0.2,
)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

print(f"Video: {width}x{height} @ {fps:.2f} FPS")
print(f"Processing first {MAX_FRAMES} frames...")
print("YOLO -> OC-SORT")


# =========================
# PROCESS
# =========================
frame_count = 0
total_tracks = 0

while frame_count < MAX_FRAMES:

    ret, frame = cap.read()

    if not ret:
        break

    # YOLO detection
    result = model.predict(
        frame,
        imgsz=IMG_SIZE,
        conf=CONF,
        device="mps",
        verbose=False,
    )[0]

    boxes = result.boxes

    if boxes is not None and len(boxes) > 0:

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()

        detections = np.column_stack([
            xyxy,
            confs
        ]).astype(np.float32)

    else:
        detections = np.empty((0, 5), dtype=np.float32)

    # OC-SORT
    tracks = tracker.update(
        detections,
        img_info=(height, width),
        img_size=(height, width),
    )

    # Draw tracks
    for track in tracks:

        x1, y1, x2, y2, track_id = track

        x1 = int(max(0, min(width - 1, x1)))
        y1 = int(max(0, min(height - 1, y1)))
        x2 = int(max(0, min(width - 1, x2)))
        y2 = int(max(0, min(height - 1, y2)))

        track_id = int(track_id)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        total_tracks += 1

    writer.write(frame)

    frame_count += 1

    if frame_count % 50 == 0:
        print(f"Processed {frame_count}/{MAX_FRAMES} frames")


# =========================
# CLEANUP
# =========================
cap.release()
writer.release()

print()
print("DONE")
print(f"Frames processed: {frame_count}")
print(f"Output: {OUTPUT_PATH}")
print(f"Track outputs: {total_tracks}")
