import cv2
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "runs/detect/runs/detect/idd_train20-3/weights/best.pt"
VIDEO_PATH = "videos/indian_road.mp4"
OUTPUT_PATH = "runs/tracking_eval/bytetrack_15s.mp4"

IMG_SIZE = 960
CONF = 0.35
MAX_FRAMES = 450


Path("runs/tracking_eval").mkdir(parents=True, exist_ok=True)

print("Loading YOLO...")
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open {VIDEO_PATH}")

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
print("Tracker: Ultralytics native OC-SORT")
print(f"Processing {MAX_FRAMES} frames = {MAX_FRAMES / fps:.1f} seconds")
print()

frame_count = 0
track_outputs = 0

while frame_count < MAX_FRAMES:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=IMG_SIZE,
        conf=CONF,
        device="mps",
        verbose=False,
    )

    result = results[0]

    if result.boxes is not None and len(result.boxes) > 0:

        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        if boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = [-1] * len(xyxy)

        for box, conf, cls, track_id in zip(
            xyxy, confs, classes, track_ids
        ):

            x1, y1, x2, y2 = box.astype(int)

            name = model.names[int(cls)]

            if track_id != -1:
                label = f"{name} | ID {track_id} | {conf:.2f}"
                track_outputs += 1
            else:
                label = f"{name} | {conf:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    writer.write(frame)

    frame_count += 1

    if frame_count % 50 == 0:
        print(f"Processed {frame_count}/{MAX_FRAMES} frames")


cap.release()
writer.release()

print()
print("DONE")
print(f"Frames processed: {frame_count}")
print(f"Duration: {frame_count / fps:.2f} seconds")
print(f"Track outputs: {track_outputs}")
print(f"Output: {OUTPUT_PATH}")
