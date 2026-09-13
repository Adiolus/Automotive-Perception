import os

import cv2
import numpy as np
import torch
from PIL import Image

from ultralytics import YOLO
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

# ============================================================
# PATHS
# ============================================================

YOLO_PATH = "runs/detect/runs/detect/idd_train20-3/weights/best.pt"
SEGFORMER_CKPT = "models/segmentation/epoch=9-step=34970.ckpt"
INPUT_VIDEO = "videos/indian_road.mp4"

TRACKER_CONFIG = (
    "venv/lib/python3.14/site-packages/"
    "ultralytics/cfg/trackers/tracktrack.yaml"
)

OUTPUT_DIR = "runs/fusion"
OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "indian_road_fused_clean.mp4",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("AUTOMOTIVE PERCEPTION FUSION")
print("=" * 70)
print("Device:", device)
print("YOLO:", YOLO_PATH)
print("SegFormer:", SEGFORMER_CKPT)
print("Tracker:", TRACKER_CONFIG)
print("Input:", INPUT_VIDEO)
print("Output:", OUTPUT_VIDEO)
print()

# ============================================================
# LOAD YOLO
# ============================================================

print("===== LOADING YOLO =====")

yolo = YOLO(YOLO_PATH)

print("YOLO loaded.")
print("YOLO classes:")

for class_id, name in yolo.names.items():
    print(f"  {class_id}: {name}")

print()

# ============================================================
# LOAD SEGFORMER
# ============================================================

print("===== LOADING SEGFORMER =====")

processor = SegformerImageProcessor.from_pretrained(
    "nvidia/mit-b0"
)

segformer = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b0",
    num_labels=41,
    ignore_mismatched_sizes=True,
)

checkpoint = torch.load(
    SEGFORMER_CKPT,
    map_location="cpu",
)

state_dict = checkpoint.get(
    "state_dict",
    checkpoint,
)

clean_state_dict = {}

for key, value in state_dict.items():
    if key.startswith("model."):
        key = key[len("model."):]

    clean_state_dict[key] = value

model_state = segformer.state_dict()

compatible_state = {}

for key, value in clean_state_dict.items():
    if key in model_state and model_state[key].shape == value.shape:
        compatible_state[key] = value

missing, unexpected = segformer.load_state_dict(
    compatible_state,
    strict=False,
)

print("Loaded checkpoint parameters:", len(compatible_state))
print("Missing parameters:", len(missing))
print("Unexpected parameters:", len(unexpected))

segformer = segformer.to(device)
segformer.eval()

print("SegFormer loaded.")
print()

# ============================================================
# OPEN VIDEO
# ============================================================

print("===== OPENING VIDEO =====")

cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video: {INPUT_VIDEO}"
    )

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

print("Resolution:", width, "x", height)
print("FPS:", fps)
print("Frames:", total_frames)
print()

# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height),
)

if not out.isOpened():
    raise RuntimeError(
        f"Could not create output video: {OUTPUT_VIDEO}"
    )

# ============================================================
# FIXED SEGMENTATION PALETTE
# BGR because OpenCV is used for drawing.
# ============================================================

SEG_COLORS = np.array(
    [
        [128, 64, 128],   # 0 road
        [200, 200, 200],  # 1 parking
        [160, 120, 80],   # 2 drivable fallback
        [244, 35, 232],   # 3 sidewalk
        [70, 70, 70],     # 4 rail track
        [102, 102, 156],  # 5 non-drivable fallback
        [220, 20, 60],    # 6 person
        [119, 11, 32],    # 7 animal
        [255, 0, 0],      # 8 rider
        [0, 0, 255],      # 9 motorcycle
        [119, 0, 119],    # 10 bicycle
        [0, 165, 255],    # 11 autorickshaw
        [0, 255, 0],      # 12 car
        [0, 140, 255],    # 13 truck
        [255, 165, 0],    # 14 bus
        [255, 215, 0],    # 15 caravan
        [255, 140, 0],    # 16 trailer
        [128, 0, 128],    # 17 train
        [0, 128, 128],    # 18 vehicle fallback
        [128, 128, 0],    # 19 curb
        [64, 64, 64],     # 20 wall
        [191, 62, 255],   # 21 fence
        [0, 255, 255],    # 22 guard rail
        [42, 42, 165],    # 23 billboard
        [0, 215, 255],    # 24 traffic sign
        [0, 255, 255],    # 25 traffic light
        [255, 0, 255],    # 26 pole
        [255, 100, 100],  # 27 polegroup
        [180, 180, 180],  # 28 obs-str-bar-fallback
        [70, 120, 180],   # 29 building
        [120, 70, 40],    # 30 bridge
        [80, 80, 120],    # 31 tunnel
        [35, 142, 107],   # 32 vegetation
        [235, 206, 135],   # 33 sky
        [90, 90, 90],     # 34 fallback background
        [0, 0, 0],        # 35 unlabeled
        [50, 50, 50],     # 36 ego vehicle
        [160, 160, 160],  # 37 rectification border
        [20, 20, 20],     # 38 out of roi
        [255, 255, 255],  # 39 license plate
        [40, 40, 40],     # 40 background
    ],
    dtype=np.uint8,
)

# ============================================================
# SEGFORMER INFERENCE
# ============================================================

@torch.no_grad()
def run_segmentation(frame_bgr):
    frame_rgb = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB,
    )

    image = Image.fromarray(frame_rgb)

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    outputs = segformer(**inputs)

    logits = outputs.logits

    logits = torch.nn.functional.interpolate(
        logits,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )

    prediction = torch.argmax(
        logits,
        dim=1,
    )[0]

    return prediction.cpu().numpy()

# ============================================================
# MAIN LOOP
# ============================================================

print("===== RUNNING FUSION =====")
print()

frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    if frame_count % 10 == 0:
        print(
            f"Processing frame "
            f"{frame_count}/{total_frames}"
        )

    # --------------------------------------------------------
    # SEGFORMER
    # --------------------------------------------------------

    seg_mask = run_segmentation(frame)

    seg_color = SEG_COLORS[
        np.clip(seg_mask, 0, len(SEG_COLORS) - 1)
    ]

    # BGR-safe overlay
    fused_frame = cv2.addWeighted(
        frame,
        0.78,
        seg_color,
        0.22,
        0,
    )

    # --------------------------------------------------------
    # YOLO + TRACKTRACK
    # IMPORTANT:
    # Do NOT use SegFormer class IDs to filter YOLO.
    # --------------------------------------------------------

    results = yolo.track(
        frame,
        persist=True,
        tracker=TRACKER_CONFIG,
        conf=0.25,
        verbose=False,
    )

    result = results[0]

    # --------------------------------------------------------
    # DRAW OBJECT TRACKS
    # --------------------------------------------------------

    if result.boxes is not None:

        boxes = result.boxes

        for i in range(len(boxes)):

            xyxy = boxes.xyxy[i].cpu().numpy()

            x1, y1, x2, y2 = map(
                int,
                xyxy,
            )

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            # YOLO class
            yolo_class_id = int(
                boxes.cls[i].item()
            )

            yolo_class_name = yolo.names.get(
                yolo_class_id,
                str(yolo_class_id),
            )

            confidence = float(
                boxes.conf[i].item()
            )

            # Track ID
            track_id = None

            if boxes.id is not None:
                track_id = int(
                    boxes.id[i].item()
                )

            # ------------------------------------------------
            # DRAW BOX
            # ------------------------------------------------

            cv2.rectangle(
                fused_frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2,
            )

            # ------------------------------------------------
            # LABEL
            # ------------------------------------------------

            if track_id is not None:

                label = (
                    f"{yolo_class_name} "
                    f"{confidence:.2f} | "
                    f"ID {track_id}"
                )

            else:

                label = (
                    f"{yolo_class_name} "
                    f"{confidence:.2f}"
                )

            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                2,
            )

            label_y = max(
                text_h + baseline + 4,
                y1,
            )

            cv2.rectangle(
                fused_frame,
                (
                    x1,
                    label_y - text_h - baseline - 4,
                ),
                (
                    x1 + text_w + 6,
                    label_y,
                ),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                fused_frame,
                label,
                (x1 + 3, label_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    # --------------------------------------------------------
    # FRAME NUMBER
    # --------------------------------------------------------

    cv2.putText(
        fused_frame,
        f"Frame: {frame_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # WRITE
    # --------------------------------------------------------

    out.write(fused_frame)

# ============================================================
# CLEANUP
# ============================================================

cap.release()
out.release()

print()
print("=" * 70)
print("FUSION COMPLETE")
print("=" * 70)
print("Frames processed:", frame_count)
print("Output:", OUTPUT_VIDEO)
print()
