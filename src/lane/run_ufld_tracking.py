from pathlib import Path
import time

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

import sys

ROOT = Path(
    "/Users/adityachinchore/automotive-perception"
)

UFLD_ROOT = (
    ROOT
    / "external/lane_models/Ultra-Fast-Lane-Detection-v2"
)

sys.path.insert(
    0,
    str(UFLD_ROOT)
)

from model.model_culane import parsingNet
from utils.config import Config

sys.path.insert(
    0,
    str(ROOT)
)

from src.lane.lane_tracker import LaneTracker


VIDEO = ROOT / "videos/highway_test.mp4"

WEIGHTS = (
    UFLD_ROOT
    / "weights/culane_res18.pth"
)

OUT_DIR = (
    ROOT
    / "runs/ufld_tracker_clean"
)

OUT_VIDEO = (
    OUT_DIR
    / "highway_ufld_tracked.mp4"
)

OUT_CSV = (
    OUT_DIR
    / "lane_tracks.csv"
)

MAX_FRAMES = 600

DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("=" * 72)
print("UFLD-V2 + CLEAN LANE TRACKER")
print("=" * 72)
print("Device:", DEVICE)
print("Video:", VIDEO)
print("Weights:", WEIGHTS)


if not VIDEO.exists():
    raise FileNotFoundError(
        VIDEO
    )

if not WEIGHTS.exists():
    raise FileNotFoundError(
        WEIGHTS
    )


cfg = Config.fromfile(
    str(
        UFLD_ROOT
        / "configs/culane_res18.py"
    )
)

cfg.row_anchor = np.linspace(
    0.42,
    1.0,
    cfg.num_row,
)

cfg.col_anchor = np.linspace(
    0.0,
    1.0,
    cfg.num_col,
)


net = parsingNet(
    pretrained=False,
    backbone=cfg.backbone,
    num_grid_row=cfg.num_cell_row,
    num_cls_row=cfg.num_row,
    num_grid_col=cfg.num_cell_col,
    num_cls_col=cfg.num_col,
    num_lane_on_row=cfg.num_lanes,
    num_lane_on_col=cfg.num_lanes,
    use_aux=cfg.use_aux,
    input_height=cfg.train_height,
    input_width=cfg.train_width,
    fc_norm=cfg.fc_norm,
)


checkpoint = torch.load(
    str(WEIGHTS),
    map_location="cpu",
)

state = checkpoint["model"]

clean_state = {}

for key, value in state.items():

    if key.startswith(
        "module."
    ):
        key = key[7:]

    clean_state[key] = value


missing, unexpected = (
    net.load_state_dict(
        clean_state,
        strict=False,
    )
)

if missing or unexpected:
    raise RuntimeError(
        f"Checkpoint mismatch: "
        f"missing={len(missing)}, "
        f"unexpected={len(unexpected)}"
    )


net = net.to(
    DEVICE
)

net.eval()


transform = transforms.Compose([
    transforms.Resize(
        (
            int(
                cfg.train_height
                / cfg.crop_ratio
            ),
            cfg.train_width,
        )
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    ),
])


def decode_lanes(
    pred,
    width,
    height,
):
    num_grid_row = (
        pred["loc_row"].shape[1]
    )

    num_cls_row = (
        pred["loc_row"].shape[2]
    )

    num_grid_col = (
        pred["loc_col"].shape[1]
    )

    num_cls_col = (
        pred["loc_col"].shape[2]
    )

    max_row = (
        pred["loc_row"]
        .argmax(1)
        .cpu()
    )

    valid_row = (
        pred["exist_row"]
        .argmax(1)
        .cpu()
    )

    max_col = (
        pred["loc_col"]
        .argmax(1)
        .cpu()
    )

    valid_col = (
        pred["exist_col"]
        .argmax(1)
        .cpu()
    )

    loc_row = (
        pred["loc_row"]
        .cpu()
    )

    loc_col = (
        pred["loc_col"]
        .cpu()
    )

    lanes = []

    for lane_idx in [1, 2]:

        lane = []

        if (
            valid_row[
                0,
                :,
                lane_idx
            ].sum()
            > num_cls_row / 2
        ):

            for k in range(
                valid_row.shape[1]
            ):

                if not valid_row[
                    0,
                    k,
                    lane_idx
                ]:
                    continue

                center = int(
                    max_row[
                        0,
                        k,
                        lane_idx
                    ]
                )

                lo = max(
                    0,
                    center - 1
                )

                hi = min(
                    num_grid_row - 1,
                    center + 1
                )

                idx = torch.arange(
                    lo,
                    hi + 1
                )

                value = (
                    loc_row[
                        0,
                        idx,
                        k,
                        lane_idx
                    ]
                    .softmax(0)
                    * idx.float()
                ).sum() + 0.5

                x = (
                    value
                    / max(
                        num_grid_row - 1,
                        1
                    )
                    * width
                )

                y = (
                    cfg.row_anchor[k]
                    * height
                )

                if (
                    np.isfinite(
                        float(x)
                    )
                    and
                    np.isfinite(
                        float(y)
                    )
                ):

                    lane.append(
                        (
                            float(x),
                            float(y)
                        )
                    )

        lanes.append(
            lane
        )

    for lane_idx in [0, 3]:

        lane = []

        if (
            valid_col[
                0,
                :,
                lane_idx
            ].sum()
            > num_cls_col / 4
        ):

            for k in range(
                valid_col.shape[1]
            ):

                if not valid_col[
                    0,
                    k,
                    lane_idx
                ]:
                    continue

                center = int(
                    max_col[
                        0,
                        k,
                        lane_idx
                    ]
                )

                lo = max(
                    0,
                    center - 1
                )

                hi = min(
                    num_grid_col - 1,
                    center + 1
                )

                idx = torch.arange(
                    lo,
                    hi + 1
                )

                value = (
                    loc_col[
                        0,
                        idx,
                        k,
                        lane_idx
                    ]
                    .softmax(0)
                    * idx.float()
                ).sum() + 0.5

                x = (
                    cfg.col_anchor[k]
                    * width
                )

                y = (
                    value
                    / max(
                        num_grid_col - 1,
                        1
                    )
                    * height
                )

                if (
                    np.isfinite(
                        float(x)
                    )
                    and
                    np.isfinite(
                        float(y)
                    )
                ):

                    lane.append(
                        (
                            float(x),
                            float(y)
                        )
                    )

        lanes.append(
            lane
        )

    return lanes


cap = cv2.VideoCapture(
    str(VIDEO)
)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open highway video."
    )


frame_width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

frame_height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

video_fps = (
    cap.get(
        cv2.CAP_PROP_FPS
    )
    or 30.0
)


writer = cv2.VideoWriter(
    str(OUT_VIDEO),
    cv2.VideoWriter_fourcc(
        *"mp4v"
    ),
    video_fps,
    (
        frame_width,
        frame_height
    ),
)

if not writer.isOpened():
    raise RuntimeError(
        "Could not create output video."
    )


tracker = LaneTracker(
    width=frame_width,
    height=frame_height,
    max_lanes=4,
    max_missed=8,
    match_threshold=0.14,
    ema_alpha=0.25,
)


csv_rows = [
    "frame,lane_id,age,hits,missed,bottom_x,confidence"
]

processed = 0
frames_with_tracks = 0
total_active = 0
max_active = 0

start = time.perf_counter()


while processed < MAX_FRAMES:

    ok, frame = cap.read()

    if not ok:
        break

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    image = Image.fromarray(
        rgb
    )

    tensor = transform(
        image
    )

    tensor = tensor[
        :,
        -cfg.train_height:,
        :
    ]

    tensor = tensor.unsqueeze(
        0
    ).to(
        DEVICE
    )

    with torch.no_grad():
        pred = net(tensor)

    raw_lanes = decode_lanes(
        pred,
        frame_width,
        frame_height,
    )

    tracks = tracker.update(
        raw_lanes
    )

    assert len(tracks) <= 4

    ids = [
        t.lane_id
        for t in tracks
    ]

    assert len(ids) == len(
        set(ids)
    )

    assert all(
        1 <= lane_id <= 4
        for lane_id in ids
    )

    for track in tracks:
        assert np.isfinite(
            track.curve
        ).all()

    if tracks:
        frames_with_tracks += 1

    total_active += len(
        tracks
    )

    max_active = max(
        max_active,
        len(tracks)
    )

    vis = frame.copy()

    for track in tracks:

        points = np.column_stack(
            [
                track.curve
                * frame_width,
                tracker.y_norm
                * frame_height,
            ]
        )

        points = np.round(
            points
        ).astype(
            np.int32
        )

        points[:, 0] = np.clip(
            points[:, 0],
            0,
            frame_width - 1,
        )

        points[:, 1] = np.clip(
            points[:, 1],
            0,
            frame_height - 1,
        )

        color = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 200, 255),
            (255, 0, 255),
        ][
            (track.lane_id - 1) % 4
        ]

        cv2.polylines(
            vis,
            [
                points.reshape(
                    -1,
                    1,
                    2
                )
            ],
            False,
            color,
            5,
            cv2.LINE_AA,
        )

        bottom_x = int(
            np.clip(
                track.curve[-1]
                * frame_width,
                0,
                frame_width - 1,
            )
        )

        label = (
            f"Lane {track.lane_id}"
        )

        if track.missed > 0:
            label += (
                f" | predicted"
            )

        cv2.putText(
            vis,
            label,
            (
                bottom_x,
                frame_height - 35,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color,
            2,
            cv2.LINE_AA,
        )

        csv_rows.append(
            f"{processed},"
            f"{track.lane_id},"
            f"{track.age},"
            f"{track.hits},"
            f"{track.missed},"
            f"{bottom_x},"
            f"{track.confidence:.4f}"
        )

    cv2.putText(
        vis,
        (
            f"UFLD-v2 + Clean LaneTracker"
            f" | lanes={len(tracks)}"
        ),
        (
            20,
            40
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    writer.write(
        vis
    )

    processed += 1

    if processed % 50 == 0:

        if DEVICE.type == "mps":
            torch.mps.synchronize()

        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            f"Frame {processed:4d} | "
            f"active={len(tracks)} | "
            f"new={tracker.total_new_tracks} | "
            f"FPS={processed / max(elapsed, 1e-6):.2f}"
        )


if DEVICE.type == "mps":
    torch.mps.synchronize()

elapsed = (
    time.perf_counter()
    - start
)

cap.release()
writer.release()

OUT_CSV.write_text(
    "\n".join(csv_rows)
)

print()
print("=" * 72)
print("CLEAN LANE TRACKER COMPLETE")
print("=" * 72)
print("Frames:", processed)
print(
    "Frames with tracks:",
    frames_with_tracks
)
print(
    "Mean active lanes/frame:",
    f"{total_active / max(processed, 1):.2f}"
)
print(
    "Maximum active lanes:",
    max_active
)
print(
    "Total new track slots:",
    tracker.total_new_tracks
)
print(
    "Successful associations:",
    tracker.total_matches
)
print(
    "End-to-end FPS:",
    f"{processed / max(elapsed, 1e-6):.2f}"
)
print("Video:", OUT_VIDEO)
print("CSV:", OUT_CSV)

assert max_active <= 4
print("PASS: active lane count never exceeded 4")

print("PASS: no NaN/Inf lane curves")
print("PASS: no duplicate lane IDs")
print("PASS: lane IDs restricted to 1..4")
