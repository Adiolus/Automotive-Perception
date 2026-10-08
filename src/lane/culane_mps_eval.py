from pathlib import Path
import os
import time
import json
import subprocess

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

ROOT = Path.home() / "automotive-perception"
UFLD = ROOT / "external/lane_models/Ultra-Fast-Lane-Detection-v2"
DATA = ROOT / "data/lane_datasets/CULane/extracted"
WEIGHTS = UFLD / "weights/culane_res18.pth"
OUT = ROOT / "runs/ufld_culane_eval"
PRED = OUT / "culane_eval_tmp"
EVAL = UFLD / "evaluation/culane/evaluate"

PYTHON = ROOT / "lane_venv/bin/python"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

MAX_LANES = 4
BATCH = 4
WORKERS = 0

SMOKE = int(os.environ.get("UFLD_EVAL_LIMIT", "0"))

OUT.mkdir(parents=True, exist_ok=True)
PRED.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("UFLD-v2 CULane MPS EVALUATION")
print("=" * 70)
print("Device:", DEVICE)
print("Weights:", WEIGHTS)
print("Data:", DATA)

assert DEVICE.type == "mps"
assert WEIGHTS.exists()
assert (DATA / "list/test.txt").exists()
assert EVAL.exists()

print("PASS: required files exist")

# ------------------------------------------------------------
# UFLD imports
# ------------------------------------------------------------

import sys
sys.path.insert(0, str(UFLD))

from utils.config import Config
from model.model_culane import parsingNet
from data.dataset import LaneTestDataset

cfg = Config.fromfile(
    str(UFLD / "configs/culane_res18.py")
)

cfg.row_anchor = np.linspace(
    0.42, 1.0, cfg.num_row
)
cfg.col_anchor = np.linspace(
    0.0, 1.0, cfg.num_col
)

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

net = parsingNet(
    pretrained=False,
    backbone=cfg.backbone,
    num_grid_row=cfg.num_cell_row,
    num_cls_row=cfg.num_row,
    num_grid_col=cfg.num_cell_col,
    num_cls_col=cfg.num_col,
    num_lane_on_row=cfg.num_lanes,
    num_lane_on_col=cfg.num_lanes,
    use_aux=False,
    input_height=cfg.train_height,
    input_width=cfg.train_width,
    fc_norm=cfg.fc_norm,
)

state = torch.load(
    str(WEIGHTS),
    map_location="cpu"
)

state = state["model"]

clean = {}

for k, v in state.items():
    if k.startswith("module."):
        k = k[7:]
    clean[k] = v

missing, unexpected = net.load_state_dict(
    clean,
    strict=True,
)

assert not missing
assert not unexpected

net = net.to(DEVICE)
net.eval()

print("PASS: UFLD-v2 model loaded")

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

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

dataset = LaneTestDataset(
    str(DATA),
    str(DATA / "list/test.txt"),
    img_transform=transform,
    crop_size=cfg.train_height,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH,
    shuffle=False,
    num_workers=WORKERS,
)

print("CULane test images:", len(dataset))

# ------------------------------------------------------------
# Decoder copied from official UFLD-v2 CULane generation logic
# ------------------------------------------------------------

def generate_lines(
    pred,
    names,
):

    loc_row = pred["loc_row"]
    ext_row = pred["exist_row"]

    loc_col = pred["loc_col"]
    ext_col = pred["exist_col"]

    for b in range(len(names)):

        name = names[b]

        out_path = (
            PRED
            / (
                name[:-3]
                + "lines.txt"
            )
        )

        out_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            out_path,
            "w"
        ) as f:

            # Row-anchor lanes: 1,2
            row_valid = (
                ext_row[b]
                .argmax(0)
                .cpu()
            )

            row_out = (
                loc_row[b]
                .cpu()
            )

            for lane_id in [1, 2]:

                if (
                    row_valid[:, lane_id]
                    .sum()
                    <= cfg.num_row / 2
                ):
                    continue

                for k in range(
                    cfg.num_row
                ):

                    if not row_valid[
                        k,
                        lane_id
                    ]:
                        continue

                    center = int(
                        row_out[
                            :,
                            k,
                            lane_id
                        ].argmax()
                    )

                    lo = max(
                        0,
                        center - 1
                    )

                    hi = min(
                        cfg.num_cell_row - 1,
                        center + 1
                    )

                    ids = torch.arange(
                        lo,
                        hi + 1
                    )

                    vals = (
                        row_out[
                            ids,
                            k,
                            lane_id
                        ]
                        .softmax(0)
                    )

                    coord = (
                        vals
                        * ids.float()
                    ).sum() + 0.5

                    x = (
                        coord
                        / (
                            cfg.num_cell_row
                            - 1
                        )
                        * 1640.0
                    )

                    y = (
                        cfg.row_anchor[k]
                        * 590.0
                    )

                    f.write(
                        f"{float(x):.3f} "
                        f"{float(y):.3f} "
                    )

                f.write("\n")

            # Column-anchor lanes: 0,3
            col_valid = (
                ext_col[b]
                .argmax(0)
                .cpu()
            )

            col_out = (
                loc_col[b]
                .cpu()
            )

            for lane_id in [0, 3]:

                if (
                    col_valid[:, lane_id]
                    .sum()
                    <= cfg.num_col / 4
                ):
                    continue

                for k in range(
                    cfg.num_col
                ):

                    if not col_valid[
                        k,
                        lane_id
                    ]:
                        continue

                    center = int(
                        col_out[
                            :,
                            k,
                            lane_id
                        ].argmax()
                    )

                    lo = max(
                        0,
                        center - 1
                    )

                    hi = min(
                        cfg.num_cell_col - 1,
                        center + 1
                    )

                    ids = torch.arange(
                        lo,
                        hi + 1
                    )

                    vals = (
                        col_out[
                            ids,
                            k,
                            lane_id
                        ]
                        .softmax(0)
                    )

                    coord = (
                        vals
                        * ids.float()
                    ).sum() + 0.5

                    x = (
                        cfg.col_anchor[k]
                        * 1640.0
                    )

                    y = (
                        coord
                        / (
                            cfg.num_cell_col
                            - 1
                        )
                        * 590.0
                    )

                    f.write(
                        f"{float(x):.3f} "
                        f"{float(y):.3f} "
                    )

                f.write("\n")


# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------

start = time.perf_counter()
processed = 0

limit = (
    SMOKE
    if SMOKE > 0
    else len(dataset)
)

for imgs, names in loader:

    if processed >= limit:
        break

    remaining = limit - processed

    if len(imgs) > remaining:
        imgs = imgs[:remaining]
        names = names[:remaining]

    imgs = imgs.to(DEVICE)

    with torch.inference_mode():
        pred = net(imgs)

    generate_lines(
        pred,
        names,
    )

    processed += len(imgs)

    if processed % 100 == 0 or processed == limit:

        if DEVICE.type == "mps":
            torch.mps.synchronize()

        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            f"{processed}/{limit} "
            f"FPS={processed / max(elapsed, 1e-6):.2f}"
        )

if DEVICE.type == "mps":
    torch.mps.synchronize()

elapsed = (
    time.perf_counter()
    - start
)

print()
print(
    "Inference complete:",
    processed,
    "images"
)
print(
    "Inference FPS:",
    f"{processed / max(elapsed,1e-6):.2f}"
)

# ------------------------------------------------------------
# Official CULane evaluator
# ------------------------------------------------------------

if processed == len(dataset):

    test_split = DATA / "list/test_split"

    categories = [
        "test0_normal.txt",
        "test1_crowd.txt",
        "test2_hlight.txt",
        "test3_shadow.txt",
        "test4_noline.txt",
        "test5_arrow.txt",
        "test6_curve.txt",
        "test7_cross.txt",
        "test8_night.txt",
    ]

    result_dir = OUT / "txt"
    result_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results = {}

    for category in categories:

        list_file = (
            test_split
            / category
        )

        if not list_file.exists():
            print(
                "WARNING missing:",
                list_file
            )
            continue

        stem = category.replace(
            ".txt",
            ""
        )

        output_file = (
            result_dir
            / f"{stem}.txt"
        )

        cmd = [
            str(EVAL),
            "-a", str(DATA) + "/",
            "-d", str(PRED) + "/",
            "-i", str(DATA) + "/",
            "-l", str(list_file),
            "-w", "30",
            "-t", "0.5",
            "-c", "1640",
            "-r", "590",
            "-f", "1",
            "-o", str(output_file),
        ]

        print()
        print("Evaluating:", stem)

        subprocess.run(
            cmd,
            check=True,
        )

        text = output_file.read_text(
            errors="ignore"
        ).split()

        values = {}

        for i in range(
            0,
            len(text) - 1,
            2
        ):

            key = (
                text[i]
                .rstrip(":")
            )

            value = text[i + 1]

            try:
                values[key] = float(
                    value
                )
            except ValueError:
                values[key] = value

        results[stem] = values

    (OUT / "categories.json").write_text(
        json.dumps(
            results,
            indent=2
        )
    )

    print()
    print("=" * 70)
    print("FINAL CULANE RESULTS")
    print("=" * 70)

    for category, values in results.items():

        print(
            category,
            values
        )

    print()
    print(
        "JSON:",
        OUT / "categories.json"
    )

else:

    print()
    print(
        "SMOKE TEST COMPLETE"
    )
    print(
        "No official evaluation yet."
    )

print()
print("PASS: MPS inference pipeline completed")
