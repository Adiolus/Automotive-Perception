"""Create contact sheets for baseline handoff and pedestrian evidence."""

import csv
from collections import defaultdict
from pathlib import Path

import cv2


VIDEO = Path("videos/indian_road.mp4")
CSV = Path("runs/tracking_eval/FINAL_TRACKER_V2_nms.csv")
OUTPUT = Path("runs/tracking_eval/FINAL_TRACKER_V2_nms_handoff_contact_sheet.jpg")


def crop_box(frame, box):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return frame[y1:y2, x1:x2]


def add_label(image, label):
    cv2.rectangle(image, (0, 0), (image.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        image,
        label,
        (6, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def main():
    by_frame = defaultdict(list)
    with CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            row["frame"] = int(row["frame"])
            row["track_id"] = int(row["track_id"])
            row["box"] = [float(row[key]) for key in ("x1", "y1", "x2", "y2")]
            by_frame[row["frame"]].append(row)

    # Includes ID ping-pong cases and all pedestrian handoffs in the early scene.
    selections = [
        (45, "rider handoff"),
        (102, "motorcycle handoff"),
        (122, "person ID ping-pong"),
        (149, "person handoff"),
        (193, "rider handoff"),
        (197, "rider ID ping-pong"),
        (200, "motorcycle handoff"),
        (441, "motorcycle ID ping-pong"),
        (614, "motorcycle cluster"),
        (865, "rider cluster"),
    ]

    cap = cv2.VideoCapture(str(VIDEO))
    tiles = []
    for frame_number, description in selections:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        ok, frame = cap.read()
        if not ok:
            continue

        selected = by_frame[frame_number]
        overlay = frame.copy()
        for row in selected:
            color = (0, 220, 0) if row["class"] == "person" else (0, 160, 255)
            x1, y1, x2, y2 = [int(value) for value in row["box"]]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                overlay,
                f'{row["class"]} ID:{row["track_id"]}',
                (x1, max(18, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        overlay = cv2.resize(overlay, (480, 270))
        add_label(overlay, f"frame {frame_number}: {description}")
        tiles.append(overlay)

        for row in selected:
            if row["class"] not in {"person", "rider", "motorcycle"}:
                continue
            crop = crop_box(frame, row["box"])
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (160, 180))
            add_label(crop, f'{row["class"]} ID:{row["track_id"]} f:{frame_number}')
            tiles.append(crop)
    cap.release()

    columns = 4
    tile_width = 480 if any(tile.shape[1] == 480 for tile in tiles) else 160
    tile_height = 270
    normalized = []
    for tile in tiles:
        normalized.append(cv2.resize(tile, (tile_width, tile_height)))
    rows = (len(normalized) + columns - 1) // columns
    sheet = 255 * __import__("numpy").ones(
        (rows * tile_height, columns * tile_width, 3), dtype="uint8"
    )
    for index, tile in enumerate(normalized):
        row = index // columns
        column = index % columns
        sheet[
            row * tile_height : (row + 1) * tile_height,
            column * tile_width : (column + 1) * tile_width,
        ] = tile
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT), sheet)
    print(f"Wrote {OUTPUT} with {len(normalized)} tiles")


if __name__ == "__main__":
    main()