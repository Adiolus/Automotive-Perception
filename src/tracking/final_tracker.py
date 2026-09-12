from ultralytics import YOLO
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import csv
import time


# ============================================================
# FINAL PERSISTENT ID TRACKER
#
# Designed specifically for this project:
#
# YOLO detection
#       |
#       +-- class gate
#       |
#       +-- motion prediction
#       |
#       +-- IoU
#       |
#       +-- center distance
#       |
#       +-- appearance memory
#       |
#       +-- global Hungarian assignment
#       |
#       +-- lost-track re-identification
#       |
#       +-- persistent ID
#
# NO OC-SORT
# NO ByteTrack
# NO Deep OC-SORT
# NO neural ReID
#
# ============================================================


MODEL_PATH = "runs/detect/runs/detect/idd_train20-3/weights/best.pt"
VIDEO_PATH = "videos/indian_road.mp4"

OUTPUT_DIR = Path("runs/tracking_eval")

VIDEO_OUTPUT = OUTPUT_DIR / "FINAL_PERSISTENT_TRACKER.mp4"
CSV_OUTPUT = OUTPUT_DIR / "FINAL_PERSISTENT_TRACKER.csv"


# ============================================================
# DETECTOR
# ============================================================

CONF_THRESHOLD = 0.35
IMAGE_SIZE = 960


# ============================================================
# TRACK LIFETIME
# ============================================================

# 30 FPS -> 4 seconds of memory.
#
# A person temporarily hidden behind traffic therefore
# does NOT automatically get a new ID.
MAX_MISSED = 120


# ============================================================
# TRACK CONFIRMATION
# ============================================================

# New tracks need several observations before becoming
# fully confirmed identities.
#
# This prevents one-frame false detections from polluting
# the ID system.
MIN_HITS = 2


# ============================================================
# ASSOCIATION PARAMETERS
# ============================================================

# IoU gate for strong spatial matching.
STRONG_IOU = 0.30

# General minimum IoU.
MIN_IOU = 0.01

# Maximum normalized center displacement.
MAX_CENTER_RATIO = 3.5


# ============================================================
# APPEARANCE
# ============================================================

# Number of historical appearance descriptors retained.
APPEARANCE_MEMORY = 8

# Appearance similarity threshold for re-identification.
REID_THRESHOLD = 0.58


# ============================================================
# CLASS-SPECIFIC ASSOCIATION WEIGHTS
# ============================================================

# Person tracking gets stronger appearance weighting because
# multiple nearby people can have almost identical motion.
#
# Vehicles rely more heavily on geometry because their visual
# appearance changes substantially with scale/viewpoint.

PERSON_CLASSES = {
    "person",
    "rider",
}

VEHICLE_CLASSES = {
    "car",
    "truck",
    "bus",
    "autorickshaw",
    "motorcycle",
    "bicycle",
    "vehicle_fallback",
}


# ============================================================
# TRACK OBJECT
# ============================================================

class Track:

    def __init__(
        self,
        track_id,
        detection,
        frame,
        frame_idx,
    ):

        self.id = track_id

        self.cls = detection["cls"]

        self.bbox = detection["bbox"].copy()

        self.center = detection["center"].copy()

        self.velocity = np.zeros(
            2,
            dtype=np.float32,
        )

        self.conf = detection["conf"]

        self.age = 1

        self.hits = 1

        self.missed = 0

        self.last_seen = frame_idx

        self.confirmed = False

        self.visible = True

        # Appearance memory.
        self.appearances = [
            detection["appearance"].copy()
        ]

        # Position history.
        self.history = [
            self.center.copy()
        ]

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    def predicted_center(self):

        # Limit velocity influence.
        velocity = np.clip(
            self.velocity,
            -100.0,
            100.0,
        )

        return (
            self.center
            + velocity
        )

    def predicted_box(self):

        dx, dy = self.velocity

        box = self.bbox.copy()

        box[[0, 2]] += dx
        box[[1, 3]] += dy

        return box

    # --------------------------------------------------------
    # APPEARANCE SIMILARITY
    # --------------------------------------------------------

    def appearance_similarity(
        self,
        descriptor,
    ):

        if not self.appearances:
            return 0.0

        similarities = []

        for old in self.appearances:

            a = old
            b = descriptor

            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)

            if na < 1e-6 or nb < 1e-6:
                continue

            similarities.append(
                float(
                    np.dot(a, b)
                    / (na * nb)
                )
            )

        if not similarities:
            return 0.0

        # MAX is deliberate.
        #
        # We don't want the identity to disappear just because
        # the latest frame has changed due to blur/occlusion.
        return float(
            np.clip(
                max(similarities),
                0.0,
                1.0,
            )
        )

    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    def update(
        self,
        detection,
        frame_idx,
    ):

        old_center = self.center.copy()

        new_center = detection["center"].copy()

        measured_velocity = (
            new_center
            - old_center
        )

        # Smooth velocity.
        self.velocity = (
            0.65 * self.velocity
            + 0.35 * measured_velocity
        )

        self.bbox = detection["bbox"].copy()

        self.center = new_center.copy()

        self.conf = detection["conf"]

        self.hits += 1

        self.age += 1

        self.missed = 0

        self.last_seen = frame_idx

        self.visible = True

        if self.hits >= MIN_HITS:
            self.confirmed = True

        # ----------------------------------------------------
        # APPEARANCE MEMORY
        # ----------------------------------------------------

        appearance = detection["appearance"]

        self.appearances.append(
            appearance.copy()
        )

        if len(self.appearances) > APPEARANCE_MEMORY:
            self.appearances.pop(0)

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        self.history.append(
            self.center.copy()
        )

        if len(self.history) > 60:
            self.history.pop(0)

    # --------------------------------------------------------
    # MISS
    # --------------------------------------------------------

    def miss(self):

        self.missed += 1

        self.age += 1

        self.visible = False

        # Keep predicted trajectory.
        predicted = self.predicted_box()

        self.bbox = predicted

        self.center = box_center(
            predicted
        )

        self.history.append(
            self.center.copy()
        )

        if len(self.history) > 60:
            self.history.pop(0)


# ============================================================
# BASIC GEOMETRY
# ============================================================

def box_center(box):

    return np.array(
        [
            (box[0] + box[2]) * 0.5,
            (box[1] + box[3]) * 0.5,
        ],
        dtype=np.float32,
    )


def box_size(box):

    return np.array(
        [
            max(
                1.0,
                box[2] - box[0],
            ),
            max(
                1.0,
                box[3] - box[1],
            ),
        ],
        dtype=np.float32,
    )


def box_iou(a, b):

    x1 = max(
        float(a[0]),
        float(b[0]),
    )

    y1 = max(
        float(a[1]),
        float(b[1]),
    )

    x2 = min(
        float(a[2]),
        float(b[2]),
    )

    y2 = min(
        float(a[3]),
        float(b[3]),
    )

    w = max(
        0.0,
        x2 - x1,
    )

    h = max(
        0.0,
        y2 - y1,
    )

    intersection = w * h

    area_a = max(
        0.0,
        float(a[2] - a[0]),
    ) * max(
        0.0,
        float(a[3] - a[1]),
    )

    area_b = max(
        0.0,
        float(b[2] - b[0]),
    ) * max(
        0.0,
        float(b[3] - b[1]),
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# APPEARANCE DESCRIPTOR
# ============================================================

def appearance_descriptor(
    frame,
    box,
):

    h, w = frame.shape[:2]

    x1, y1, x2, y2 = (
        box.astype(int)
    )

    x1 = max(
        0,
        min(w - 1, x1),
    )

    y1 = max(
        0,
        min(h - 1, y1),
    )

    x2 = max(
        x1 + 1,
        min(w, x2),
    )

    y2 = max(
        y1 + 1,
        min(h, y2),
    )

    crop = frame[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:

        return np.zeros(
            320,
            dtype=np.float32,
        )

    ch, cw = crop.shape[:2]

    # Crop a small border to reduce background influence.
    if ch > 12 and cw > 12:

        mx = int(cw * 0.08)
        my = int(ch * 0.06)

        crop = crop[
            my:ch-my,
            mx:cw-mx,
        ]

    # --------------------------------------------------------
    # HSV COLOR HISTOGRAM
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV,
    )

    hist = cv2.calcHist(
        [hsv],
        [0, 1],
        None,
        [16, 16],
        [0, 180, 0, 256],
    )

    hist = cv2.normalize(
        hist,
        hist,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_L2,
    ).flatten()

    # --------------------------------------------------------
    # COLOR THUMBNAIL
    # --------------------------------------------------------

    thumb = cv2.resize(
        crop,
        (8, 12),
        interpolation=cv2.INTER_AREA,
    )

    thumb = cv2.cvtColor(
        thumb,
        cv2.COLOR_BGR2HSV,
    )

    thumb = thumb.astype(
        np.float32
    )

    thumb[:, :, 0] /= 180.0
    thumb[:, :, 1] /= 255.0
    thumb[:, :, 2] /= 255.0

    thumb = thumb.flatten()

    # --------------------------------------------------------
    # GRAYSCALE SHAPE / TEXTURE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.resize(
        gray,
        (8, 8),
        interpolation=cv2.INTER_AREA,
    )

    gray = (
        gray.astype(
            np.float32
        )
        / 255.0
    )

    gray = gray - gray.mean()

    norm = np.linalg.norm(
        gray
    )

    if norm > 1e-6:
        gray /= norm

    descriptor = np.concatenate(
        [
            hist.astype(
                np.float32
            ),
            thumb,
            gray.flatten(),
        ]
    )

    norm = np.linalg.norm(
        descriptor
    )

    if norm > 1e-6:
        descriptor /= norm

    return descriptor.astype(
        np.float32
    )


# ============================================================
# FINAL TRACKER
# ============================================================

class PersistentTracker:

    def __init__(
        self,
        class_names,
    ):

        self.tracks = []

        self.next_id = 1

        self.class_names = class_names

        self.created = 0

        self.matches = 0

        self.reactivations = 0

        self.frames = 0

    # --------------------------------------------------------
    # CLASS TYPE
    # --------------------------------------------------------

    def class_name(
        self,
        cls,
    ):

        return self.class_names.get(
            cls,
            str(cls),
        )

    def is_person(
        self,
        cls,
    ):

        return (
            self.class_name(cls)
            in PERSON_CLASSES
        )

    # --------------------------------------------------------
    # ASSOCIATION SCORE
    # --------------------------------------------------------

    def association_cost(
        self,
        track,
        detection,
        lost=False,
    ):

        predicted_box = (
            track.predicted_box()
        )

        predicted_center = (
            track.predicted_center()
        )

        det_box = detection["bbox"]

        det_center = detection["center"]

        overlap = box_iou(
            predicted_box,
            det_box,
        )

        distance = float(
            np.linalg.norm(
                predicted_center
                - det_center
            )
        )

        size = box_size(
            predicted_box
        )

        diagonal = max(
            20.0,
            float(
                np.linalg.norm(size)
            ),
        )

        normalized_distance = (
            distance
            / diagonal
        )

        appearance = (
            track.appearance_similarity(
                detection[
                    "appearance"
                ]
            )
        )

        # ----------------------------------------------------
        # PERSONS
        # ----------------------------------------------------

        if self.is_person(
            track.cls
        ):

            # People benefit heavily from appearance because
            # nearby people often have similar motion.
            w_iou = 0.40
            w_center = 0.20
            w_app = 0.40

            center_gate = (
                5.0
                if lost
                else 4.0
            )

            # For a lost person, appearance is allowed to
            # rescue the association.
            if lost:

                if appearance < REID_THRESHOLD:
                    return 1e6

                if normalized_distance > center_gate:
                    return 1e6

            else:

                # Active tracks primarily use spatial continuity.
                if (
                    overlap < MIN_IOU
                    and normalized_distance
                    > center_gate
                ):
                    return 1e6

        # ----------------------------------------------------
        # VEHICLES
        # ----------------------------------------------------

        else:

            w_iou = 0.60
            w_center = 0.25
            w_app = 0.15

            center_gate = (
                4.0
                if lost
                else 3.5
            )

            if lost:

                if (
                    appearance
                    < 0.45
                    and overlap < 0.05
                ):
                    return 1e6

                if normalized_distance > center_gate:
                    return 1e6

            else:

                if (
                    overlap < MIN_IOU
                    and normalized_distance
                    > center_gate
                ):
                    return 1e6

        # ----------------------------------------------------
        # COST
        # ----------------------------------------------------

        iou_cost = (
            1.0 - overlap
        )

        center_cost = min(
            normalized_distance
            / center_gate,
            1.0,
        )

        appearance_cost = (
            1.0 - appearance
        )

        cost = (
            w_iou * iou_cost
            + w_center * center_cost
            + w_app * appearance_cost
        )

        # Very strong overlap should nearly guarantee
        # continuity.
        if overlap >= STRONG_IOU:

            cost *= 0.50

        return cost

    # --------------------------------------------------------
    # CREATE
    # --------------------------------------------------------

    def create_track(
        self,
        detection,
        frame,
        frame_idx,
    ):

        track = Track(
            self.next_id,
            detection,
            frame,
            frame_idx,
        )

        self.tracks.append(
            track
        )

        self.next_id += 1

        self.created += 1

    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    def update(
        self,
        detections,
        frame,
        frame_idx,
    ):

        self.frames += 1

        # ----------------------------------------------------
        # FIRST APPEARANCE EXTRACTION
        # ----------------------------------------------------

        for detection in detections:

            detection[
                "appearance"
            ] = appearance_descriptor(
                frame,
                detection["bbox"],
            )

        # ----------------------------------------------------
        # NO TRACKS
        # ----------------------------------------------------

        if not self.tracks:

            for detection in detections:

                if (
                    detection["conf"]
                    >= CONF_THRESHOLD
                ):

                    self.create_track(
                        detection,
                        frame,
                        frame_idx,
                    )

            return

        # ----------------------------------------------------
        # ACTIVE TRACKS
        # ----------------------------------------------------

        active_indices = [
            i
            for i, t
            in enumerate(self.tracks)
            if t.missed == 0
        ]

        lost_indices = [
            i
            for i, t
            in enumerate(self.tracks)
            if t.missed > 0
            and t.missed <= MAX_MISSED
        ]

        matched_tracks = set()

        matched_detections = set()

        # ====================================================
        # STAGE 1
        #
        # ACTIVE TRACKS
        #
        # Strong spatial continuity.
        # ====================================================

        if active_indices and detections:

            cost = np.full(
                (
                    len(active_indices),
                    len(detections),
                ),
                1e6,
                dtype=np.float32,
            )

            for r, ti in enumerate(
                active_indices
            ):

                track = self.tracks[ti]

                for di, detection in enumerate(
                    detections
                ):

                    if (
                        track.cls
                        != detection["cls"]
                    ):
                        continue

                    cost[r, di] = (
                        self.association_cost(
                            track,
                            detection,
                            lost=False,
                        )
                    )

            rows, cols = (
                linear_sum_assignment(
                    cost
                )
            )

            for r, di in zip(
                rows,
                cols,
            ):

                if cost[r, di] >= 1e5:
                    continue

                ti = active_indices[r]

                self.tracks[ti].update(
                    detections[di],
                    frame_idx,
                )

                matched_tracks.add(
                    ti
                )

                matched_detections.add(
                    di
                )

                self.matches += 1

        # ====================================================
        # STAGE 2
        #
        # RECENTLY LOST TRACKS
        #
        # Re-identification.
        # ====================================================

        remaining_detections = [
            di
            for di in range(
                len(detections)
            )
            if di not in matched_detections
        ]

        available_lost = [
            ti
            for ti in lost_indices
            if ti not in matched_tracks
        ]

        if (
            available_lost
            and remaining_detections
        ):

            cost = np.full(
                (
                    len(available_lost),
                    len(remaining_detections),
                ),
                1e6,
                dtype=np.float32,
            )

            for r, ti in enumerate(
                available_lost
            ):

                track = self.tracks[ti]

                for c, di in enumerate(
                    remaining_detections
                ):

                    detection = detections[
                        di
                    ]

                    if (
                        track.cls
                        != detection["cls"]
                    ):
                        continue

                    cost[r, c] = (
                        self.association_cost(
                            track,
                            detection,
                            lost=True,
                        )
                    )

            rows, cols = (
                linear_sum_assignment(
                    cost
                )
            )

            for r, c in zip(
                rows,
                cols,
            ):

                if cost[r, c] >= 1e5:
                    continue

                ti = available_lost[r]

                di = remaining_detections[c]

                self.tracks[ti].update(
                    detections[di],
                    frame_idx,
                )

                matched_tracks.add(
                    ti
                )

                matched_detections.add(
                    di
                )

                self.matches += 1

                self.reactivations += 1

        # ====================================================
        # STAGE 3
        #
        # UNMATCHED OLD TRACKS
        # ====================================================

        for ti, track in enumerate(
            self.tracks
        ):

            if ti not in matched_tracks:

                track.miss()

        # ====================================================
        # STAGE 4
        #
        # NEW IDs
        #
        # Before creating a new ID, check whether the
        # detection looks like an old identity.
        #
        # This is the important anti-fragmentation step.
        # ====================================================

        for di, detection in enumerate(
            detections
        ):

            if di in matched_detections:
                continue

            if (
                detection["conf"]
                < CONF_THRESHOLD
            ):
                continue

            should_create = True

            # Look at ALL existing tracks, including lost ones.
            for track in self.tracks:

                if (
                    track.cls
                    != detection["cls"]
                ):
                    continue

                # Don't interfere with a track that was
                # already matched this frame.
                if track.missed == 0:
                    continue

                appearance = (
                    track.appearance_similarity(
                        detection[
                            "appearance"
                        ]
                    )
                )

                predicted = (
                    track.predicted_center()
                )

                distance = float(
                    np.linalg.norm(
                        predicted
                        - detection[
                            "center"
                        ]
                    )
                )

                scale = max(
                    30.0,
                    float(
                        np.linalg.norm(
                            box_size(
                                track.bbox
                            )
                        )
                    ),
                )

                # Strong identity evidence:
                # same class + appearance + plausible location.
                if (
                    appearance
                    >= REID_THRESHOLD
                    and distance
                    <= 6.0 * scale
                ):

                    should_create = False

                    break

            if should_create:

                self.create_track(
                    detection,
                    frame,
                    frame_idx,
                )

        # ----------------------------------------------------
        # REMOVE EXPIRED TRACKS
        # ----------------------------------------------------

        self.tracks = [
            t
            for t in self.tracks
            if t.missed
            <= MAX_MISSED
        ]

    # --------------------------------------------------------
    # VISIBLE TRACKS
    # --------------------------------------------------------

    def visible_tracks(self):

        return [
            t
            for t in self.tracks
            if (
                t.visible
                and t.missed == 0
                and t.confirmed
            )
        ]


# ============================================================
# DRAW
# ============================================================

def id_color(track_id):

    # Stable deterministic ID color.
    rng = np.random.default_rng(
        track_id * 123457
    )

    return tuple(
        int(x)
        for x in rng.integers(
            60,
            245,
            3,
        )
    )


def draw_track(
    frame,
    track,
    class_name,
):

    x1, y1, x2, y2 = (
        track.bbox.astype(int)
    )

    color = id_color(
        track.id
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    label = (
        f"{class_name} "
        f"ID:{track.id} "
        f"{track.conf:.2f}"
    )

    (tw, th), baseline = (
        cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )
    )

    ly = max(
        th + 8,
        y1,
    )

    cv2.rectangle(
        frame,
        (
            x1,
            ly - th - 8,
        ),
        (
            x1 + tw + 8,
            ly + baseline,
        ),
        color,
        -1,
    )

    cv2.putText(
        frame,
        label,
        (
            x1 + 4,
            ly - 4,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


# ============================================================
# MAIN
# ============================================================

print("")
print("=" * 70)
print("FINAL PERSISTENT OBJECT TRACKER")
print("=" * 70)
print("")
print("Architecture:")
print("  YOLO")
print("   -> class gating")
print("   -> motion")
print("   -> IoU")
print("   -> center distance")
print("   -> appearance memory")
print("   -> Hungarian assignment")
print("   -> lost-track re-identification")
print("   -> persistent IDs")
print("")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

model = YOLO(
    MODEL_PATH
)

cap = cv2.VideoCapture(
    VIDEO_PATH
)

if not cap.isOpened():

    raise RuntimeError(
        f"Could not open {VIDEO_PATH}"
    )

fps = cap.get(
    cv2.CAP_PROP_FPS
)

width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

print(
    f"Video: {width}x{height}"
)

print(
    f"FPS: {fps:.2f}"
)

print(
    f"Frames: {total_frames}"
)

print(
    f"Track memory: "
    f"{MAX_MISSED} frames"
)

print("")

out = cv2.VideoWriter(
    str(VIDEO_OUTPUT),
    cv2.VideoWriter_fourcc(
        *"mp4v"
    ),
    fps,
    (
        width,
        height,
    ),
)

tracker = PersistentTracker(
    model.names
)

csv_file = open(
    CSV_OUTPUT,
    "w",
    newline="",
)

csv_writer = csv.writer(
    csv_file
)

csv_writer.writerow(
    [
        "frame",
        "timestamp",
        "track_id",
        "class",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
        "velocity_x",
        "velocity_y",
        "age",
        "hits",
        "missed",
        "confirmed",
    ]
)

frame_idx = 0

start_time = time.perf_counter()

total_detections = 0


# ============================================================
# VIDEO LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_idx += 1

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    result = model.predict(
        frame,
        imgsz=IMAGE_SIZE,
        conf=CONF_THRESHOLD,
        device="mps",
        verbose=False,
    )[0]

    detections = []

    if result.boxes is not None:

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        classes = (
            result.boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )

        for box, cls, conf in zip(
            boxes,
            classes,
            confidences,
        ):

            detections.append(
                {
                    "bbox": box.astype(
                        np.float32
                    ),
                    "cls": int(cls),
                    "conf": float(conf),
                    "center": box_center(
                        box
                    ),
                }
            )

    total_detections += len(
        detections
    )

    # --------------------------------------------------------
    # TRACK
    # --------------------------------------------------------

    tracker.update(
        detections,
        frame,
        frame_idx,
    )

    visible = (
        tracker.visible_tracks()
    )

    # --------------------------------------------------------
    # DRAW
    # --------------------------------------------------------

    for track in visible:

        class_name = (
            model.names.get(
                track.cls,
                str(track.cls),
            )
        )

        draw_track(
            frame,
            track,
            class_name,
        )

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        csv_writer.writerow(
            [
                frame_idx,
                frame_idx / fps,
                track.id,
                class_name,
                f"{track.conf:.4f}",
                f"{track.bbox[0]:.2f}",
                f"{track.bbox[1]:.2f}",
                f"{track.bbox[2]:.2f}",
                f"{track.bbox[3]:.2f}",
                f"{track.center[0]:.2f}",
                f"{track.center[1]:.2f}",
                f"{track.velocity[0]:.2f}",
                f"{track.velocity[1]:.2f}",
                track.age,
                track.hits,
                track.missed,
                track.confirmed,
            ]
        )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    cv2.putText(
        frame,
        (
            f"Frame {frame_idx}/{total_frames}  "
            f"Objects:{len(visible)}"
        ),
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    out.write(frame)

    if frame_idx % 100 == 0:

        elapsed = (
            time.perf_counter()
            - start_time
        )

        speed = (
            frame_idx
            / max(elapsed, 1e-6)
        )

        print(
            f"Frame {frame_idx:4d}/"
            f"{total_frames} | "
            f"det={len(detections):2d} | "
            f"visible={len(visible):2d} | "
            f"IDs={tracker.created:4d} | "
            f"reactivated={tracker.reactivations:3d} | "
            f"FPS={speed:.2f}"
        )


# ============================================================
# FINISH
# ============================================================

cap.release()
out.release()
csv_file.close()

elapsed = (
    time.perf_counter()
    - start_time
)

print("")
print("=" * 70)
print("FINAL TRACKER COMPLETE")
print("=" * 70)

print(
    f"Frames processed:       {frame_idx}"
)

print(
    f"Total YOLO detections:  {total_detections}"
)

print(
    f"Average detections/frame: "
    f"{total_detections / max(frame_idx,1):.2f}"
)

print(
    f"Total IDs created:      "
    f"{tracker.created}"
)

print(
    f"Total associations:     "
    f"{tracker.matches}"
)

print(
    f"Lost-track reactivations:"
    f" {tracker.reactivations}"
)

print(
    f"Processing FPS:         "
    f"{frame_idx / max(elapsed,1e-6):.2f}"
)

print("")
print(
    f"VIDEO: {VIDEO_OUTPUT}"
)

print(
    f"CSV:   {CSV_OUTPUT}"
)

print("=" * 70)
