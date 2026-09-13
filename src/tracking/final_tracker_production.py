from ultralytics import YOLO
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import csv
import time
import subprocess
import shutil
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18


# ============================================================
# FINAL TRACKER V2
#
# Persistent identity architecture:
#
#                 YOLO
#                  |
#          +-------+-------+
#          |               |
#      active tracks    detections
#          |               |
#          +-------+-------+
#                  |
#       global Hungarian matching
#                  |
#      +-----------+-----------+
#      |                       |
#   matched                unmatched
#      |                       |
#  update ID           search lost tracks
#                              |
#                       still unmatched?
#                              |
#                       SEARCH ID GALLERY
#                              |
#                   +----------+----------+
#                   |                     |
#               strong match         no match
#                   |                     |
#             RESURRECT ID           NEW ID
#
# IMPORTANT:
#
# A track can disappear completely from the image and still
# retain its identity in the long-term gallery.
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    "runs/detect/runs/detect/"
    "idd_train20-3/weights/best.pt"
)

VIDEO_PATH = "videos/indian_road.mp4"

OUTPUT_DIR = Path(
    "runs/tracking_eval"
)

RAW_OUTPUT = (
    OUTPUT_DIR /
    "FINAL_TRACKER_PRODUCTION.mp4"
)

FINAL_OUTPUT = (
    OUTPUT_DIR /
    "FINAL_TRACKER_PRODUCTION_H264.mp4"
)

CSV_OUTPUT = (
    OUTPUT_DIR /
    "FINAL_TRACKER_PRODUCTION.csv"
)


# ============================================================
# VIDEO / YOLO
# ============================================================

CONF_THRESHOLD = 0.35
IMAGE_SIZE = 960

# Experiment-only input hygiene. The checkpoint contains classes that are
# useful for detection but are not physical traffic objects to track.
TRACKABLE_CLASS_NAMES = {
    "person",
    "rider",
    "motorcycle",
    "bicycle",
    "autorickshaw",
    "car",
    "truck",
    "bus",
    "vehicle_fallback",
}

DUPLICATE_IOU_THRESHOLD = 0.50

OWNERSHIP_IOU_THRESHOLD = 0.50

REID_AMBIGUITY_IOU = 0.20

REID_AMBIGUITY_CENTER_RATIO = 1.5

REID_LOST_CENTER_RATIO = 5.0

ENABLE_REID_EXPERIMENT = False

ENABLE_TRAJECTORY_EXPERIMENT = False

TRAJECTORY_COST_WEIGHT = 0.15


# ============================================================
# LIVE TRACK MEMORY
# ============================================================

# 30 FPS * 1.5 sec.
#
# A normally occluded object stays as an actual live track.
MAX_MISSED = 45


# ============================================================
# LONG-TERM IDENTITY MEMORY
# ============================================================

# Keep archived identities for the whole video.
#
# This is what allows:
#
# ID 21 -> exits frame -> disappears -> returns -> ID 21
#
# instead of:
#
# ID 21 -> exits frame -> new ID 36
#
ARCHIVE_MAX_AGE = 100000


# ============================================================
# REENTRY DETECTION
# ============================================================

# A returning object generally appears near an image border.
ENTRY_MARGIN = 150

# Maximum time between exit and re-entry that we will consider
# for explicit re-identification.
MAX_REENTRY_GAP = 900  # 30 seconds at 30 FPS


# ============================================================
# APPEARANCE THRESHOLDS
# ============================================================

# This must be high enough to prevent two random people from
# stealing each other's IDs.
REID_MIN_SCORE = 0.70

# Best candidate must beat second candidate by this amount
# when there are ambiguous visually similar objects.
REID_MIN_MARGIN = 0.07


# ============================================================
# ASSOCIATION
# ============================================================

MIN_IOU = 0.02

STRONG_IOU = 0.35

MAX_CENTER_RATIO = 4.0


# ============================================================
# CLASSES
# ============================================================

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
# GEOMETRY
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


def box_area(box):

    return max(
        1.0,
        float(box[2] - box[0]),
    ) * max(
        1.0,
        float(box[3] - box[1]),
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

    area_a = box_area(a)

    area_b = box_area(b)

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def suppress_duplicate_detections(detections):

    ordered = sorted(
        detections,
        key=lambda detection: detection["conf"],
        reverse=True,
    )

    kept = []

    for detection in ordered:

        duplicate = any(
            detection["cls"] == previous["cls"]
            and box_iou(
                detection["bbox"],
                previous["bbox"],
            ) >= DUPLICATE_IOU_THRESHOLD
            for previous in kept
        )

        if not duplicate:
            kept.append(detection)

    return kept


# ============================================================
# IMAGE BORDER
# ============================================================

def near_border(box, width, height):

    x1, y1, x2, y2 = box

    return (
        x1 <= ENTRY_MARGIN
        or y1 <= ENTRY_MARGIN
        or x2 >= width - ENTRY_MARGIN
        or y2 >= height - ENTRY_MARGIN
    )


def border_side(box, width, height):

    x1, y1, x2, y2 = box

    distances = {
        "left": x1,
        "right": width - x2,
        "top": y1,
        "bottom": height - y2,
    }

    return min(
        distances,
        key=distances.get,
    )


# ============================================================
# APPEARANCE DESCRIPTOR
# ============================================================

def make_appearance(frame, box):

    height, width = frame.shape[:2]

    x1, y1, x2, y2 = (
        box.astype(int)
    )

    x1 = max(
        0,
        min(width - 1, x1),
    )

    y1 = max(
        0,
        min(height - 1, y1),
    )

    x2 = max(
        x1 + 1,
        min(width, x2),
    )

    y2 = max(
        y1 + 1,
        min(height, y2),
    )

    crop = frame[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:

        return {
            "hist": np.zeros(
                256,
                dtype=np.float32,
            ),
            "grid": np.zeros(
                48,
                dtype=np.float32,
            ),
            "shape": np.zeros(
                96,
                dtype=np.float32,
            ),
        }

    ch, cw = crop.shape[:2]

    # Remove tiny border/background region.
    if ch > 16 and cw > 16:

        mx = max(
            1,
            int(cw * 0.07),
        )

        my = max(
            1,
            int(ch * 0.05),
        )

        crop = crop[
            my:ch-my,
            mx:cw-mx,
        ]

    # --------------------------------------------------------
    # HSV HISTOGRAM
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
    # GRID COLOR SIGNATURE
    # --------------------------------------------------------

    small_hsv = cv2.resize(
        hsv,
        (4, 4),
        interpolation=cv2.INTER_AREA,
    ).astype(
        np.float32
    )

    small_hsv[:, :, 0] /= 180.0
    small_hsv[:, :, 1] /= 255.0
    small_hsv[:, :, 2] /= 255.0

    grid = small_hsv.flatten()

    # --------------------------------------------------------
    # SHAPE/TEXTURE SIGNATURE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.resize(
        gray,
        (12, 8),
        interpolation=cv2.INTER_AREA,
    )

    gray = (
        gray.astype(
            np.float32
        )
        / 255.0
    )

    gray = (
        gray
        - gray.mean()
    )

    norm = np.linalg.norm(
        gray
    )

    if norm > 1e-6:
        gray /= norm

    return {
        "hist": hist.astype(
            np.float32
        ),
        "grid": grid.astype(
            np.float32
        ),
        "shape": gray.flatten()
        .astype(np.float32),
    }


def cosine_similarity(a, b):

    na = np.linalg.norm(a)

    nb = np.linalg.norm(b)

    if na < 1e-8 or nb < 1e-8:
        return 0.0

    value = (
        np.dot(a, b)
        / (na * nb)
    )

    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


def appearance_similarity(
    appearance_a,
    appearance_b,
):

    hist_sim = cosine_similarity(
        appearance_a["hist"],
        appearance_b["hist"],
    )

    grid_sim = cosine_similarity(
        appearance_a["grid"],
        appearance_b["grid"],
    )

    shape_sim = cosine_similarity(
        appearance_a["shape"],
        appearance_b["shape"],
    )

    # Histogram gives broad clothing/vehicle color.
    #
    # Grid gives spatial color layout.
    #
    # Shape gives coarse appearance consistency.
    #
    # Color dominates deliberately because the red-shirt /
    # backpack person is highly distinctive.
    score = (
        0.50 * hist_sim
        + 0.35 * grid_sim
        + 0.15 * shape_sim
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0,
        )
    )


class LightweightEmbedding:

    def __init__(self):

        weights = ResNet18_Weights.DEFAULT

        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        self.preprocess = weights.transforms()

        self.model = resnet18(
            weights=weights
        )

        self.model.fc = torch.nn.Identity()

        self.model = (
            self.model.to(self.device)
            .eval()
        )

        self.inference_count = 0

    def encode(self, frame, boxes):

        crops = []

        height, width = frame.shape[:2]

        for box in boxes:

            x1, y1, x2, y2 = box.astype(int)

            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(x1 + 1, min(width, x2))
            y2 = max(y1 + 1, min(height, y2))

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                crop = frame[0:1, 0:1]

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crops.append(
                self.preprocess(Image.fromarray(rgb))
            )

        if not crops:
            return []

        batch = torch.stack(crops).to(self.device)

        with torch.inference_mode():
            embeddings = self.model(batch)

        embeddings = torch.nn.functional.normalize(
            embeddings,
            dim=1,
        )

        self.inference_count += len(crops)

        return [
            embedding.detach().cpu().numpy().astype(np.float32)
            for embedding in embeddings
        ]


def embedding_similarity(first, second):

    if first is None or second is None:
        return None

    score = float(np.dot(first, second))

    return float(np.clip(score, 0.0, 1.0))


# ============================================================
# TRACK
# ============================================================

class Track:

    def __init__(
        self,
        track_id,
        detection,
        frame_idx,
    ):

        self.id = track_id

        self.cls = detection[
            "cls"
        ]

        self.bbox = detection[
            "bbox"
        ].copy()

        self.center = detection[
            "center"
        ].copy()

        self.velocity = (
            np.zeros(
                2,
                dtype=np.float32,
            )
        )

        self.conf = detection[
            "conf"
        ]

        self.embedding = detection.get(
            "embedding"
        )

        self.age = 1

        self.hits = 1

        self.missed = 0

        self.last_seen = frame_idx

        self.created_frame = frame_idx

        self.visible = True

        self.confirmed = False

        # ----------------------------------------------------
        # LONG TERM APPEARANCE MEMORY
        # ----------------------------------------------------

        self.appearances = [
            detection[
                "appearance"
            ]
        ]

        # ----------------------------------------------------
        # TRAJECTORY
        # ----------------------------------------------------

        self.history = [
            self.center.copy()
        ]

        # ----------------------------------------------------
        # EXIT MEMORY
        # ----------------------------------------------------

        self.last_border_side = None

        self.last_exit_frame = None

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    def predicted_center(self):

        velocity = np.clip(
            self.velocity,
            -120.0,
            120.0,
        )

        return (
            self.center
            + velocity
        )

    def predicted_box(self):

        box = self.bbox.copy()

        box[[0, 2]] += (
            self.velocity[0]
        )

        box[[1, 3]] += (
            self.velocity[1]
        )

        return box

    # --------------------------------------------------------
    # APPEARANCE
    # --------------------------------------------------------

    def best_appearance_similarity(
        self,
        appearance,
    ):

        best = 0.0

        for old in self.appearances:

            score = (
                appearance_similarity(
                    old,
                    appearance,
                )
            )

            best = max(
                best,
                score,
            )

        return best

    def add_appearance(
        self,
        appearance,
    ):

        self.appearances.append(
            appearance
        )

        # Keep a compact gallery.
        if len(
            self.appearances
        ) > 12:

            self.appearances.pop(0)

    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    def update(
        self,
        detection,
        frame_idx,
    ):

        old_center = (
            self.center.copy()
        )

        new_center = (
            detection[
                "center"
            ].copy()
        )

        measured_velocity = (
            new_center
            - old_center
        )

        self.velocity = (
            0.65
            * self.velocity
            + 0.35
            * measured_velocity
        )

        self.bbox = (
            detection[
                "bbox"
            ].copy()
        )

        self.center = (
            new_center
        )

        self.conf = (
            detection[
                "conf"
            ]
        )

        if detection.get("embedding") is not None:
            self.embedding = detection[
                "embedding"
            ]

        self.hits += 1

        self.age += 1

        self.missed = 0

        self.last_seen = (
            frame_idx
        )

        self.visible = True

        if self.hits >= 2:
            self.confirmed = True

        self.add_appearance(
            detection[
                "appearance"
            ]
        )

        self.history.append(
            self.center.copy()
        )

        if len(
            self.history
        ) > 90:

            self.history.pop(0)

    # --------------------------------------------------------
    # MISS
    # --------------------------------------------------------

    def mark_missed(
        self,
        width,
        height,
        frame_idx,
    ):

        # Remember whether the track was approaching an edge.
        predicted = (
            self.predicted_box()
        )

        if near_border(
            predicted,
            width,
            height,
        ):

            self.last_border_side = (
                border_side(
                    predicted,
                    width,
                    height,
                )
            )

            self.last_exit_frame = (
                frame_idx
            )

        self.missed += 1

        self.visible = False

        self.bbox = (
            predicted
        )

        self.center = (
            box_center(
                predicted
            )
        )

        self.age += 1


# ============================================================
# FINAL TRACKER
# ============================================================

class IdentityTracker:

    def __init__(
        self,
        class_names,
        width,
        height,
    ):

        self.class_names = (
            class_names
        )

        self.width = width

        self.height = height

        self.active = []

        # ----------------------------------------------------
        # ARCHIVED IDENTITIES
        #
        # IMPORTANT:
        # These identities survive after live tracks disappear.
        # ----------------------------------------------------

        self.archive = {}

        self.next_id = 1

        self.total_created = 0

        self.total_matches = 0

        self.total_reactivations = 0

        self.total_archive_resurrections = 0

        self.total_new_ids = 0

        self.reid = (
            LightweightEmbedding()
            if ENABLE_REID_EXPERIMENT
            else None
        )

        self.reid_frames = 0

    # --------------------------------------------------------
    # CLASS
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

    def reid_is_needed(self, detections):

        for track in self.active:

            if track.missed > 0 or track.hits < 3:
                continue

            candidates = []
            predicted = track.predicted_box()
            scale = max(
                25.0,
                float(
                    np.linalg.norm(
                        box_size(predicted)
                    )
                ),
            )

            for detection in detections:

                if track.cls != detection["cls"]:
                    continue

                overlap = box_iou(
                    predicted,
                    detection["bbox"],
                )
                distance = float(
                    np.linalg.norm(
                        track.predicted_center()
                        - detection["center"]
                    )
                )

                if (
                    overlap >= MIN_IOU
                    or distance / scale <= MAX_CENTER_RATIO
                ):
                    candidates.append(detection)

            for first_index, first in enumerate(candidates):

                for second in candidates[first_index + 1 :]:

                    if (
                        box_iou(
                            first["bbox"],
                            second["bbox"],
                        ) >= REID_AMBIGUITY_IOU
                    ):
                        return True

        return False

    # --------------------------------------------------------
    # CREATE
    # --------------------------------------------------------

    def create_track(
        self,
        detection,
        frame_idx,
    ):

        track = Track(
            self.next_id,
            detection,
            frame_idx,
        )

        self.active.append(
            track
        )

        self.next_id += 1

        self.total_created += 1

        self.total_new_ids += 1

    # --------------------------------------------------------
    # ASSOCIATION COST
    # --------------------------------------------------------

    def active_cost(
        self,
        track,
        detection,
    ):

        predicted_box = (
            track.predicted_box()
        )

        predicted_center = (
            track.predicted_center()
        )

        overlap = box_iou(
            predicted_box,
            detection[
                "bbox"
            ],
        )

        distance = float(
            np.linalg.norm(
                predicted_center
                - detection[
                    "center"
                ]
            )
        )

        scale = max(
            25.0,
            float(
                np.linalg.norm(
                    box_size(
                        predicted_box
                    )
                )
            ),
        )

        normalized_distance = (
            distance / scale
        )

        measured_velocity = (
            detection["center"]
            - track.center
        )

        expected_velocity = np.clip(
            track.velocity,
            -80.0,
            80.0,
        )

        motion_error = min(
            float(
                np.linalg.norm(
                    measured_velocity
                    - expected_velocity
                )
            )
            / scale,
            3.0,
        )

        appearance = (
            track.best_appearance_similarity(
                detection[
                    "appearance"
                ]
            )
        )

        reid = embedding_similarity(
            track.embedding,
            detection.get("embedding"),
        )

        if (
            overlap < MIN_IOU
            and normalized_distance
            > MAX_CENTER_RATIO
        ):

            return 1e6

        # People:
        # appearance is useful, but geometry remains primary
        # when the object is continuously visible.
        if self.is_person(
            track.cls
        ):

            if ENABLE_TRAJECTORY_EXPERIMENT and reid is None:
                iou_w = 0.35
                center_w = 0.20
                appearance_w = 0.30
                motion_w = TRAJECTORY_COST_WEIGHT
            else:
                iou_w = 0.40
                center_w = 0.20
                appearance_w = 0.20
                motion_w = 0.0
            reid_w = 0.20

        else:

            if ENABLE_TRAJECTORY_EXPERIMENT and reid is None:
                iou_w = 0.45
                center_w = 0.20
                appearance_w = 0.20
                motion_w = TRAJECTORY_COST_WEIGHT
            else:
                iou_w = 0.50
                center_w = 0.20
                appearance_w = 0.10
                motion_w = 0.0
            reid_w = 0.20

        if reid is None:
            reid_w = 0.0
            if not ENABLE_TRAJECTORY_EXPERIMENT:
                iou_w = 0.50 if self.is_person(track.cls) else 0.60
                center_w = 0.25
                appearance_w = 0.25 if self.is_person(track.cls) else 0.15
            motion_w = (
                TRAJECTORY_COST_WEIGHT
                if ENABLE_TRAJECTORY_EXPERIMENT
                else 0.0
            )

        cost = (
            iou_w
            * (1.0 - overlap)
            +
            center_w
            * min(
                normalized_distance
                / MAX_CENTER_RATIO,
                1.0,
            )
            +
            appearance_w
            * (1.0 - appearance)
            + motion_w
            * min(
                motion_error,
                1.0,
            )
            + reid_w
            * (1.0 - (reid or 0.0))
        )

        # Stable overlapping detections get a major bonus.
        if overlap >= STRONG_IOU:

            cost *= 0.45

        return cost

    # --------------------------------------------------------
    # LOST TRACK COST
    # --------------------------------------------------------

    def lost_cost(
        self,
        track,
        detection,
    ):

        predicted_box = (
            track.predicted_box()
        )

        predicted_center = (
            track.predicted_center()
        )

        overlap = box_iou(
            predicted_box,
            detection[
                "bbox"
            ],
        )

        distance = float(
            np.linalg.norm(
                predicted_center
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
                        predicted_box
                    )
                )
            ),
        )

        normalized_distance = (
            distance / scale
        )

        appearance = (
            track.best_appearance_similarity(
                detection[
                    "appearance"
                ]
            )
        )

        reid = embedding_similarity(
            track.embedding,
            detection.get("embedding"),
        )

        # Lost people need strong appearance evidence.
        if self.is_person(
            track.cls
        ):

            if (
                appearance < 0.48
                and overlap < 0.03
                and (reid is None or reid < 0.45)
            ):
                return 1e6

            iou_w = 0.25
            center_w = 0.15
            appearance_w = 0.35
            reid_w = 0.25

        else:

            iou_w = 0.40
            center_w = 0.20
            appearance_w = 0.20
            reid_w = 0.20

        if reid is None:
            reid_w = 0.0
            if not ENABLE_REID_EXPERIMENT:
                if self.is_person(track.cls):
                    iou_w = 0.30
                    center_w = 0.20
                    appearance_w = 0.50
                else:
                    iou_w = 0.45
                    center_w = 0.25
                    appearance_w = 0.30
            else:
                appearance_w += 0.20

        if (
            normalized_distance
            > 7.0
            and appearance
            < 0.72
        ):

            return 1e6

        return (
            iou_w
            * (1.0 - overlap)
            +
            center_w
            * min(
                normalized_distance
                / 7.0,
                1.0,
            )
            +
            appearance_w
            * (1.0 - appearance)
            + reid_w
            * (1.0 - (reid or 0.0))
        )

    # --------------------------------------------------------
    # ARCHIVE SEARCH
    # --------------------------------------------------------

    def archive_candidates(
        self,
        detection,
        frame_idx,
    ):

        candidates = []

        for track_id, info in (
            self.archive.items()
        ):

            track = info[
                "track"
            ]

            if (
                track.cls
                != detection["cls"]
            ):

                continue

            # Don't resurrect extremely old identities.
            if (
                frame_idx
                - track.last_seen
                > MAX_REENTRY_GAP
            ):

                continue

            appearance = (
                track.best_appearance_similarity(
                    detection[
                        "appearance"
                    ]
                )
            )

            reid = embedding_similarity(
                track.embedding,
                detection.get("embedding"),
            )

            if reid is not None:
                appearance = (
                    0.55 * appearance
                    + 0.45 * reid
                )

            if (
                appearance
                < REID_MIN_SCORE
            ):

                continue

            # ------------------------------------------------
            # ENTRY / EXIT GEOMETRY
            # ------------------------------------------------

            current_near_border = (
                near_border(
                    detection[
                        "bbox"
                    ],
                    self.width,
                    self.height,
                )
            )

            # For an object that completely left the frame,
            # re-entry normally happens near a border.
            #
            # If it happens in the middle of the frame, allow
            # it only when appearance is extremely strong.
            if not current_near_border:

                if appearance < 0.82:

                    continue

            # If we know the exit side, use it as a bonus.
            side_bonus = 0.0

            if (
                track.last_border_side
                is not None
                and current_near_border
            ):

                current_side = (
                    border_side(
                        detection[
                            "bbox"
                        ],
                        self.width,
                        self.height,
                    )
                )

                if (
                    current_side
                    == track.last_border_side
                ):

                    side_bonus = 0.08

            # ------------------------------------------------
            # SIZE CONSISTENCY
            # ------------------------------------------------

            old_area = (
                box_area(
                    track.bbox
                )
            )

            new_area = (
                box_area(
                    detection[
                        "bbox"
                    ]
                )
            )

            size_ratio = (
                min(
                    old_area,
                    new_area,
                )
                /
                max(
                    old_area,
                    new_area,
                )
            )

            # Don't reject outright because perspective can
            # change substantially.
            size_score = size_ratio

            # ------------------------------------------------
            # TIME GAP
            # ------------------------------------------------

            gap = (
                frame_idx
                - track.last_seen
            )

            time_score = max(
                0.0,
                1.0
                - (
                    gap
                    / MAX_REENTRY_GAP
                ),
            )

            # ------------------------------------------------
            # FINAL REENTRY SCORE
            # ------------------------------------------------

            score = (
                0.72
                * appearance
                +
                0.16
                * size_score
                +
                0.04
                * time_score
                +
                side_bonus
            )

            candidates.append(
                (
                    score,
                    appearance,
                    track_id,
                )
            )

        candidates.sort(
            reverse=True
        )

        return candidates

    # --------------------------------------------------------
    # REACTIVATE FROM ARCHIVE
    # --------------------------------------------------------

    def resurrect(
        self,
        detection,
        track_id,
        frame_idx,
    ):

        info = self.archive[
            track_id
        ]

        track = info[
            "track"
        ]

        track.update(
            detection,
            frame_idx,
        )

        self.active.append(
            track
        )

        del self.archive[
            track_id
        ]

        self.total_archive_resurrections += 1

    # --------------------------------------------------------
    # UPDATE
    # --------------------------------------------------------

    def update(
        self,
        detections,
        frame,
        frame_idx,
    ):

        # ====================================================
        # APPEARANCE EXTRACTION
        # ====================================================

        for detection in detections:

            detection[
                "appearance"
            ] = make_appearance(
                frame,
                detection[
                    "bbox"
                ],
            )

        if (
            ENABLE_REID_EXPERIMENT
            and detections
            and self.reid_is_needed(detections)
        ):

            embeddings = self.reid.encode(
                frame,
                [detection["bbox"] for detection in detections],
            )

            for detection, embedding in zip(
                detections,
                embeddings,
            ):
                detection["embedding"] = embedding

            self.reid_frames += 1

        matched_active = set()

        matched_detections = set()

        # ====================================================
        # STAGE 1:
        # ACTIVE TRACKS
        # ====================================================

        locked_active = set()
        locked_detections = set()

        # Preserve an unambiguous existing owner before global assignment.
        # This prevents nearly identical boxes from swapping IDs frame to frame.
        ownership_candidates = []

        for ti, track in enumerate(self.active):

            if track.missed > 0:
                continue

            for di, detection in enumerate(detections):

                if track.cls != detection["cls"]:
                    continue

                overlap = box_iou(
                    track.bbox,
                    detection["bbox"],
                )

                if overlap >= OWNERSHIP_IOU_THRESHOLD:
                    ownership_candidates.append(
                        (overlap, ti, di)
                    )

        ownership_candidates.sort(reverse=True)

        for overlap, ti, di in ownership_candidates:

            if ti in locked_active or di in locked_detections:
                continue

            self.active[ti].update(
                detections[di],
                frame_idx,
            )

            locked_active.add(ti)
            locked_detections.add(di)
            matched_active.add(ti)
            matched_detections.add(di)
            self.total_matches += 1

        if self.active and detections:

            remaining_active = [
                ti
                for ti in range(len(self.active))
                if ti not in locked_active
            ]

            remaining_detections = [
                di
                for di in range(len(detections))
                if di not in locked_detections
            ]

            cost = np.full(
                (
                    len(remaining_active),
                    len(remaining_detections),
                ),
                1e6,
                dtype=np.float32,
            )

            for row, ti in enumerate(remaining_active):

                track = self.active[ti]

                for column, di in enumerate(remaining_detections):

                    detection = detections[di]

                    if (
                        track.cls
                        != detection[
                            "cls"
                        ]
                    ):

                        continue

                    cost[row, column] = (
                        self.active_cost(
                            track,
                            detection,
                        )
                    )

            rows, cols = (
                linear_sum_assignment(
                    cost
                )
            )

            for row, column in zip(
                rows,
                cols,
            ):

                if (
                    cost[
                        row,
                        column
                    ] >= 1e5
                ):
                    continue

                ti = remaining_active[row]
                di = remaining_detections[column]

                self.active[ti].update(
                    detections[di],
                    frame_idx,
                )

                matched_active.add(
                    ti
                )

                matched_detections.add(
                    di
                )

                self.total_matches += 1

        # ====================================================
        # STAGE 2:
        # TEMPORARILY LOST LIVE TRACKS
        # ====================================================

        lost_indices = [
            i
            for i, track
            in enumerate(
                self.active
            )
            if (
                i not in matched_active
                and track.missed > 0
            )
        ]

        remaining_detections = [
            di
            for di in range(
                len(detections)
            )
            if di
            not in matched_detections
        ]

        if (
            lost_indices
            and remaining_detections
        ):

            cost = np.full(
                (
                    len(lost_indices),
                    len(remaining_detections),
                ),
                1e6,
                dtype=np.float32,
            )

            for r, ti in enumerate(
                lost_indices
            ):

                track = self.active[
                    ti
                ]

                for c, di in enumerate(
                    remaining_detections
                ):

                    detection = (
                        detections[
                            di
                        ]
                    )

                    if (
                        track.cls
                        != detection[
                            "cls"
                        ]
                    ):

                        continue

                    cost[
                        r,
                        c
                    ] = (
                        self.lost_cost(
                            track,
                            detection,
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

                if (
                    cost[
                        r,
                        c
                    ] >= 1e5
                ):
                    continue

                ti = lost_indices[
                    r
                ]

                di = remaining_detections[
                    c
                ]

                self.active[
                    ti
                ].update(
                    detections[
                        di
                    ],
                    frame_idx,
                )

                matched_active.add(
                    ti
                )

                matched_detections.add(
                    di
                )

                self.total_reactivations += 1

                self.total_matches += 1

        # ====================================================
        # STAGE 3:
        # MARK REMAINING ACTIVE TRACKS LOST
        # ====================================================

        for ti, track in enumerate(
            self.active
        ):

            if ti not in matched_active:

                track.mark_missed(
                    self.width,
                    self.height,
                    frame_idx,
                )

        # ====================================================
        # STAGE 4:
        # MOVE DEAD LIVE TRACKS INTO IDENTITY ARCHIVE
        # ====================================================

        still_active = []

        for track in self.active:

            if (
                track.missed
                <= MAX_MISSED
            ):

                still_active.append(
                    track
                )

            else:

                self.archive[
                    track.id
                ] = {
                    "track": track,
                    "archived_frame":
                        frame_idx,
                }

        self.active = (
            still_active
        )

        # ====================================================
        # STAGE 5:
        # SEARCH LONG-TERM ID ARCHIVE
        #
        # THIS IS THE IMPORTANT NEW PART.
        # ====================================================

        remaining_detections = [
            di
            for di in range(
                len(detections)
            )
            if di
            not in matched_detections
        ]

        archive_matches = []

        for di in remaining_detections:

            detection = (
                detections[
                    di
                ]
            )

            candidates = (
                self.archive_candidates(
                    detection,
                    frame_idx,
                )
            )

            if not candidates:
                continue

            best = candidates[
                0
            ]

            best_score = best[
                0
            ]

            best_id = best[
                2
            ]

            second_score = (
                candidates[1][0]
                if len(
                    candidates
                ) > 1
                else 0.0
            )

            margin = (
                best_score
                - second_score
            )

            # ------------------------------------------------
            # REQUIRE CONFIDENT IDENTITY MATCH
            # ------------------------------------------------

            if (
                best_score
                >= REID_MIN_SCORE
                and (
                    margin
                    >= REID_MIN_MARGIN
                    or best_score
                    >= 0.82
                )
            ):

                archive_matches.append(
                    (
                        best_score,
                        di,
                        best_id,
                    )
                )

        # Highest-confidence archive matches first.
        archive_matches.sort(
            reverse=True
        )

        used_archive_ids = set()

        for (
            score,
            di,
            track_id,
        ) in archive_matches:

            if di in matched_detections:
                continue

            if (
                track_id
                in used_archive_ids
            ):
                continue

            if (
                track_id
                not in self.archive
            ):
                continue

            self.resurrect(
                detections[
                    di
                ],
                track_id,
                frame_idx,
            )

            matched_detections.add(
                di
            )

            used_archive_ids.add(
                track_id
            )

        # ====================================================
        # STAGE 6:
        # CREATE GENUINELY NEW IDs
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

            self.create_track(
                detection,
                frame_idx,
            )

    # --------------------------------------------------------
    # VISIBLE
    # --------------------------------------------------------

    def visible_tracks(self):

        return [
            track
            for track in self.active
            if (
                track.visible
                and track.missed == 0
                and track.confirmed
            )
        ]


# ============================================================
# DRAWING
# ============================================================

def id_color(
    track_id
):

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

    (
        tw,
        th,
    ), baseline = (
        cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )
    )

    label_y = max(
        th + 10,
        y1,
    )

    cv2.rectangle(
        frame,
        (
            x1,
            label_y
            - th
            - 8,
        ),
        (
            x1
            + tw
            + 8,
            label_y
            + baseline,
        ),
        color,
        -1,
    )

    cv2.putText(
        frame,
        label,
        (
            x1 + 4,
            label_y - 4,
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
print("=" * 72)
print("FINAL TRACKER V2 — LONG-TERM IDENTITY")
print("=" * 72)

print("")
print("Features:")
print("  [1] YOLO detection")
print("  [2] Class-aware association")
print("  [3] Motion prediction")
print("  [4] IoU matching")
print("  [5] Center-distance matching")
print("  [6] Appearance memory")
print("  [7] Global Hungarian assignment")
print("  [8] Temporary occlusion recovery")
print("  [9] LONG-TERM ID ARCHIVE")
print("  [10] EXIT -> REENTRY ID RESURRECTION")
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
        f"Could not open video: "
        f"{VIDEO_PATH}"
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
    f"Resolution: "
    f"{width}x{height}"
)

print(
    f"FPS: "
    f"{fps:.2f}"
)

print(
    f"Frames: "
    f"{total_frames}"
)

print(
    f"Live memory: "
    f"{MAX_MISSED} frames"
)

print(
    f"Long-term identity memory: ENABLED"
)

print("")

out = cv2.VideoWriter(
    str(RAW_OUTPUT),
    cv2.VideoWriter_fourcc(
        *"mp4v"
    ),
    fps,
    (
        width,
        height,
    ),
)

if not out.isOpened():

    raise RuntimeError(
        "Could not create video writer."
    )


# ============================================================
# CSV
# ============================================================

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
        "trajectory",
        "age",
        "hits",
        "missed",
        "confirmed",
    ]
)


# ============================================================
# TRACKER
# ============================================================

tracker = IdentityTracker(
    model.names,
    width,
    height,
)


# ============================================================
# LOOP
# ============================================================

frame_idx = 0

total_detections = 0

start_time = time.perf_counter()

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

            class_name = model.names.get(
                int(cls),
                str(cls),
            )

            if class_name not in TRACKABLE_CLASS_NAMES:
                continue

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

    detections = suppress_duplicate_detections(
        detections
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
    # DRAW + CSV
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

        csv_writer.writerow(
            [
                frame_idx,
                f"{frame_idx / fps:.4f}",
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
                "|".join(
                    f"{point[0]:.1f},{point[1]:.1f}"
                    for point in track.history[-15:]
                ),
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
            f"Objects:{len(visible)}  "
            f"IDs:{tracker.total_created}"
        ),
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    out.write(
        frame
    )

    if frame_idx % 100 == 0:

        elapsed = (
            time.perf_counter()
            - start_time
        )

        speed = (
            frame_idx
            / max(
                elapsed,
                1e-6,
            )
        )

        print(
            f"Frame "
            f"{frame_idx:4d}/"
            f"{total_frames} | "
            f"det={len(detections):2d} | "
            f"visible={len(visible):2d} | "
            f"live={len(tracker.active):2d} | "
            f"archive={len(tracker.archive):3d} | "
            f"IDs={tracker.total_created:4d} | "
            f"reID={tracker.total_archive_resurrections:3d} | "
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
print("=" * 72)
print("TRACKER V2 COMPLETE")
print("=" * 72)

print(
    f"Frames processed:           "
    f"{frame_idx}"
)

print(
    f"Total YOLO detections:      "
    f"{total_detections}"
)

print(
    f"Average detections/frame:   "
    f"{total_detections / max(frame_idx, 1):.2f}"
)

print(
    f"Total IDs created:          "
    f"{tracker.total_created}"
)

print(
    f"Total frame associations:   "
    f"{tracker.total_matches}"
)

print(
    f"Temporary reactivations:    "
    f"{tracker.total_reactivations}"
)

print(
    f"LONG-TERM ID resurrections: "
    f"{tracker.total_archive_resurrections}"
)

print(
    f"Processing FPS:             "
    f"{frame_idx / max(elapsed, 1e-6):.2f}"
)

print(
    f"ReID inference crops:       "
    f"{tracker.reid.inference_count if tracker.reid else 0}"
)

print(
    f"ReID frames:                "
    f"{tracker.reid_frames}"
)

print(
    f"ReID frame percentage:      "
    f"{100.0 * tracker.reid_frames / max(frame_idx, 1):.2f}%"
)

print("")
print(
    f"Raw video: "
    f"{RAW_OUTPUT}"
)

print(
    f"CSV: "
    f"{CSV_OUTPUT}"
)

print("")


# ============================================================
# QUICKTIME COMPATIBILITY
# ============================================================

ffmpeg = shutil.which(
    "ffmpeg"
)

if ffmpeg:

    print(
        "Converting to H.264 "
        "for QuickTime..."
    )

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(RAW_OUTPUT),
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(FINAL_OUTPUT),
    ]

    subprocess.run(
        command,
        check=True,
    )

    print("")
    print(
        f"FINAL VIDEO: "
        f"{FINAL_OUTPUT}"
    )

else:

    print(
        "ffmpeg not installed."
    )

    print(
        "Using raw MP4 output."
    )

    FINAL_OUTPUT = RAW_OUTPUT


print("")
print("=" * 72)
print("OPENING FINAL VIDEO")
print("=" * 72)

print(
    f"Video: {FINAL_OUTPUT}"
)

print(
    f"CSV:   {CSV_OUTPUT}"
)

print("=" * 72)
