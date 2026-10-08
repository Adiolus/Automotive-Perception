from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class LaneDetection:
    points: np.ndarray
    confidence: float = 1.0


@dataclass
class LaneTrack:
    lane_id: int
    curve: np.ndarray
    confidence: float
    age: int = 1
    hits: int = 1
    missed: int = 0


class LaneTracker:
    """
    Lane-specific temporal tracker.

    Guarantees:
      - lane IDs are restricted to 1..max_lanes
      - no arbitrary IDs are created
      - malformed/NaN curves are rejected
      - temporary disappearance does not immediately create a new ID
      - temporal smoothing is applied to sampled lane curves
      - association uses geometry + bottom position + heading
    """

    def __init__(
        self,
        width: int,
        height: int,
        max_lanes: int = 4,
        max_missed: int = 8,
        match_threshold: float = 0.14,
        ema_alpha: float = 0.25,
    ):
        self.width = width
        self.height = height
        self.max_lanes = max_lanes
        self.max_missed = max_missed
        self.match_threshold = match_threshold
        self.ema_alpha = ema_alpha

        self.y_norm = np.linspace(
            0.45,
            0.98,
            24,
            dtype=np.float32,
        )

        self.tracks: List[Optional[LaneTrack]] = [
            None for _ in range(max_lanes)
        ]

        self.total_matches = 0
        self.total_new_tracks = 0

    @staticmethod
    def _finite(points: np.ndarray) -> bool:
        return (
            points is not None
            and len(points) >= 5
            and np.isfinite(points).all()
        )

    def _fit_curve(
        self,
        points: np.ndarray,
    ) -> Optional[np.ndarray]:

        if not self._finite(points):
            return None

        p = np.asarray(
            points,
            dtype=np.float32,
        )

        order = np.argsort(p[:, 1])
        p = p[order]

        y = p[:, 1]
        x = p[:, 0]

        valid = (
            (y >= 0)
            & (y < self.height)
            & (x >= -0.25 * self.width)
            & (x <= 1.25 * self.width)
        )

        x = x[valid]
        y = y[valid]

        if len(x) < 5:
            return None

        yn = y / float(self.height)
        xn = x / float(self.width)

        # Remove duplicate y samples.
        unique_y, unique_idx = np.unique(
            yn,
            return_index=True,
        )

        yn = unique_y
        xn = xn[unique_idx]

        if len(yn) < 5:
            return None

        try:
            coeff = np.polyfit(
                yn,
                xn,
                2,
            )
        except Exception:
            return None

        if not np.isfinite(coeff).all():
            return None

        sampled_x = np.polyval(
            coeff,
            self.y_norm,
        )

        if not np.isfinite(sampled_x).all():
            return None

        sampled_x = np.clip(
            sampled_x,
            0.0,
            1.0,
        )

        return sampled_x.astype(
            np.float32
        )

    def _bottom_x(
        self,
        curve: np.ndarray,
    ) -> float:

        if curve is None:
            return float("nan")

        return float(
            curve[-1]
        )

    def _heading(
        self,
        curve: np.ndarray,
    ) -> float:

        if curve is None or len(curve) < 4:
            return 0.0

        dx = (
            float(curve[-1])
            - float(curve[-5])
        )

        dy = (
            float(self.y_norm[-1])
            - float(self.y_norm[-5])
        )

        return float(
            np.arctan2(
                dx,
                max(abs(dy), 1e-6),
            )
        )

    def _cost(
        self,
        track: LaneTrack,
        detection: LaneDetection,
    ) -> float:

        a = track.curve
        b = detection.points

        curve_dist = float(
            np.mean(
                np.abs(a - b)
            )
        )

        bottom_dist = abs(
            self._bottom_x(a)
            - self._bottom_x(b)
        )

        bottom_dist /= max(
            float(self.width),
            1.0,
        )

        angle_dist = abs(
            self._heading(a)
            - self._heading(b)
        ) / np.pi

        return (
            0.60 * curve_dist
            + 0.25 * bottom_dist
            + 0.15 * angle_dist
        )

    def _prepare(
        self,
        detections,
    ) -> List[LaneDetection]:

        result = []

        for item in detections:

            if isinstance(
                item,
                LaneDetection,
            ):
                points = item.points
                confidence = item.confidence
            else:
                points = np.asarray(
                    item,
                    dtype=np.float32,
                )
                confidence = 1.0

            curve = self._fit_curve(
                points
            )

            if curve is None:
                continue

            result.append(
                LaneDetection(
                    points=curve,
                    confidence=float(
                        np.clip(
                            confidence,
                            0.0,
                            1.0,
                        )
                    ),
                )
            )

        # Left-to-right deterministic ordering.
        result.sort(
            key=lambda d: (
                self._bottom_x(
                    d.points
                )
            )
        )

        return result[: self.max_lanes]

    def _initialize(
        self,
        detections: List[LaneDetection],
    ):

        for track_id in range(
            self.max_lanes
        ):

            if track_id >= len(
                detections
            ):
                break

            d = detections[
                track_id
            ]

            self.tracks[
                track_id
            ] = LaneTrack(
                lane_id=track_id + 1,
                curve=d.points.copy(),
                confidence=d.confidence,
            )

            self.total_new_tracks += 1

    def _expire_empty_slots(self):

        for i, track in enumerate(
            self.tracks
        ):

            if track is None:
                continue

            if track.missed > self.max_missed:
                self.tracks[i] = None

    def update(
        self,
        detections,
    ) -> List[LaneTrack]:

        detections = self._prepare(
            detections
        )

        active_indices = [
            i
            for i, track in enumerate(
                self.tracks
            )
            if track is not None
        ]

        if not active_indices:

            self._initialize(
                detections
            )

            return self.active()

        matched_tracks = set()
        matched_detections = set()

        if detections:

            cost_matrix = np.full(
                (
                    len(active_indices),
                    len(detections),
                ),
                1e6,
                dtype=np.float32,
            )

            for r, slot_idx in enumerate(
                active_indices
            ):

                track = self.tracks[
                    slot_idx
                ]

                assert track is not None

                for c, detection in enumerate(
                    detections
                ):

                    cost_matrix[
                        r, c
                    ] = self._cost(
                        track,
                        detection,
                    )

            rows, cols = (
                linear_sum_assignment(
                    cost_matrix
                )
            )

            for r, c in zip(
                rows,
                cols
            ):

                cost = float(
                    cost_matrix[
                        r, c
                    ]
                )

                if cost > self.match_threshold:
                    continue

                slot_idx = active_indices[
                    r
                ]

                track = self.tracks[
                    slot_idx
                ]

                detection = detections[
                    c
                ]

                assert track is not None

                alpha = self.ema_alpha

                track.curve = (
                    (1.0 - alpha)
                    * track.curve
                    +
                    alpha
                    * detection.points
                ).astype(
                    np.float32
                )

                track.confidence = (
                    (1.0 - alpha)
                    * track.confidence
                    +
                    alpha
                    * detection.confidence
                )

                track.age += 1
                track.hits += 1
                track.missed = 0

                matched_tracks.add(
                    slot_idx
                )

                matched_detections.add(
                    c
                )

                self.total_matches += 1

        # Keep unmatched existing lanes alive temporarily.
        for slot_idx in active_indices:

            if slot_idx in matched_tracks:
                continue

            track = self.tracks[
                slot_idx
            ]

            assert track is not None

            track.age += 1
            track.missed += 1

            # Confidence decays during occlusion,
            # but the physical lane ID survives.
            track.confidence *= 0.88

        self._expire_empty_slots()

        # Fill genuinely empty slots only.
        # Never create more than max_lanes IDs.
        empty_slots = [
            i
            for i, track in enumerate(
                self.tracks
            )
            if track is None
        ]

        unmatched = [
            detections[i]
            for i in range(
                len(detections)
            )
            if i not in matched_detections
        ]

        for slot_idx, detection in zip(
            empty_slots,
            unmatched,
        ):

            self.tracks[
                slot_idx
            ] = LaneTrack(
                lane_id=slot_idx + 1,
                curve=detection.points.copy(),
                confidence=detection.confidence,
            )

            self.total_new_tracks += 1

        return self.active()

    def active(self) -> List[LaneTrack]:

        result = []

        for track in self.tracks:

            if track is None:
                continue

            if track.missed > self.max_missed:
                continue

            if track.hits < 2:
                continue

            if not np.isfinite(
                track.curve
            ).all():
                continue

            result.append(
                track
            )

        return result
