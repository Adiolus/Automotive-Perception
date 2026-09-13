"""Optional-input, timestamp-aligned fusion core for tracking and depth."""

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    from .fusion_types import (
        DepthObservation,
        FusedTrackRecord,
        LaneObservation,
        SegmentationObservation,
        TrackingObservation,
    )
except ImportError:
    from fusion_types import (
        DepthObservation,
        FusedTrackRecord,
        LaneObservation,
        SegmentationObservation,
        TrackingObservation,
    )


@dataclass
class FusionConfig:
    depth_fps: float = 30.0
    depth_stale_seconds: float = 1.0
    auxiliary_stale_seconds: float = 0.5
    minimum_closing_speed: float = 1e-3
    maximum_ttc_seconds: float = 120.0
    depth_history_length: int = 8


class FusionEngine:
    """Join timestamped streams without requiring optional streams."""

    def __init__(self, config: FusionConfig):
        self.config = config
        self.depth_by_track: dict[str, list[DepthObservation]] = defaultdict(list)
        self.depth_history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=config.depth_history_length)
        )
        self.segmentation: list[SegmentationObservation] = []
        self.lanes: list[LaneObservation] = []
        self.metrics = {
            "tracking_records": 0,
            "depth_joins": 0,
            "depth_missing": 0,
            "depth_stale": 0,
            "valid_ttc": 0,
            "invalid_ttc": 0,
            "invalid_ttc_reasons": defaultdict(int),
            "segmentation_available": 0,
            "lane_available": 0,
        }
        self.track_ids = set()

    def add_depth(self, observation: DepthObservation) -> None:
        self.depth_by_track[observation.track_id].append(observation)

    def add_segmentation(self, observation: SegmentationObservation) -> None:
        self.segmentation.append(observation)

    def add_lane(self, observation: LaneObservation) -> None:
        self.lanes.append(observation)

    def fuse(self, tracking: TrackingObservation) -> FusedTrackRecord:
        self.metrics["tracking_records"] += 1
        self.track_ids.add(tracking.track_id)
        depth = self._latest_depth(tracking)
        depth_value = depth.depth if depth else None
        depth_quality = depth.quality if depth else None
        depth_change = depth.depth_change if depth else None
        depth_history = list(self.depth_history[tracking.track_id])
        if depth and depth_value is not None and math.isfinite(depth_value):
            self.depth_history[tracking.track_id].append(depth_value)
            depth_history = list(self.depth_history[tracking.track_id])

        closing_speed, ttc, ttc_valid, ttc_reason = self._ttc(
            depth_value,
            depth_change,
            tracking.timestamp,
        )
        segmentation = self._latest_optional(self.segmentation, tracking.timestamp)
        lane = self._latest_optional(self.lanes, tracking.timestamp)
        segmentation_available = bool(segmentation and segmentation.available)
        lane_available = bool(lane and lane.available)
        if segmentation_available:
            self.metrics["segmentation_available"] += 1
        if lane_available:
            self.metrics["lane_available"] += 1

        depth_age = None
        depth_status = "missing"
        if depth:
            depth_age = (
                depth.source_age_frames / self.config.depth_fps
                if depth.source_age_frames is not None
                else max(0.0, tracking.timestamp - depth.timestamp)
            )
            depth_status = (
                "stale"
                if depth_age > self.config.depth_stale_seconds
                else "fresh"
            )
            self.metrics["depth_joins"] += 1
            self.metrics["depth_stale"] += int(depth_status == "stale")
        else:
            self.metrics["depth_missing"] += 1

        confidence_parts = [tracking.confidence]
        if depth_quality is not None and depth_status == "fresh":
            confidence_parts.append(max(0.0, min(1.0, depth_quality)))
        confidence = sum(confidence_parts) / len(confidence_parts)
        speed = math.hypot(tracking.velocity_x, tracking.velocity_y)
        risk_inputs = {
            "depth_available": depth_value is not None,
            "depth_trend": depth.trend if depth else "unknown",
            "approaching": bool(closing_speed is not None and closing_speed > 0),
            "image_velocity_px_per_frame": speed,
            "trajectory_points": len(tracking.trajectory),
            "semantic_available": segmentation_available,
            "lane_available": lane_available,
            "risk_thresholds_applied": False,
        }
        data_age = {
            "depth_seconds": depth_age,
            "segmentation_seconds": self._age(segmentation, tracking.timestamp),
            "lane_seconds": self._age(lane, tracking.timestamp),
        }
        staleness = {
            "depth": depth_status,
            "segmentation": self._optional_status(segmentation, tracking.timestamp),
            "lane": self._optional_status(lane, tracking.timestamp),
        }
        return FusedTrackRecord(
            frame=tracking.frame,
            timestamp=tracking.timestamp,
            track_id=tracking.track_id,
            class_name=tracking.class_name,
            bbox=tracking.bbox,
            velocity={"x": tracking.velocity_x, "y": tracking.velocity_y},
            trajectory=tracking.trajectory,
            depth=depth_value,
            depth_quality=depth_quality,
            depth_change=depth_change,
            depth_history=depth_history,
            depth_trend=depth.trend if depth else "unknown",
            closing_speed=closing_speed,
            ttc=ttc,
            ttc_valid=ttc_valid,
            ttc_reason=ttc_reason,
            lane_relation=lane.lane_relation if lane else None,
            lane_center_offset=lane.lane_center_offset if lane else None,
            lane_crossing=lane.lane_crossing if lane else None,
            lane_conflict=lane.lane_conflict if lane else None,
            lane_confidence=lane.confidence if lane else None,
            semantic_context=segmentation.semantic_class if segmentation else None,
            semantic_confidence=segmentation.confidence if segmentation else None,
            confidence=confidence,
            risk_inputs=risk_inputs,
            data_age=data_age,
            staleness=staleness,
        )

    def _latest_depth(self, tracking: TrackingObservation) -> Optional[DepthObservation]:
        candidates = [
            item
            for item in self.depth_by_track.get(tracking.track_id, [])
            if item.timestamp <= tracking.timestamp + 1e-6
        ]
        return max(candidates, key=lambda item: item.timestamp) if candidates else None

    def _ttc(self, depth, depth_change, timestamp):
        if depth is None or not math.isfinite(depth):
            self.metrics["invalid_ttc"] += 1
            self.metrics["invalid_ttc_reasons"]["missing_depth"] += 1
            return None, None, False, "missing_depth"
        if depth_change is None or not math.isfinite(depth_change):
            self.metrics["invalid_ttc"] += 1
            self.metrics["invalid_ttc_reasons"]["missing_depth_change"] += 1
            return None, None, False, "missing_depth_change"
        if depth_change <= self.config.minimum_closing_speed:
            self.metrics["invalid_ttc"] += 1
            self.metrics["invalid_ttc_reasons"]["not_approaching"] += 1
            return depth_change, None, False, "not_approaching"
        ttc = depth / depth_change
        if not math.isfinite(ttc) or ttc <= 0:
            self.metrics["invalid_ttc"] += 1
            self.metrics["invalid_ttc_reasons"]["invalid_denominator"] += 1
            return depth_change, None, False, "invalid_denominator"
        if ttc > self.config.maximum_ttc_seconds:
            self.metrics["invalid_ttc"] += 1
            self.metrics["invalid_ttc_reasons"]["outside_horizon"] += 1
            return depth_change, None, False, "outside_horizon"
        self.metrics["valid_ttc"] += 1
        return depth_change, ttc, True, "valid_approaching"

    @staticmethod
    def _latest_optional(items, timestamp):
        candidates = [item for item in items if item.timestamp <= timestamp + 1e-6]
        return max(candidates, key=lambda item: item.timestamp) if candidates else None

    def _age(self, item, timestamp):
        if item is None:
            return None
        return max(0.0, timestamp - item.timestamp)

    def _optional_status(self, item, timestamp):
        if item is None or not item.available:
            return "unavailable"
        age = self._age(item, timestamp)
        return "stale" if age > self.config.auxiliary_stale_seconds else "fresh"

    def summary(self) -> dict[str, Any]:
        result = dict(self.metrics)
        result["invalid_ttc_reasons"] = dict(self.metrics["invalid_ttc_reasons"])
        result["timestamp_alignment_success_rate"] = (
            self.metrics["depth_joins"] / max(self.metrics["tracking_records"], 1)
        )
        result["unique_tracks_fused"] = len(self.track_ids)
        result["stale_depth_percentage"] = (
            100.0 * self.metrics["depth_stale"] / max(self.metrics["depth_joins"], 1)
        )
        return result


def parse_config(path: Path) -> FusionConfig:
    values = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = [part.strip() for part in line.split(":", 1)]
        values[key] = float(value)
    return FusionConfig(
        depth_fps=values.get("depth_fps", 30.0),
        depth_stale_seconds=values.get("depth_stale_seconds", 1.0),
        auxiliary_stale_seconds=values.get("auxiliary_stale_seconds", 0.5),
        minimum_closing_speed=values.get("minimum_closing_speed", 1e-3),
        maximum_ttc_seconds=values.get("maximum_ttc_seconds", 120.0),
        depth_history_length=int(values.get("depth_history_length", 8)),
    )


def parse_tracking(row: dict[str, str]) -> TrackingObservation:
    trajectory = []
    raw_trajectory = row.get("trajectory", "")
    for point in raw_trajectory.split("|"):
        if "," not in point:
            continue
        x, y = point.split(",", 1)
        trajectory.append([float(x), float(y)])
    return TrackingObservation(
        frame=int(row["frame"]),
        timestamp=float(row["timestamp"]),
        track_id=str(row["track_id"]),
        class_name=row["class"],
        bbox=[float(row[key]) for key in ("x1", "y1", "x2", "y2")],
        confidence=float(row["confidence"]),
        velocity_x=float(row.get("velocity_x", 0.0)),
        velocity_y=float(row.get("velocity_y", 0.0)),
        trajectory=trajectory,
    )


def parse_depth(row: dict[str, str]) -> DepthObservation:
    def optional_float(key):
        value = row.get(key, "")
        return float(value) if value not in ("", None) else None

    history = []
    previous = optional_float("depth_previous")
    current = optional_float("depth")
    if previous is not None:
        history.append(previous)
    if current is not None:
        history.append(current)
    return DepthObservation(
        frame=int(row["frame"]),
        timestamp=float(row["timestamp"]),
        track_id=str(row["track_id"]),
        depth=current,
        quality=optional_float("depth_quality"),
        depth_change=optional_float("depth_change"),
        depth_history=history,
        trend=row.get("trend", "unknown"),
        source_frame=int(row["depth_frame"]) if row.get("depth_frame") else None,
        source_age_frames=int(row["depth_age_frames"]) if row.get("depth_age_frames") else None,
    )
