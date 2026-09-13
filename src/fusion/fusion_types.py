"""Typed data contract for asynchronous tracking, depth, segmentation, and lane fusion."""

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class TrackingObservation:
    frame: int
    timestamp: float
    track_id: str
    class_name: str
    bbox: list[float]
    confidence: float
    velocity_x: float
    velocity_y: float
    trajectory: list[list[float]] = field(default_factory=list)


@dataclass
class DepthObservation:
    frame: int
    timestamp: float
    track_id: str
    depth: Optional[float]
    quality: Optional[float]
    depth_change: Optional[float]
    depth_history: list[float] = field(default_factory=list)
    trend: str = "unknown"
    source_frame: Optional[int] = None
    source_age_frames: Optional[int] = None


@dataclass
class SegmentationObservation:
    timestamp: float
    semantic_class: Optional[str] = None
    confidence: Optional[float] = None
    drivable: Optional[bool] = None
    available: bool = False


@dataclass
class LaneObservation:
    timestamp: float
    lane_relation: Optional[str] = None
    lane_center_offset: Optional[float] = None
    lane_crossing: Optional[bool] = None
    lane_conflict: Optional[bool] = None
    confidence: Optional[float] = None
    available: bool = False


@dataclass
class FusedTrackRecord:
    frame: int
    timestamp: float
    track_id: str
    class_name: str
    bbox: list[float]
    velocity: dict[str, float]
    trajectory: list[list[float]]
    depth: Optional[float]
    depth_quality: Optional[float]
    depth_change: Optional[float]
    depth_history: list[float]
    depth_trend: str
    closing_speed: Optional[float]
    ttc: Optional[float]
    ttc_valid: bool
    ttc_reason: str
    lane_relation: Optional[str]
    lane_center_offset: Optional[float]
    lane_crossing: Optional[bool]
    lane_conflict: Optional[bool]
    lane_confidence: Optional[float]
    semantic_context: Optional[str]
    semantic_confidence: Optional[float]
    confidence: float
    risk_inputs: dict[str, Any]
    data_age: dict[str, Optional[float]]
    staleness: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
